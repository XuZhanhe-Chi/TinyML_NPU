package venuscore.dma

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.ahblite._
import venuscore.config._
import venuscore.common._

/**
 * DmaEngineAhb (Fixed Pipeline Timing, Address/Data Decoupled, Write First-Beat Skid)
 * -------------------------------------------------------------------------------
 * 1. 严格区分 AHB 地址相位和数据相位。
 * 2. 使用 addr_rem 跟踪剩余待发地址数，data_rem 跟踪剩余数据数。
 * 3. 使用 addr_sent / data_done 计数器，判断当前 HREADY 周期是否对应有效数据。
 * 4. 写通道支持 wrData 反压（避免“少写一拍/残留数据”）：
 *    - 写数据来自上游 Stream（例如 OBuf FIFO），valid 可能出现气泡；
 *    - AHB master 不能控制 HREADY，因此必须在发出下一拍地址之前确保后续数据拍一定有数据；
 *    - 当前实现选择“写不做地址相位流水”（地址/数据不重叠），每个 word 两拍完成：
 *      - 拍 N：发地址（HTRANS=NONSEQ）
 *      - 拍 N+1：数据相位，消费 1 个 wrData（从内部 FIFO pop）
 */
case class DmaEngineAhb(cfg: DmaConfig) extends Component {
  val io = new Bundle {
    val ahb = master(AhbLite3Master(cfg.ahbLite3Cfg))
    val memReq = slave(Stream(DmaMemReq(cfg)))
    // 读数据返回，携带 chId
    val rdData = master(Stream(Fragment(Bits(cfg.dataWidth bits))))
    val rdDataChId = out UInt (cfg.chIdWidth bits)
    val wrData = slave(Stream(Bits(cfg.dataWidth bits)))

    // profiling/debug taps（只读）：每个 beat 完成时产生 1 拍脉冲
    val dbg_rd_fire = out(Bool())
    val dbg_wr_fire = out(Bool())
  }

  val WORD_BYTES = cfg.dataWidth / 8

  object AhbState extends SpinalEnum {
    val IDLE, RUN = newElement()
  }

  import AhbState._

  val state = RegInit(IDLE)
  val req_reg = Reg(DmaMemReq(cfg))
  val current_addr = Reg(UInt(cfg.addrWidth bits))

  // 剩余地址/数据计数
  val addr_rem = Reg(UInt(cfg.maxWordWidth bits)) init (0)
  val data_rem = Reg(UInt(cfg.maxWordWidth bits)) init (0)

  // 已发地址个数 / 已完成数据个数
  val addr_sent = Reg(UInt(cfg.maxWordWidth bits)) init (0)
  val data_done = Reg(UInt(cfg.maxWordWidth bits)) init (0)

  // -------------------------------------------------------------------------
  // AHB burst 策略：
  // - 为了兼容简单/定制的 AHB slave（例如 QSPI XIP），读侧也不使用 SEQ burst；
  // - 全部访问统一使用 HBURST=SINGLE + 每拍 HTRANS=NONSEQ；
  // - 这样可避免 slave 仅在 NONSEQ 采样地址、忽略 SEQ 导致“重复读同一地址”这类 silent 错误。
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // 写数据缓冲：用一个小 FIFO 吸收 wrData 反压/气泡
  // -------------------------------------------------------------------------
  val wr_fifo = StreamFifo(Bits(cfg.dataWidth bits), depth = 2)
  // depth 很小，强制用寄存器实现，避免综合推到 BRAM（节约 BRAM 资源）
  // private val SMALL_FIFO_MAX_DEPTH = 32
  // if (2 <= SMALL_FIFO_MAX_DEPTH) {
  //   wr_fifo.logic.ram.addAttribute("syn_ramstyle", "registers")
  // }
  wr_fifo.io.flush := False
  wr_fifo.io.pop.ready := False
  wr_fifo.io.push.payload := io.wrData.payload

  // 写数据接收计数：严格限制每个 memReq 只接收 length 个 word，
  // 避免最后一个 beat 仍保持 ready 导致“多吃一拍”，从而让下一条写回的数据提前进入 FIFO，
  // 引发跨命令的 1-word 偏移（并行 cluster 下更容易暴露）。
  val wr_accept_cnt = Reg(UInt(cfg.maxWordWidth bits)) init (0)

  // -------------------------------------------------------------------------
  // 默认输出赋值（组合逻辑）
  // -------------------------------------------------------------------------
  io.ahb.HADDR := current_addr
  io.ahb.HWRITE := req_reg.isWrite
  io.ahb.HSIZE := B(log2Up(WORD_BYTES), 3 bits) // word 对齐
  // 统一使用 SINGLE，避免 burst/SEQ 语义在部分互联/bridge 上触发兼容性问题。
  io.ahb.HBURST := B"000"
  io.ahb.HPROT := B"0011"
  io.ahb.HMASTLOCK := False

  // 写数据：来自 wr_fifo.pop（无效时输出 0，避免 X 传播）
  io.ahb.HWDATA := B(0, cfg.dataWidth bits)
  when(wr_fifo.io.pop.valid) {
    io.ahb.HWDATA := wr_fifo.io.pop.payload
  }

  // 默认 IDLE：仅在 RUN 且 addr_rem>0 时拉成 NONSEQ/SEQ
  // NOTE: AHB requires address/control to remain stable while `HREADY=0`.
  // 这里保持 HTRANS 只依赖内部寄存器状态（这些状态仅在 HREADY 时更新），
  // 避免 HREADY->HTRANS 的组合路径引入 SoC 互联环路。
  val htrans_calc = Bits(2 bits)
  htrans_calc := B"00"
  io.ahb.HTRANS := htrans_calc

  io.memReq.ready := False
  // io.wrData.ready 在下方统一赋值（避免多处驱动）

  // 读数据接口
  //
  // Read data path:
  // - AHB read data phase is 1-cycle latency; downstream may apply backpressure.
  // - Add a 1-beat skid buffer to avoid dropping HRDATA when `rdData.ready` is low.
  //
  val rd_hold_valid = Reg(Bool()) init (False)
  val rd_hold_data = Reg(Bits(cfg.dataWidth bits)) init (0)
  val rd_hold_last = Reg(Bool()) init (False)
  // 下游 ready 的“稳定采样”：只在 HREADY=1 的周期更新，保证 wait-state 期间控制信号不抖动。
  val rd_ready_q = Reg(Bool()) init (True)

  io.rdData.valid := False
  io.rdData.fragment := B(0, cfg.dataWidth bits)
  io.rdData.last := False
  io.rdDataChId := req_reg.chId

  // 默认无 beat
  io.dbg_rd_fire := False
  io.dbg_wr_fire := False

  // NOTE: Previously this engine assumed `rdData.ready=1` and would drop beats under backpressure.
  // The skid buffer above makes the read path robust for short/occasional bubbles.
  //
  // 注意（AHB wait-state 稳定性）：
  // - `StreamFifo.pop.valid` 可能在 HREADY=0 的周期因为上游 push 而变化；
  // - 若直接用它参与 `HTRANS` 决策，会导致 wait-state 期间 `HTRANS` 抖动，违反 AHB 约束；
  // - 这里用 `wr_avail_q`（仅在 HREADY=1 时采样）来做写侧发起判定，保证控制信号稳定。
  val wr_avail_q = Reg(Bool()) init (False)
  val wr_occ = wr_fifo.io.occupancy

  // 写数据接收条件：只有在 RUN 且当前请求为写，并且还没接收满 length 个 word 时，才允许上游提供 wrData
  val can_accept_more_wrdata =
    (wr_accept_cnt < req_reg.length) && (req_reg.length =/= 0)
  val accept_wrdata =
    (state === RUN) && req_reg.isWrite && can_accept_more_wrdata
  wr_fifo.io.push.valid := accept_wrdata && io.wrData.valid
  io.wrData.ready := accept_wrdata && wr_fifo.io.push.ready

  // -------------------------------------------------------------------------
  // 状态机逻辑
  // -------------------------------------------------------------------------
  switch(state) {
    is(IDLE) {
      io.memReq.ready := True

      when(io.memReq.fire) {
        req_reg := io.memReq.payload
        current_addr := io.memReq.base

        // 初始化计数器
        addr_rem := io.memReq.length
        data_rem := io.memReq.length
        addr_sent := 0
        data_done := 0
        wr_accept_cnt := 0

        state := RUN
      }
    }

    is(RUN) {
      // =====================================================
      // RUN 阶段：地址相位与数据相位（写不做地址相位流水）
      // =====================================================

      // outstanding==1 表示当前拍存在“数据相位”（对应上一拍发出的地址）
      val has_outstanding = addr_sent > data_done

      // Write: keep at most 1 beat outstanding (address/data not overlapped).
      // This is the most conservative policy and avoids any risk of write-data underflow
      // when upstream `wrData` has bubbles.
      val can_issue_addr_write =
        req_reg.isWrite && !has_outstanding && wr_avail_q && (addr_rem > 0)

      // Read response appears in the data phase when `has_outstanding && HREADY`.
      val rd_rsp_bus_valid = has_outstanding && !req_reg.isWrite && io.ahb.HREADY

      // Skid-buffered `rdData` output (hold has priority).
      val rd_out_valid = rd_hold_valid || rd_rsp_bus_valid
      val rd_out_data = rd_hold_valid ? rd_hold_data | io.ahb.HRDATA
      val rd_out_last = rd_hold_valid ? rd_hold_last | (data_rem === 1)

      io.rdData.valid := rd_out_valid
      io.rdData.fragment := rd_out_data
      io.rdData.last := rd_out_last

      // 读侧“轻量”流水：在下游 ready 稳定为 1 的场景下，允许地址相位与数据相位重叠，减少 data_valid 空泡。
      //
      // 关键点：
      // - 为避免 AHB 互联/从设备的 `HREADYOUT<->HTRANS` 组合环路，**不能**让 `HTRANS` 组合依赖 `HREADY`；
      // - 因此这里使用 `rd_ready_q`（仅在 `HREADY=1` 时采样的 ready）做门控，保证 wait-state 期间控制信号稳定。
      val can_issue_addr_read =
        !req_reg.isWrite && (addr_rem > 0) && !rd_hold_valid && (!has_outstanding || rd_ready_q)

      // 1) 地址相位逻辑
      when(addr_rem > 0) {
        when(req_reg.isWrite) {
          htrans_calc := can_issue_addr_write ? B"10" | B"00" // NONSEQ / IDLE
        } otherwise {
          // Read: issue NONSEQ when allowed, otherwise insert IDLE to stall.
          htrans_calc := can_issue_addr_read ? B"10" | B"00"
        }
      }

      // 2) 握手与计数更新 (Handshake)
      when(io.ahb.HREADY) {
        // 仅在 HREADY=1 的周期更新“下一拍是否仍有写数据”，避免 wait-state 期间 HTRANS 抖动；
        // 同时要避免“pop 最后一个 word 后 wr_avail_q 仍为 1”导致发出下一拍地址但写 0。
        when(req_reg.isWrite) {
          val push_fire = wr_fifo.io.push.fire
          val pop_fire = wr_fifo.io.pop.fire
          val occ_s = wr_occ.resize(4 bits).asSInt
          val push_s = push_fire ? S(1, 4 bits) | S(0, 4 bits)
          val pop_s = pop_fire ? S(1, 4 bits) | S(0, 4 bits)
          val occ_next = (occ_s + push_s - pop_s).asUInt
          wr_avail_q := occ_next =/= 0
        }
        when(!req_reg.isWrite) {
          rd_ready_q := io.rdData.ready
        }
        // --- 地址计数器 / 已发地址数 ---
        when(addr_rem > 0) {
          when(req_reg.isWrite ? can_issue_addr_write | can_issue_addr_read) {
            addr_rem := addr_rem - 1
            addr_sent := addr_sent + 1
            current_addr := current_addr + U(WORD_BYTES)
          }
        }

        // --- 数据相位逻辑 (Data Phase) ---
        when(has_outstanding) {
          when(req_reg.isWrite) {
            // 写：消费 1 个 FIFO word
            wr_fifo.io.pop.ready := True
            when(wr_fifo.io.pop.fire) {
              io.dbg_wr_fire := True
              data_rem := data_rem - 1
              data_done := data_done + 1

              when(data_rem === 1) {
                state := IDLE
                addr_sent := 0
                data_done := 0
              }
            }
          } otherwise {
            // Read: capture the bus response, and either bypass to downstream or store into skid buffer.
            when(rd_rsp_bus_valid) {
              io.dbg_rd_fire := True
              data_rem := data_rem - 1
              data_done := data_done + 1

              when(!io.rdData.ready) {
                rd_hold_valid := True
                rd_hold_data := io.ahb.HRDATA
                rd_hold_last := (data_rem === 1)
              }
            }
          }
        }
      }

      // Finish read request only after the last beat is drained (including held beat).
      when(!req_reg.isWrite) {
        val rd_all_bus_done = (addr_rem === 0) && (data_rem === 0) && !has_outstanding
        when(rd_all_bus_done && !rd_hold_valid) {
          state := IDLE
          addr_sent := 0
          data_done := 0
        }
      }
    }
  }

  // 写数据接收计数（仅对写请求生效）
  when(state === RUN && req_reg.isWrite && wr_fifo.io.push.fire) {
    wr_accept_cnt := wr_accept_cnt + 1
  }

  // Drain held read beat.
  when(state === RUN && !req_reg.isWrite && rd_hold_valid && io.rdData.fire) {
    rd_hold_valid := False
  }

  // HTRANS 由寄存器状态驱动；寄存器更新受 HREADY 保护，因此不需要额外 hold 寄存器。
}

object DmaEngineAhb extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new DmaEngineAhb(fpgaCfg)).printPruned()
}
