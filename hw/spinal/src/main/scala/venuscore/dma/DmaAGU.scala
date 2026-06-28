package venuscore.dma

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * DmaAGU
 * 接收来自 Arbiter 的 Command，拆解成多个 DmaMemReq
 */
case class DmaAGU(cfg: DmaConfig) extends Component {
  val io = new Bundle {
    // 输入命令（带 Channel ID tag）
    // StreamArbiterFactory 会自动处理 Payload 聚合
    val cmd_in = slave(Stream(DmaCmd(cfg)))
    val cmd_ch = in UInt (cfg.chIdWidth bits) // 当前命令属于哪个 Channel
    val cmd_is_write = in Bool() // 当前命令是读还是写

    // 输出给 AHB Engine 的请求
    val mem_req = master(Stream(DmaMemReq(cfg)))

    // 状态信号，用于锁定外部 Arbiter
    // 当 AGU 正在处理一个多步(repeat)命令时，busy 拉高
    val busy = out Bool()
  }

  // 寄存器组
  val base_reg = Reg(UInt(cfg.addrWidth bits))
  val length_reg = Reg(UInt(cfg.maxWordWidth bits))
  val repeat_reg = Reg(UInt(cfg.maxRepeat bits))
  val stride_reg = Reg(UInt(cfg.addrWidth bits))

  // 上下文信息
  val ch_id_reg = Reg(UInt(cfg.chIdWidth bits))
  val is_write_reg = Reg(Bool())

  // 状态机
  object State extends SpinalEnum {
    val IDLE, RUN = newElement()
  }

  val state = RegInit(State.IDLE)

  // --------------------------------------------------------
  // 逻辑实现
  // --------------------------------------------------------
  val WORD_BYTES = cfg.dataWidth / 8
  val WORD_SHIFT = log2Up(WORD_BYTES)

  // 默认输出
  io.cmd_in.ready := False
  io.mem_req.valid := False
  io.mem_req.base := base_reg
  io.mem_req.length := length_reg
  io.mem_req.chId := ch_id_reg
  io.mem_req.isWrite := is_write_reg

  // Busy 信号：只要不是 IDLE，或者正在握手进入 RUN，都算 Busy
  io.busy := (state =/= State.IDLE)

  switch(state) {
    is(State.IDLE) {
      io.cmd_in.ready := True

      when(io.cmd_in.valid) {
        // 读通道合并：当 repeat 的步距等于一段长度（字节）时，等价于单段连续读
        val length_bytes =
          (io.cmd_in.payload.length << WORD_SHIFT).resize(cfg.addrWidth bits)
        val stride_is_contig =
          io.cmd_in.payload.stride === length_bytes
        val merge_len_width = cfg.maxWordWidth + cfg.maxRepeat
        val merged_len = io.cmd_in.payload.repeat.resize(merge_len_width bits)
        val can_merge =
          !io.cmd_is_write &&
            (io.cmd_in.payload.length === 1) &&
            (io.cmd_in.payload.repeat > 1) &&
            stride_is_contig

        // 锁存命令参数
        base_reg := io.cmd_in.base
        length_reg :=
          can_merge ? merged_len.resize(cfg.maxWordWidth bits) | io.cmd_in.payload.length
        repeat_reg := can_merge ? U(1, cfg.maxRepeat bits) | io.cmd_in.payload.repeat
        stride_reg := io.cmd_in.payload.stride

        // 锁存元数据
        ch_id_reg := io.cmd_ch
        is_write_reg := io.cmd_is_write

        // 简单的零长检查
        when(io.cmd_in.payload.length === 0 || io.cmd_in.payload.repeat === 0) {
          state := State.IDLE
        } otherwise {
          state := State.RUN
        }
      }
    }

    is(State.RUN) {
      // 发送 MemReq
      io.mem_req.valid := True

      when(io.mem_req.ready) {
        // 一个 Burst 发送完毕
        when(repeat_reg === 1) {
          // 全部完成
          state := State.IDLE
        } otherwise {
          // 准备下一次 Burst
          repeat_reg := repeat_reg - 1
          base_reg := base_reg + stride_reg
          state := State.RUN // 保持 RUN
        }
      }
    }
  }
}


// ==============================
// Verilog Generator
// ==============================
object DmaAGU extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg

  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(DmaAGU(fpgaCfg)).printPruned()
}
