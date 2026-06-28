package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._

/**
 * ClusterIBuf：三行输入缓存（3-bank，32-bit word）
 *
 * 公开版使用 SpinalHDL Mem 实现本地缓存，便于 Vivado 直接综合。
 *
 * 写侧：
 *   - act_stream（来自 ActDMA 的 32bit 流）
 *   - ctrl.load_start/load_row_id/load_len_words（来自 ClusterCtrl）
 *
 * 读侧：
 *   - rd_port：统一读口
 *     * 标量模式：Conv3x3 / PW1x1 / MatMul 等，一拍读一行
 *     * DW 模式：DW3x3，一拍读三行 (top/mid/bot)
 */
class ClusterIBuf(cfg: ClusterConfig) extends Component {

  // ===========================================================================
  // 参数推导 & 约束检查
  // ===========================================================================
  val word_width = cfg.ibufWordWidth // 32 bits (4 * 8)
  val total_words = cfg.ibufDepth // 3 * line_cap_words
  val addr_width = cfg.ibufAddrWidth // 总地址宽（如有需要）
  val line_cap_words = cfg.ibufLineCapWords // 单行容量（word 数）
  val coord_width = cfg.ibufCoordWidth
  val row_id_width = cfg.ibufRowIdWidth
  val line_addr_width = log2Up(line_cap_words)

  assert(total_words == cfg.ibufLines * line_cap_words,
    s"IBUF total_words($total_words) must equal ibufLines(${cfg.ibufLines}) * line_cap_words($line_cap_words)"
  )

  val io = new Bundle {
    // 写入数据接口 (ActDMA -> ClusterIBuf)
    val act_stream = slave(Stream(Bits(word_width bits)))

    // 控制端口 (来自 ClusterCtrl)
    val ctrl = slave(IBufCtrlPort(cfg))

    // 统一读出接口 (Conv / PW / MatMul / DW)
    val rd_port = slave(IBufRdPort(cfg))
  }
  noIoPrefix()

  // ===========================================================================
  // 配置寄存器：存储当前 tile 尺寸 (供读侧计算 line_index 使用)
  // ===========================================================================
  val w_tile_reg = Reg(UInt(coord_width bits)) init (0)
  val c4_reg = Reg(UInt(coord_width bits)) init (0)

  when(io.ctrl.cfg_valid) {
    w_tile_reg := io.ctrl.cfg_w_tile
    c4_reg := io.ctrl.cfg_c4
  }

  // ===========================================================================
  // 内存体：3 行 * line_cap_words words
  // ===========================================================================
  val mem_banks = Array.fill(cfg.ibufLines) {
    Mem(Bits(word_width bits), line_cap_words)
  }

  // ===========================================================================
  // 写入控制 FSM：load_start + load_len_words + act_stream
  // ===========================================================================
  val load_busy_reg = Reg(Bool()) init (False)
  val load_done_reg = Reg(Bool()) init (False)
  val wr_row_id_reg = Reg(UInt(row_id_width bits)) init (0)
  val wr_index_reg = Reg(UInt(line_addr_width bits)) init (0) // 行内 index [0..line_cap_words)
  val wr_remain_reg = Reg(UInt(line_addr_width bits)) init (0) // 剩余待写 word 数

  // 默认：load_done 只打一拍
  load_done_reg := False

  // act_stream 只有在 load_busy 期间才会被消费
  io.act_stream.ready := load_busy_reg

  // 对外状态输出
  io.ctrl.load_busy := load_busy_reg
  io.ctrl.load_done := load_done_reg

  // 启动一轮加载
  when(io.ctrl.load_start && !load_busy_reg) {
    load_busy_reg := True
    load_done_reg := False
    wr_row_id_reg := io.ctrl.load_row_id
    wr_index_reg := 0
    wr_remain_reg := io.ctrl.load_len_words.resize(line_addr_width bits)

    if (GenerationFlags.simulation) {
      assert(io.ctrl.load_len_words <= line_cap_words,
        "ClusterIBuf Load Error: load_len_words exceeds line_cap_words")
      assert(io.ctrl.load_row_id < cfg.ibufLines,
        "ClusterIBuf Load Error: Invalid load_row_id")
    }
  }

  when(io.act_stream.fire && load_busy_reg) {
    // 写入当前 word 到对应行的 bank
    switch(wr_row_id_reg) {
      is(U(0, row_id_width bits)) {
        mem_banks(0).write(
          address = wr_index_reg,
          data = io.act_stream.payload
        )
      }
      is(U(1, row_id_width bits)) {
        mem_banks(1).write(
          address = wr_index_reg,
          data = io.act_stream.payload
        )
      }
      is(U(2, row_id_width bits)) {
        mem_banks(2).write(
          address = wr_index_reg,
          data = io.act_stream.payload
        )
      }
    }

    // index / remain 更新
    wr_index_reg := wr_index_reg + 1
    wr_remain_reg := wr_remain_reg - 1

    // 最后一个 word 写完
    when(wr_remain_reg === 1) {
      load_busy_reg := False
      load_done_reg := True
    }
  }

  // ===========================================================================
  // 读出逻辑：统一 line_index + 三 bank 并行读
  //   - 标量模式：从三 bank 中选一行输出 rd_data_top（= rd_data）
  //   - DW 模式：同一 line_index 下输出三行 rd_data_top/mid/bot
  // ===========================================================================
  val rd_req = io.rd_port.rd_en
  val rd_dw_mode = io.rd_port.rd_dw_mode

  // 公共 line_index = c_group * w_tile + x
  val rd_line_index =
    (io.rd_port.rd_cgroup * w_tile_reg).resize(line_addr_width bits) +
      io.rd_port.rd_x.resize(line_addr_width bits)

  if (GenerationFlags.simulation) {
    when(rd_req) {
      assert(rd_line_index < line_cap_words,
        "ClusterIBuf Read Error: rd_line_index exceeded line capacity")
      assert(io.rd_port.rd_row_id < cfg.ibufLines,
        "ClusterIBuf Read Error: rd_row_id out of range")
      when(rd_dw_mode) {
        assert(io.rd_port.rd_row_top < cfg.ibufLines &&
          io.rd_port.rd_row_mid < cfg.ibufLines &&
          io.rd_port.rd_row_bot < cfg.ibufLines,
          "ClusterIBuf Read Error: rd_row_top/mid/bot out of range")
      }
    }
  }

  // 三 bank 同地址并行读
  val bank_addr = rd_line_index
  val bank_en = rd_req

  val bank_rdata = Vec(Bits(word_width bits), cfg.ibufLines)
  for (i <- 0 until cfg.ibufLines) {
    bank_rdata(i) := mem_banks(i).readSync(
      address = bank_addr,
      enable = bank_en
    )
  }

  // 对齐一拍：模式与行号
  val rd_req_reg = RegNext(rd_req) init (False)
  val rd_dw_mode_reg = RegNext(rd_dw_mode) init (False)
  val rd_row_id_reg2 = RegNext(io.rd_port.rd_row_id) init (0)
  val rd_row_top_reg = RegNext(io.rd_port.rd_row_top) init (0)
  val rd_row_mid_reg = RegNext(io.rd_port.rd_row_mid) init (0)
  val rd_row_bot_reg = RegNext(io.rd_port.rd_row_bot) init (0)

  // 辅助函数：根据 row_id 选择对应 bank 的数据
  def select_bank_data(row_id: UInt): Bits = {
    val data = Bits(word_width bits)
    data := bank_rdata(0)
    switch(row_id) {
      is(U(0, row_id_width bits)) {
        data := bank_rdata(0)
      }
      is(U(1, row_id_width bits)) {
        data := bank_rdata(1)
      }
      is(U(2, row_id_width bits)) {
        data := bank_rdata(2)
      }
    }
    data
  }

  // 标量模式：rd_data_top（= rd_data）输出单行
  val rd_data_scalar = select_bank_data(rd_row_id_reg2)

  // DW 模式：三个行的输出
  val rd_data_top = select_bank_data(rd_row_top_reg)
  val rd_data_mid = select_bank_data(rd_row_mid_reg)
  val rd_data_bot = select_bank_data(rd_row_bot_reg)

  // 统一 valid：上一拍有 rd_en，则本拍 rd_valid = 1
  io.rd_port.rd_valid := rd_req_reg
  io.rd_port.rd_data_top := Mux(rd_dw_mode_reg, rd_data_top, rd_data_scalar)
  io.rd_port.rd_data_mid := rd_data_mid
  io.rd_port.rd_data_bot := rd_data_bot
}

// ===========================================================================
// Verilog 生成器
// ===========================================================================
object ClusterIBuf extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = false,
    headerWithDate = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new ClusterIBuf(VenusCoreConfig.default.clusterCfg)).printPruned()
}
