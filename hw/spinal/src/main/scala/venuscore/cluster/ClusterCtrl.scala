package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * ClusterCtrl（Cluster 控制器）
 * -----------
 * 单个 Cluster 的顶层控制器（控制平面）。
 *
 * 职责：
 * 1) 接收上层解码后的 ClusterUop，锁存本 tile 的几何/地址配置；
 * 2) 预计算 IBuf / WBuf / DMA 所需的几何参数、stride、长度等，尽量寄存器化以优化时序；
 * 3) 为 ActDMA / WgtDMA / OutDMA 插入一级 Stream pipeline，隔离外部总线时序；
 * 4) 通过 MacDpCtrlPort / IBufCtrlPort / WBufCtrlPort / SfuCtrlPort / OBufCtrlPort
 * 驱动下游 IBuf / WBuf / MAC-DP / SFU / OBuf 子模块。
   *
   * 代码结构：
   * - ClusterScheduler.scala：调度器（tile 级状态机）
   * - ClusterFrontend.scala ：前端（uOP 锁存/解码 + tile 初始化）
   * - ClusterBackend.scala  ：后端（循环/DMA/buffer/compute 控制）
   */
class ClusterCtrl(val cfg: ClusterConfig) extends Component {

  // =========================================================
  // 顶层 IO
  // =========================================================
  val io = new Bundle {
    // 上层发来的 uOP / 控制
    val uop_data = slave(Stream(ClusterUop(cfg)))
    val ctrl = slave(ClusterCtrlPort(cfg))

    // DMA 命令端口（读激活 / 读权重 / 写输出）
    val act_dma_cmd = master(Stream(DmaCmd(cfg.dmaConfig)))
    val wgt_dma_cmd = master(Stream(DmaCmd(cfg.dmaConfig)))
    val out_dma_cmd = master(Stream(DmaCmd(cfg.dmaConfig)))

    // IBuf / WBuf / MacDp 控制端口
    val ibuf_ctrl = master(IBufCtrlPort(cfg))
    val wbuf_ctrl = master(WBufCtrlPort(cfg))
    val mac_dp_ctrl = master(MacDpCtrlPort(cfg))

    // SFU / OBuf 控制端口
    val sfu_ctrl = master(SfuCtrlPort(cfg))
    val obuf_ctrl = master(OBufCtrlPort(cfg))
  }
  noIoPrefix()

  // 内部与 DMA 外部总线之间插入一级 Stream pipeline：改善时序
  val act_dma_stream = Stream(DmaCmd(cfg.dmaConfig))
  val wgt_dma_stream = Stream(DmaCmd(cfg.dmaConfig))
  val out_dma_stream = Stream(DmaCmd(cfg.dmaConfig))

  // 注意：这里必须用“寄存器 stage”，而不是 fall-through FIFO：
  // - LOAD_* 控制需要在第一个 beat 返回前就能拉起（避免读通路持续反压）
  // - 切断 cluster/DMA 边界的组合路径，利于时序收敛
  io.act_dma_cmd << act_dma_stream.stage()
  io.wgt_dma_cmd << wgt_dma_stream.stage()
  io.out_dma_cmd << out_dma_stream.stage()

  // 顶层控制简写
  def cl_enable = io.ctrl.cl_enable
  def cl_flush = io.ctrl.cl_flush

  // =========================================================
  // 一、全局状态与配置寄存器
  // =========================================================

  /**
   * 集群控制状态机：按 tile 执行流程划分
   *
   * IDLE      : 空闲，等待 compute uOP
   * LOAD_WGT  : 加载当前 tile 的权重 + 量化参数到 WBuf
   * LOAD_ACT  : 加载当前输出行所需的 IFM 行到 IBuf
   * COMPUTE   : MAC-DP 完成当前 tile 全部行的 MAC
   * SFU_DRAIN : SFU 将 MAC 结果 drain 到 OBuf
   * DRAIN_OBUF: OBuf drain 到 OutDMA，写回外部 OFM
   * DONE      : 单条 uOP 完成，向上层报告
   * ERROR     : 保留，用于异常处理（当前未使用）
   */
  object ClState extends SpinalEnum {
    val IDLE, LOAD_WGT, LOAD_ACT, COMPUTE, SFU_DRAIN, DRAIN_OBUF, DONE, ERROR =
      newElement()
  }

  val state_reg = Reg(ClState()) init (ClState.IDLE)

  // ---- uOP 寄存器（整个 tile 生命周期内生效）----
  val uop_reg = Reg(ClusterUop(cfg)) init (ClusterUop.resetValue(cfg))

  case class TileCfgReg(cfg: ClusterConfig) extends Bundle {
    val h_tile = UInt(8 bits)
    val w_tile = UInt(8 bits)
    val c4_in = UInt(10 bits)
    val c4_out = UInt(10 bits)
    val y_index = UInt(10 bits)
    val stride = UInt(2 bits)
    val pad_top = Bool()
    val pad_bot = Bool()
    val pad_left = Bool()
    val pad_right = Bool()
    val kernel_taps = UInt(4 bits)
    val fo_stride = UInt(16 bits)
  }

  val tile_cfg_reg = Reg(TileCfgReg(cfg))
  tile_cfg_reg.h_tile init (0)
  tile_cfg_reg.w_tile init (0)
  tile_cfg_reg.c4_in init (0)
  tile_cfg_reg.c4_out init (0)
  tile_cfg_reg.y_index init (0)
  tile_cfg_reg.stride init (1)
  tile_cfg_reg.pad_top init (False)
  tile_cfg_reg.pad_bot init (False)
  tile_cfg_reg.pad_left init (False)
  tile_cfg_reg.pad_right init (False)
  tile_cfg_reg.kernel_taps init (1)
  tile_cfg_reg.fo_stride init (0)

  val cfg_valid_reg = Reg(Bool()) init (False)

  // ---- IBUF/WBUF 复用（key + valid + reuse 标志）----
  private val actKeyWidth =
    uop_reg.precalc.fi_addr_tile.getWidth +
      uop_reg.precalc.fi_stride_bytes.getWidth +
      uop_reg.precalc.ibuf_line_words.getWidth +
      uop_reg.c4_in.getWidth +
      uop_reg.stride.getWidth +
      tile_cfg_reg.kernel_taps.getWidth +
      4 + // top/bot/left/right pad
      uop_reg.opcode.asBits.getWidth
  private val wgtKeyWidth =
    uop_reg.coe_addr.getWidth +
      uop_reg.w_addr.getWidth +
      uop_reg.c4_in.getWidth +
      uop_reg.c4_out.getWidth +
      tile_cfg_reg.kernel_taps.getWidth +
      uop_reg.qmode.asBits.getWidth +
      uop_reg.opcode.asBits.getWidth

  val act_key_reg = Reg(Bits(actKeyWidth bits)) init (0)
  val act_key_pending_reg = Reg(Bits(actKeyWidth bits)) init (0)
  val act_valid_reg = Reg(Bool()) init (False)
  val act_reuse_reg = Reg(Bool()) init (False)
  val act_reuse_next = Bool()

  val wgt_key_reg = Reg(Bits(wgtKeyWidth bits)) init (0)
  val wgt_key_pending_reg = Reg(Bits(wgtKeyWidth bits)) init (0)
  val wgt_valid_reg = Reg(Bool()) init (False)
  val wgt_reuse_reg = Reg(Bool()) init (False)
  val wgt_reuse_next = Bool()

  when(!cl_enable || cl_flush) {
    act_valid_reg := False
    wgt_valid_reg := False
    act_reuse_reg := False
    wgt_reuse_reg := False
  }

  // ---- 行循环 / IBUF 行映射 ----
  val y_out_reg = Reg(UInt(8 bits)) init (0)
  val rowid_top_reg = Reg(UInt(cfg.ibufRowIdWidth bits)) init (0)
  val rowid_mid_reg = Reg(UInt(cfg.ibufRowIdWidth bits)) init (1)
  val rowid_bot_reg = Reg(UInt(cfg.ibufRowIdWidth bits)) init (2)

  // ---- PW IBUF 乒乓（ping-pong）----
  val pw_rd_bank_reg = Reg(Bool()) init (False)
  val pw_bank_valid_reg = Vec(Reg(Bool()) init (False), 2)
  val pw_bank_row_reg = Vec(Reg(UInt(8 bits)) init (0), 2)

  // ---- 行/通道偏移累加寄存器（减少乘法器）----
  val pw_row_offset_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val out_row_offset_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val avg_row0_offset_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  // AVGPOOL：当前行 row0/row1 的基址（用于减少 LOAD_ACT 组合路径）
  val avg_row0_base_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val avg_row1_base_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)

  // AVGPOOL 行预取（由 Backend/ActDMA 驱动）：
  // - 当前行需要 row0/row1（IBUF: rowid_mid/rowid_bot）
  // - 预取下一行的 row0 暂存到 rowid_top，并在 advance_row_pulse 时旋转到 rowid_mid
  // 这些寄存器名也会被回归 TB/脚本通过 XMR 读取，因此保持稳定命名。
  val avg_row0_ready_reg = Reg(Bool()) init (False)
  val avg_next_row0_prefetched_reg = Reg(Bool()) init (False)
  val avg_next_row0_y_out_reg = Reg(UInt(8 bits)) init (0)

  // ---- CONV/DW ActDMA 行地址（去乘法版）----
  val s2_row0_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val roll_bot_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val seq_row_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)

  // ---- Area 间通信 Pulse ----
  val tile_start_pulse = Bool()
  val tile_is_avg_pulse = Bool()
  tile_is_avg_pulse := False

  // ---- 各阶段完成标志（供 FSM 使用）----
  val wgt_load_done = Bool()
  val act_load_done = Bool()
  val compute_done = Bool()
  val row_sfu_done = Bool()
  // 行推进脉冲：每行只允许 y_out++ 一次。
  // 由 ClusterScheduler 在决定“离开 SFU_DRAIN 进入下一行”时发出 1 拍 strobe。
  val advance_row_pulse = Bool()
  val drain_done = Bool()
  val is_last_row = Bool()
  val pw_next_row_ready = Bool()
  // AVGPOOL / rolling(DW/CONV) 的预取状态（由 Backend/ActDMA 驱动）。
  val avg_next_row_ready = Bool()
  val avg_pf_inflight = Bool()
  val roll_next_row_ready = Bool()
  val roll_pf_inflight = Bool()

  // =========================================================
  // 预计算几何参数（由 CtrlUopFetch 统一推导并写入 uop.precalc）
  // =========================================================
  val in_row_pixels = uop_reg.precalc.in_row_pixels
  val in_row_bytes = uop_reg.precalc.in_row_bytes
  val ibuf_line_words = uop_reg.precalc.ibuf_line_words
  val fi_stride_byte = uop_reg.precalc.fi_stride_bytes
  val row_out_words = uop_reg.precalc.row_out_words
  val row_stride_bytes = uop_reg.precalc.row_stride_bytes
  val fo_plane_stride_bytes = uop_reg.precalc.fo_plane_stride_bytes
  val total_kernel_cnt = uop_reg.precalc.total_kernel_cnt

  val ibuf_valid_width = in_row_pixels.resize(cfg.ibufCoordWidth bits)
  val pw_stride_row_bytes =
    (tile_cfg_reg.stride === U(2)) ?
      (in_row_bytes |<< 1) |
      in_row_bytes

  // 基于 uop_reg 的算子类型判断
  val is_conv_uop = uop_reg.opcode === ClusterUopOpcode.CONV2D
  val is_pw_uop = uop_reg.opcode === ClusterUopOpcode.PWCONV
  val is_dw_uop = uop_reg.opcode === ClusterUopOpcode.DWCONV
  val is_avg_uop = uop_reg.opcode === ClusterUopOpcode.AVGPOOL
  val is_max_uop = uop_reg.opcode === ClusterUopOpcode.MAXPOOL
  val is_pool_uop = is_avg_uop || is_max_uop

  // 3x3 卷积类是否启用 rolling IBUF（目前只支持 stride=1）
  val use_rolling =
    (is_conv_uop || is_dw_uop) && (tile_cfg_reg.stride === U(1))

  // Kernel 尺寸解码（1x1 / 2x2 / 3x3）
  val kernel_w = UInt(3 bits)
  val kernel_h = UInt(3 bits)
  kernel_w := 1
  kernel_h := 1
  switch(tile_cfg_reg.kernel_taps) {
    is(U(4)) {
      kernel_w := 2
      kernel_h := 2
    }
    is(U(9)) {
      kernel_w := 3
      kernel_h := 3
    }
  }
  val kernel_h_minus1 = (kernel_h - 1).resize(3 bits)
  val kernel_h_minus2 = (kernel_h - 2).resize(3 bits)

  // =========================================================
  // 模块组合（各功能拆分在独立 .scala 文件中实现）
  // =========================================================
  val scheduler = new ClusterScheduler(this)
  val frontend = new ClusterFrontend(this)
  val backend = new ClusterBackend(this)
}

// ==============================
// Verilog 生成入口
// ==============================
object ClusterCtrl extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = false,
    headerWithDate = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(
    new ClusterCtrl(VenusCoreConfig.default.clusterCfg)
  ).printPruned()
}
