package venuscore.ctrl

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb._
import spinal.lib.bus.regif._
import spinal.lib.bus.regif.AccessType._
import venuscore.config._

/**
 * CtrlRegBlock
 * ------------
 * VenusCore 顶层控制寄存器块（APB3 + RegIf 实现）。
 *
 * 寄存器地址映射（相对 NPU_REG_BASE 偏移）：
 *
 * 控制 / 状态 / 版本：
 * 0x00 : NPU_UOP_BASE    (RW)   uOP 表基地址（字节地址，对齐由软件保证）
 * 0x04 : NPU_CTRL        (RW)   启动 / 中止 / 软复位 / 模式配置
 * 0x08 : NPU_STATUS      (RO)   忙 / 错误 / 当前 op_code / 错误码
 * 0x0C : NPU_VERSION     (RO)   硬件版本号 / 功能位图
 * 0x10 : NPU_UOP_COUNT   (RW)   本次执行的有效 uOP 数（从 UOP_BASE 起算）
 * 0x14–0x1C : 保留（控制/状态扩展）
 *
 * 中断控制：
 * 0x20 : NPU_INT_ENABLE  (RW)   中断使能
 * 0x24 : NPU_INT_STATUS  (RW1C) 中断状态（写 1 清除）
 * 0x28–0x3C : 保留（中断扩展）
 *
 * 调试 / Profiling：
 * 0x80 : NPU_DEBUG0      (RO)   运行周期计数（从 busy 拉起开始计数，busy 期间每拍 +1）
 * 0x84 : NPU_DEBUG1      (RO)   状态快照（用于死锁定位，字段由 CtrlTop 统一打包）
 * 0x88 : NPU_DEBUG2      (RO)   CLUSTER_BUSY_CYCLES：busy=1 且任意 cluster busy 时每拍 +1
 * 0x8C : NPU_DEBUG3      (RO)   STALL_UOPFETCH_CYCLES：busy=1 且 uop_fetch 等待数据时每拍 +1
 * 0x90 : NPU_DEBUG4      (RO)   STALL_DMA_OR_BUF_CYCLES：busy=1 且等待数据/片上 buffer 就绪时每拍 +1
 * 0x94 : NPU_DEBUG5      (RO)   DMA_RD_BYTES：任务期间 DMA 读累计字节数
 * 0x98 : NPU_DEBUG6      (RO)   DMA_WR_BYTES：任务期间 DMA 写累计字节数
 * 0x9C : NPU_DEBUG7      (RO)   UOP_EXEC_CNT：任务期间实际下发/执行的 uOP 条数
 * 0xA0 : NPU_DEBUG8      (RO)   TILE_DONE_CNT：任务期间 cluster done 事件计数（多 cluster 会累加）
 * 0xA4 : NPU_DEBUG_CTRL  (RW1P) bit0 写 1 清零所有 debug counters（自清）
 * 0xA8–0xFC : 保留（调试扩展）
 */
case class CtrlRegBlock(cfg: CtrlConfig) extends Component {

  val io = new Bundle {
    // APB3 从接口
    val apb = slave(Apb3(cfg.apb3Config))

    // NPU 顶层控制信号
    val uop_base = out(UInt(cfg.addrWidth bits)) // NPU_UOP_BASE
    val uop_count = out(UInt(16 bits)) // NPU_UOP_COUNT（有效 uOP 数）

    val npu_ctrl = out(NpuCtrlBus(cfg))
    val npu_status = in(NpuStatusBus())

    // 中断相关信号
    val npu_int_ctrl = out(NpuIntCtrlBus())
    val npu_int_raw = in(NpuIntRawBus())
    val npu_int_status = out(NpuIntStatusBus())

    // 调试 / 版本信息
    val npu_debug = in(NpuDebugBus())

    // Debug counters 清零控制（写 1 脉冲）
    val debug_clr = out(Bool())
  }.setName("")

  noIoPrefix()

  // RegIf 接口，分配 256 Byte 寄存器窗口
  val busif = Apb3BusInterface(io.apb, (0x0000, 256 Byte))

  // ===========================================================================
  // 0x00 : NPU_UOP_BASE (RW)
  // ===========================================================================
  val npu_uop_base_reg = busif.newRegAt(
    address = 0x00,
    doc = "NPU_UOP_BASE: uOP 表基地址（字节地址）"
  )
  val npu_uop_base_field = npu_uop_base_reg.field(
    UInt(32 bits),
    RW,
    doc = "uOP 表基地址（字节地址，对齐由软件保证）"
  )

  // 截取为 cfg.addrWidth 位宽
  io.uop_base := npu_uop_base_field.resize(io.uop_base.getWidth)

  // ===========================================================================
  // 0x04 : NPU_CTRL (RW) - START / ABORT / SOFT_RESET / CFG_MODE
  // ===========================================================================
  val npu_ctrl_reg = busif.newRegAt(
    address = 0x04,
    doc = "NPU_CTRL: 启动 / 中止 / 软复位 / 模式配置"
  )

  // bit0: START (写 1 产生脉冲)
  val start_field = npu_ctrl_reg.field(
    Bool(),
    W1P,
    doc = "启动 VenusCore uOP 执行（写 1 产生一拍脉冲）"
  )

  // bit1: ABORT (写 1 产生脉冲)
  val abort_field = npu_ctrl_reg.field(
    Bool(),
    W1P,
    doc = "中止当前执行（写 1 产生一拍脉冲）"
  )

  // bit2: SOFT_RESET (写 1 产生脉冲)
  val soft_reset_field = npu_ctrl_reg.field(
    Bool(),
    W1P,
    doc = "软复位 VenusCoreCtrl 内部状态机（写 1 产生一拍脉冲）"
  )

  // bit3: 保留
  npu_ctrl_reg.reserved(1 bits)

  // bits[7:4]: CFG_MODE
  val cfg_mode_field = npu_ctrl_reg.field(
    4 bits,
    RW,
    doc = "模式/调试配置字段（当前版本软件写 0）"
  )

  // bits[31:8]: 保留
  npu_ctrl_reg.reserved(24 bits)

  // 连接到顶层控制信号束
  io.npu_ctrl.start := start_field
  io.npu_ctrl.abort := abort_field
  io.npu_ctrl.soft_reset := soft_reset_field
  io.npu_ctrl.cfg_mode := cfg_mode_field

  // ===========================================================================
  // 0x08 : NPU_STATUS (RO)
  // ===========================================================================
  val npu_status_reg = busif.newRegAt(
    address = 0x08,
    doc = "NPU_STATUS: 忙 / 错误 / 当前 op_code / 错误码"
  )

  val busy_field = npu_status_reg.field(
    Bool(),
    RO,
    doc = "BUSY 标志：1 表示正在执行 uOP 序列"
  )
  val error_field = npu_status_reg.field(
    Bool(),
    RO,
    doc = "ERROR 标志：1 表示内部检测到错误"
  )

  // bits[3:2] 保留
  npu_status_reg.reserved(2 bits)

  val curr_opcode_field = npu_status_reg.field(
    4 bits,
    RO,
    doc = "当前执行 uOP 所在 Layer 的 op_code（与 ISA 定义一致）"
  )
  val error_code_field = npu_status_reg.field(
    8 bits,
    RO,
    doc = "错误码（由内部硬件定义）"
  )

  // bits[31:16] 保留
  npu_status_reg.reserved(16 bits)

  // 从核心状态信号采样
  busy_field := io.npu_status.busy
  error_field := io.npu_status.error
  curr_opcode_field := io.npu_status.curr_opcode.asBits.resized
  error_code_field := io.npu_status.error_code

  // ===========================================================================
  // 0x0C : NPU_VERSION (RO)
  // ===========================================================================
  val npu_version_reg = busif.newRegAt(
    address = 0x0C,
    doc = "NPU_VERSION: 硬件版本号 / 功能位图"
  )
  val version_field = npu_version_reg.field(
    Bits(32 bits),
    RO,
    doc = "版本信息，由硬件给出"
  )

  version_field := io.npu_debug.version

  // ===========================================================================
  // 0x10 : NPU_UOP_COUNT (RW)
  // ===========================================================================
  val npu_uop_count_reg = busif.newRegAt(
    address = 0x10,
    doc = "NPU_UOP_COUNT: 本次执行的有效 uOP 数"
  )
  val npu_uop_count_field = npu_uop_count_reg.field(
    UInt(16 bits),
    RW,
    doc = "从 NPU_UOP_BASE 起连续执行的 uOP 数量"
  )

  io.uop_count := npu_uop_count_field

  // 0x14–0x1C 保留，不创建寄存器（访问行为未定义）

  // ===========================================================================
  // 0x20 : NPU_INT_ENABLE (RW)
  // ===========================================================================
  val npu_int_en_reg = busif.newRegAt(
    address = 0x20,
    doc = "NPU_INT_ENABLE: 中断使能"
  )
  val done_int_en_field = npu_int_en_reg.field(
    Bool(),
    RW,
    doc = "DONE 中断使能"
  )
  val error_int_en_field = npu_int_en_reg.field(
    Bool(),
    RW,
    doc = "ERROR 中断使能"
  )
  npu_int_en_reg.reserved(30 bits)

  io.npu_int_ctrl.done_int_en := done_int_en_field
  io.npu_int_ctrl.error_int_en := error_int_en_field

  // ===========================================================================
  // 0x24 : NPU_INT_STATUS (RW1C)
  // ===========================================================================
  val npu_int_status_reg = busif.newRegAt(
    address = 0x24,
    doc = "NPU_INT_STATUS: 中断状态（写 1 清除）"
  )
  val done_int_status_field = npu_int_status_reg.field(
    Bool(),
    W1C,
    doc = "DONE 中断状态（写 1 清除）"
  )
  val error_int_status_field = npu_int_status_reg.field(
    Bool(),
    W1C,
    doc = "ERROR 中断状态（写 1 清除）"
  )
  npu_int_status_reg.reserved(30 bits)

  // 中断状态寄存器自置位逻辑：
  //   - DONE：有原始事件且已使能时置位
  //   - ERROR：有原始事件且已使能时置位
  done_int_status_field.setWhen(io.npu_int_raw.done_int_raw && done_int_en_field)
  error_int_status_field.setWhen(io.npu_int_raw.error_int_raw && error_int_en_field)

  // 输出到顶层中断状态束
  io.npu_int_status.done_int_status := done_int_status_field
  io.npu_int_status.error_int_status := error_int_status_field

  // 0x28–0x3C 保留，不创建寄存器（访问行为未定义）

  // ===========================================================================
  // 0x80 : NPU_DEBUG0 (RO)
  // ===========================================================================
  val npu_debug0_reg = busif.newRegAt(
    address = 0x80,
    doc = "NPU_DEBUG0: 运行周期计数（profiling）"
  )
  val debug0_field = npu_debug0_reg.field(
    Bits(32 bits),
    RO,
    doc = "调试信号 0（建议用于周期计数等 profiling；由 CtrlTop 驱动）"
  )
  debug0_field := io.npu_debug.debug0

  // ===========================================================================
  // 0x84 : NPU_DEBUG1 (RO)
  // ===========================================================================
  val npu_debug1_reg = busif.newRegAt(
    address = 0x84,
    doc = "NPU_DEBUG1: 状态快照（死锁定位）"
  )
  val debug1_field = npu_debug1_reg.field(
    Bits(32 bits),
    RO,
    doc = "调试信号 1：{error_code,curr_opcode,sched_state,fetch_state,flags}（由 CtrlTop 驱动）"
  )
  debug1_field := io.npu_debug.debug1

  // ===========================================================================
  // 0x88 : NPU_DEBUG2 (RO)
  // ===========================================================================
  val npu_debug2_reg = busif.newRegAt(
    address = 0x88,
    doc = "NPU_DEBUG2: CLUSTER_BUSY_CYCLES（任务期间 cluster busy 周期计数）"
  )
  npu_debug2_reg.field(Bits(32 bits), RO, doc = "busy=1 且任意 cluster busy 时每拍 +1（由 CtrlTop 驱动）") := io.npu_debug.debug2

  // ===========================================================================
  // 0x8C : NPU_DEBUG3 (RO)
  // ===========================================================================
  val npu_debug3_reg = busif.newRegAt(
    address = 0x8C,
    doc = "NPU_DEBUG3: STALL_UOPFETCH_CYCLES（uOP fetch 等待数据/握手周期计数）"
  )
  npu_debug3_reg.field(Bits(32 bits), RO, doc = "busy=1 且 uop_fetch 等待数据时每拍 +1（由 CtrlTop 驱动）") := io.npu_debug.debug3

  // ===========================================================================
  // 0x90 : NPU_DEBUG4 (RO)
  // ===========================================================================
  val npu_debug4_reg = busif.newRegAt(
    address = 0x90,
    doc = "NPU_DEBUG4: STALL_DMA_OR_BUF_CYCLES（等待 DMA/片上 buffer 就绪周期计数）"
  )
  npu_debug4_reg.field(Bits(32 bits), RO, doc = "busy=1 且等待数据/片上 buffer 就绪时每拍 +1（由 CtrlTop 驱动）") := io.npu_debug.debug4

  // ===========================================================================
  // 0x94 : NPU_DEBUG5 (RO)
  // ===========================================================================
  val npu_debug5_reg = busif.newRegAt(
    address = 0x94,
    doc = "NPU_DEBUG5: DMA_RD_BYTES（DMA 读累计字节数）"
  )
  npu_debug5_reg.field(Bits(32 bits), RO, doc = "任务期间 DMA 读每个 beat 累加字节数（由 CtrlTop 驱动）") := io.npu_debug.debug5

  // ===========================================================================
  // 0x98 : NPU_DEBUG6 (RO)
  // ===========================================================================
  val npu_debug6_reg = busif.newRegAt(
    address = 0x98,
    doc = "NPU_DEBUG6: DMA_WR_BYTES（DMA 写累计字节数）"
  )
  npu_debug6_reg.field(Bits(32 bits), RO, doc = "任务期间 DMA 写每个 beat 累加字节数（由 CtrlTop 驱动）") := io.npu_debug.debug6

  // ===========================================================================
  // 0x9C : NPU_DEBUG7 (RO)
  // ===========================================================================
  val npu_debug7_reg = busif.newRegAt(
    address = 0x9C,
    doc = "NPU_DEBUG7: UOP_EXEC_CNT（实际执行/下发到调度器的 uOP 数）"
  )
  npu_debug7_reg.field(Bits(32 bits), RO, doc = "任务期间实际下发/执行的 uOP 条数（由 CtrlTop 驱动）") := io.npu_debug.debug7

  // ===========================================================================
  // 0xA0 : NPU_DEBUG8 (RO)
  // ===========================================================================
  val npu_debug8_reg = busif.newRegAt(
    address = 0xA0,
    doc = "NPU_DEBUG8: TILE_DONE_CNT（cluster done 事件计数）"
  )
  npu_debug8_reg.field(Bits(32 bits), RO, doc = "任务期间 cluster done 事件计数（多 cluster 会累加）（由 CtrlTop 驱动）") := io.npu_debug.debug8

  // ===========================================================================
  // 0xA4 : NPU_DEBUG_CTRL (RW1P)
  // ===========================================================================
  val npu_debug_ctrl_reg = busif.newRegAt(
    address = 0xA4,
    doc = "NPU_DEBUG_CTRL: debug counters 清零控制（写 1 脉冲）"
  )
  val clr_counters_field = npu_debug_ctrl_reg.field(
    Bool(),
    W1P,
    doc = "CLR_COUNTERS：写 1 清零所有 debug counters（自清）"
  )
  npu_debug_ctrl_reg.reserved(31 bits)
  io.debug_clr := clr_counters_field

  // 0xA8–0xFC 保留，不创建寄存器（访问行为未定义）

  // 生成 RegIf 对应的 HTML 文档（可选）
  busif.accept(DocHtml("VenusCore CtrlRegBlock Register File"))
}

object CtrlRegBlock {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl",
      oneFilePerComponent = false,
      enumPrefixEnable = false,
      headerWithDate = false,
      anonymSignalPrefix = "",
      defaultConfigForClockDomains = ClockDomainConfig(
        resetKind = SYNC,
        resetActiveLevel = LOW
      )
    ).generateVerilog(CtrlRegBlock(VenusCoreConfig.default.ctrlCfg))
  }
}
