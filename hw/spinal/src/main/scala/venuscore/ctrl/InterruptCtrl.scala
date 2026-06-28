package venuscore.ctrl

import spinal.core._
import spinal.lib._
import venuscore.config._

/**
 * InterruptCtrl
 * -------------
 * VenusCore 中断控制器。
 *
 * 功能：
 * 1. 接收来自 RegBlock 的中断状态 (Status) 和 使能 (Enable)。
 * 2. 汇总生成最终的 IRQ 信号给 CPU。
 * 3. (可选) 支持中断极性配置（默认高电平有效）。
 *
 * 连接关系：
 * CtrlRegBlock (Status/Enable) --> InterruptCtrl --> CPU (npu_int)
 */
case class InterruptCtrl(cfg: CtrlConfig) extends Component {
  val io = new Bundle {
    // 来自 RegBlock 的状态与使能
    val int_status = in(NpuIntStatusBus())
    val int_en = in(NpuIntCtrlBus())

    // 输出到 SoC 的中断线
    val npu_int = out Bool()
  }

  noIoPrefix()

  // ==========================================================================
  // 中断汇总逻辑
  // ==========================================================================

  // 逻辑：IRQ = (Status & Enable)
  // 虽然你的 CtrlRegBlock 中 Status 的置位已经 gate 了 Enable，
  // 但这里再次 AND 是标准的做法，允许软件通过暂时拉低 Enable 来屏蔽中断线，
  // 而不清除 Status 寄存器的内容（用于轮询模式或临界区保护）。

  val done_irq = io.int_status.done_int_status && io.int_en.done_int_en
  val error_irq = io.int_status.error_int_status && io.int_en.error_int_en

  // 汇总所有中断源
  val irq_combined = done_irq || error_irq

  // ==========================================================================
  // 输出驱动
  // ==========================================================================
  // 假设高电平有效 (Active High)
  io.npu_int := irq_combined

  // 如果需要支持 Active Low，可以在 Config 中加参数或在此反转
  // io.npu_int := !irq_combined
}