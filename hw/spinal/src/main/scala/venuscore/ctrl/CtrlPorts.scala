package venuscore.ctrl


import spinal.core._
import spinal.lib._
import venuscore.cluster._
import venuscore.common._
import venuscore.config._


/**
 * NpuCtrlBus
 * ----------
 * NPU 顶层控制信号束，对应：
 *   - NPU_CMD_PTR
 *   - NPU_CTRL
 */
case class NpuCtrlBus(cfg: CtrlConfig) extends Bundle {
  // 控制脉冲与模式
  val start = Bool() // 启动指令链
  val abort = Bool() // 中止当前执行
  val soft_reset = Bool() // 软复位
  val cfg_mode = Bits(4 bits) // 模式配置字段
}

/**
 * NpuStatusBus
 * ------------
 * NPU 状态信号束，对应：
 *   - NPU_STATUS
 */
case class NpuStatusBus() extends Bundle {
  val busy = Bool() // BUSY 标志
  val error = Bool() // ERROR 标志
  val curr_opcode = ClusterUopOpcode() // 当前指令 op_code
  val error_code = Bits(8 bits) // 错误码
}

/**
 * NpuIntCtrlBus
 * -------------
 * NPU 中断控制信号束，对应：
 *   - NPU_INT_ENABLE
 */
case class NpuIntCtrlBus() extends Bundle {
  val done_int_en = Bool() // DONE 中断使能
  val error_int_en = Bool() // ERROR 中断使能
}

/**
 * NpuIntRawBus
 * ------------
 * NPU 中断原始事件输入信号束，由内部 VenusCoreCtrl 提供。
 */
case class NpuIntRawBus() extends Bundle {
  val done_int_raw = Bool() // DONE 事件原始脉冲
  val error_int_raw = Bool() // ERROR 事件原始脉冲
}

/**
 * NpuIntStatusBus
 * ---------------
 * NPU 中断状态输出信号束，对应：
 *   - NPU_INT_STATUS（经 W1C 之后的保持状态）
 */
case class NpuIntStatusBus() extends Bundle {
  val done_int_status = Bool() // DONE 中断状态
  val error_int_status = Bool() // ERROR 中断状态
}


/**
 * NpuDebugBus
 * -----------
 * 调试与版本信息信号束，对应：
 *   - NPU_VERSION
 *   - NPU_DEBUG0
 *   - NPU_DEBUG1
 */
case class NpuDebugBus() extends Bundle {
  val version = Bits(32 bits) // 版本信息
  val debug0 = Bits(32 bits) // 调试信号 0（建议用于周期计数等 profiling）
  val debug1 = Bits(32 bits) // 调试信号 1（建议用于状态快照/死锁定位）
  val debug2 = Bits(32 bits) // 调试信号 2（profiling 扩展）
  val debug3 = Bits(32 bits) // 调试信号 3（profiling 扩展）
  val debug4 = Bits(32 bits) // 调试信号 4（profiling 扩展）
  val debug5 = Bits(32 bits) // 调试信号 5（profiling 扩展）
  val debug6 = Bits(32 bits) // 调试信号 6（profiling 扩展）
  val debug7 = Bits(32 bits) // 调试信号 7（profiling 扩展）
  val debug8 = Bits(32 bits) // 调试信号 8（profiling 扩展）
}

