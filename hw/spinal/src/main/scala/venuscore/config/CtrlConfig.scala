package venuscore.config

import spinal.core._
import spinal.lib.bus.amba3.apb._

/**
 * CtrlConfig
 * ---------------------------------------------------------------------
 * 作用：控制器（Ctrl）子系统配置。
 *
 * 覆盖内容：
 * - APB3 寄存器窗口参数（地址宽度/数据宽度）；
 * - 与 DMA/Cluster 的互连所需公共参数引用（保持同源，避免参数漂移）。
 */
case class CtrlConfig(
  dmaConfig: DmaConfig,
  clusterConfig: ClusterConfig,
  addrWidth: Int = 12,
  clusterNum: Int = 1,
  regAddrWidth: Int = 12, // APB 地址宽度（对应 4KB 寄存器窗口）
  apbDataWidth: Int = 32 // APB 数据宽度（固定 32bit）
) {

  // APB3 配置
  def apb3Config: Apb3Config = Apb3Config(
    addressWidth = regAddrWidth,
    dataWidth = apbDataWidth,
    selWidth = 1,
    useSlaveError = false
  )

}
