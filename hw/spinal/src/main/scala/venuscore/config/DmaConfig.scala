package venuscore.config

import spinal.core._
import spinal.lib.bus.amba3.ahblite._

/**
 * DmaConfig
 * ---------------------------------------------------------------------
 * 作用：集中式 DMA 子系统配置。
 *
 * 说明：
 * - `maxWord`：一次 burst 的最大 word 数（TB/AGU/Engine 的上界约束）。
 * - `clusterNum`：用于计算 DMA 通道数与 ID 宽度（Fetch/Store/Param/Wgt/...）。
 */
case class DmaConfig(
  addrWidth: Int,
  dataWidth: Int,
  maxWord: Int,
  clusterNum: Int
) {

  val maxWordWidth: Int = log2Up(maxWord)
  val chIdWidth: Int = log2Up(3 * clusterNum + 1)
  val maxRepeat: Int = log2Up(maxWord)
  val wordBytes: Int = dataWidth / 8

  def ahbLite3Cfg = AhbLite3Config(addrWidth, dataWidth)
}
