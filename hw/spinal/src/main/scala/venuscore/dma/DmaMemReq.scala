package venuscore.dma

import spinal.core._
import venuscore.config._


// 内部交互用的内存请求（拆解后的 Burst）
case class DmaMemReq(cfg: DmaConfig) extends Bundle {
  val base = UInt(cfg.addrWidth bits)
  val length = UInt(cfg.maxWordWidth bits)
  val isWrite = Bool()
  val chId = UInt(cfg.chIdWidth bits)
}