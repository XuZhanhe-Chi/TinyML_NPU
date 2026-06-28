package venuscore.dma

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * DmaStore
 * ---------------------------------------------------------------------
 * 作用：DMA 写通道的“接口适配 + 可选轻量缓冲”。
 *
 * 设计取舍（资源/时序优先）：
 * - 这里使用 `queue(0)`（本质为直连，不额外引入 FIFO 资源），只保留 Stream 语义；
 * - 若后续发现写侧对 ready 的组合路径过长，可在不改变功能的前提下升级为 `stage()` 或 1-deep queue。
 */
case class DmaStore(cfg: DmaConfig) extends Component {
  val io = new Bundle {
    val port = slave(DmaStorePort(cfg))
    // 与内部 AGU/Hub 交互的接口
    val ag_cmd = master(Stream(DmaCmd(cfg)))
    val ag_data = master(Stream(Bits(cfg.dataWidth bits)))
  }
  noIoPrefix()
  io.ag_cmd << io.port.cmd.queue(0)
  io.ag_data << io.port.data.queue(0) // 写数据通道
}

// ==============================
// Verilog Generator
// ==============================
object DmaStore extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg

  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(DmaStore(fpgaCfg)).printPruned()
}
