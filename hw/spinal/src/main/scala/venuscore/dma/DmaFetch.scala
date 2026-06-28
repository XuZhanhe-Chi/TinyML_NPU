package venuscore.dma

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * DmaFetch
 * ---------------------------------------------------------------------
 * 作用：DMA 读通道的“接口适配 + 轻量缓冲”。
 *
 * 典型连接方式：
 * - 上游：`DmaFetchPort`（cmd 由上游控制下发，data 返回到上游）。
 * - 下游：`ag_cmd/ag_rsp`（连接到集中式 `DmaAGU` + `DmaEngine`）。
 *
 * 设计要点：
 * - `cmd`：插入 1-deep `stage()`，保证在 backpressure 下 payload 稳定，同时隔离组合 ready 路径；
 * - `data`：插入 1-deep `stage()`，吸收 load 状态机偶发的 ready bubble，避免对 read data path 形成“持续反压”。
 */
case class DmaFetch(cfg: DmaConfig) extends Component {
  val io = new Bundle {
    val port = slave(DmaFetchPort(cfg))
    // 与内部 AGU/Hub 交互的接口
    val ag_cmd = master(Stream(DmaCmd(cfg)))
    val ag_rsp = slave(Stream(Bits(cfg.dataWidth bits)))
  }
  noIoPrefix()

  // 命令侧：1 拍寄存器（保持 payload 稳定 + 切断组合 ready）
  io.ag_cmd << io.port.cmd.stage()

  // 读返回侧：1 拍寄存器（吸收短暂反压，不承担长时间 backpressure）
  io.port.data << io.ag_rsp.stage()
}

// ==============================
// Verilog Generator
// ==============================
object DmaFetch extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg

  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(DmaFetch(fpgaCfg)).printPruned()
}
