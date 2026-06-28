package venuscore.top

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.ahblite._
import spinal.lib.bus.amba3.apb._
import venuscore.config._
import venuscore.ctrl._
import venuscore.dma._
import venuscore.cluster._

/**
 * VenusCoreTop
 * ------------
 * NPU 顶层集成模块。
 *
 * 架构层次：
 * 1. CtrlTop：APB3 寄存器、uOP 抓取、任务调度和中断。
 * 2. DmaTopAhb：汇总指令/权重/激活/输出 DMA 请求，通过 AHB-Lite master 访存。
 * 3. ClusterTop：接收 uOP 执行 int8 计算并发起数据搬运请求。
 */
case class VenusCoreTop(cfg: VenusCoreConfig) extends Component {

  val io = new Bundle {
    val apb_s = slave(Apb3(cfg.ctrlCfg.apb3Config))
    val ahb_m = master(AhbLite3Master(cfg.dmaCfg.ahbLite3Cfg))
    val venus_irq = out Bool()
  }
  noIoPrefix()

  val ctrl = CtrlTop(cfg.ctrlCfg)
  val dma = DmaTopAhb(cfg.dmaCfg)
  val clusters = Array.tabulate(cfg.clusterNum) { _ =>
    ClusterTop(cfg.clusterCfg)
  }

  ctrl.io.apb <> io.apb_s
  io.venus_irq := ctrl.io.npu_int
  io.ahb_m <> dma.io.ahb

  ctrl.io.dma_dbg_rd_fire := dma.io.dbg_rd_fire
  ctrl.io.dma_dbg_wr_fire := dma.io.dbg_wr_fire

  dma.io.instr_rd <> ctrl.io.instr_dma

  for (i <- 0 until cfg.clusterNum) {
    val cluster = clusters(i)

    cluster.io.uop_data << ctrl.io.uop_data(i)
    cluster.io.ctrl <> ctrl.io.ctrl(i)

    dma.io.wgt_rd_vec(i) <> cluster.io.wgt_dma_port
    dma.io.ifm_rd_vec(i) <> cluster.io.act_dma_port
    dma.io.ofm_wr_vec(i) <> cluster.io.out_dma_port
  }
}

object VenusCoreTop extends App {
  SpinalConfig(
    targetDirectory = "../../build/rtl",
    oneFilePerComponent = false,
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = LOW),
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new VenusCoreTop(VenusCoreConfig.default))
}
