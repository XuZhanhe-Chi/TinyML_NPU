package venuscore.top

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.ahblite._
import spinal.lib.bus.amba3.apb._

/**
 * VenusCoreTop BlackBox
 * ---------------------
 * 这是一个独立的黑盒定义，不依赖 NPU 的 Config 文件。
 * * 默认参数对应 VenusCoreConfig.default 配置:
 * - AHB Address: 32 bit
 * - AHB Data:    32 bit
 * - APB Address: 12 bit (4KB 空间)
 * - APB Data:    32 bit
 */
class VenusCoreTopBB extends BlackBox {
  // 1. 锁定 Verilog 模块名
  setDefinitionName("VenusCoreTop")

  // 2. 在这里直接定义总线配置，不依赖外部 case class
  //    如果你的 NPU 参数变了，记得通知 SoC 同学修改这里
  val ahbConfig = AhbLite3Config(addressWidth = 32, dataWidth = 32)
  val apbConfig = Apb3Config(addressWidth = 12, dataWidth = 32, selWidth = 1, useSlaveError = false)

  val io = new Bundle {
    // ----------------------------------------
    // SoC Control Interface (APB3 Slave)
    // ----------------------------------------
    val apb_s = slave(Apb3(apbConfig))

    // ----------------------------------------
    // DMA Interface (AHB-Lite Master)
    // ----------------------------------------
    val ahb_m = master(AhbLite3Master(ahbConfig))

    // ----------------------------------------
    // Interrupt
    // ----------------------------------------
    val venus_irq = out Bool()
  }

  // 3. 去除 IO 前缀，匹配生成的 Verilog 端口命名 (apb_s_PADDR 等)
  noIoPrefix()

}

class VenusCoreTopBBWrapper extends Component {
  setDefinitionName("VenusCoreTopBB")

  val ahbConfig = AhbLite3Config(addressWidth = 32, dataWidth = 32)
  val apbConfig = Apb3Config(addressWidth = 12, dataWidth = 32, selWidth = 1, useSlaveError = false)

  val io = new Bundle {
    val apb_s = slave(Apb3(apbConfig))
    val ahb_m = master(AhbLite3Master(ahbConfig))
    val venus_irq = out Bool()
  }
  noIoPrefix()

  val npu = new VenusCoreTopBB
  npu.io.apb_s <> io.apb_s
  io.ahb_m <> npu.io.ahb_m
  io.venus_irq := npu.io.venus_irq
}

object VenusCoreTopBB extends App {
  SpinalConfig(
    targetDirectory = "../../build/rtl",
    oneFilePerComponent = false,
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = LOW),
    anonymSignalPrefix = "tmp" // 简化生成的中间信号命名
  ).generateVerilog(new VenusCoreTopBBWrapper())
}
