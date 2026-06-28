package venuscore.dma

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.ahblite._
import venuscore.config.VenusCoreConfig

/**
 * 基于 SpinalHDL 官方库 AhbLite3OnChipRam 封装的 AHB-Lite3 存储器
 *
 * @param ahbConfig AHB-Lite3 总线配置 (主要使用 addressWidth, dataWidth)
 * @param byteCount 内存大小，单位为字节 (Byte)
 */
case class DmaRamAhb(ahbConfig: AhbLite3Config, byteCount: BigInt) extends Component {

  val io = new Bundle {
    val ahb = slave(AhbLite3(ahbConfig))
  }
  noIoPrefix()
  // 实例化官方库的 AHB-Lite3 On-Chip RAM
  // 官方库已经处理了 HREADY 反压、地址解码和字节掩码(Byte Lane)等细节
  val ram = AhbLite3OnChipRam(ahbConfig, byteCount)

  ram.noIoPrefix()
  // 直接连接 IO
  io.ahb <> ram.io.ahb
}


/** 顶层 Verilog 生成器 */
object DmaRamAhb extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = true,
    headerWithDate = false,
    anonymSignalPrefix = "tmp",
    keepAll = true
  ).generateVerilog(DmaRamAhb(fpgaCfg.ahbLite3Cfg, 4096))
}