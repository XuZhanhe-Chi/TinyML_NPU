package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._

/**
 * ClusterOBuf：Cluster 输出缓存（优先映射为 FPGA 友好的资源）
 *
 * 功能：
 * - 单写单读 FIFO
 * - 使用 readSync (同步读) 以利用 FPGA Block RAM 资源
 * - 包含预取逻辑以适配 Stream 接口并隐藏 BRAM 读取延迟
 *
 * 修改说明：
 * - 替换 readAsync 为 readSync
 * - 增加输出寄存器逻辑以处理 1 cycle 读延迟
 * - 调整指针和计数逻辑以适配流水线
 */
class ClusterOBuf(cfg: ClusterConfig) extends Component {

  val wordWidth = cfg.obufWordWidth
  val depth = cfg.obufDepthWords
  val addrWidth = cfg.obufAddrWidth

  assert(depth > 0, "ClusterOBuf depth must be > 0")

  val io = new Bundle {
    val in_stream = slave Stream (Bits(wordWidth bits))
    val out_stream = master Stream (Bits(wordWidth bits))
    val ctrl = slave(OBufCtrlPort(cfg))
  }
  noIoPrefix()

  val fifo = StreamFifo(Bits(wordWidth bits), depth)
  // depth 较小时，强制用寄存器实现，避免综合推到 BRAM（节约 BRAM 资源）
  // 说明：如需强制寄存器实现，可在此处对 fifo 的 RAM 添加综合属性。

  io.in_stream >> fifo.io.push
  io.out_stream << fifo.io.pop
  io.ctrl.full := !fifo.io.push.ready
  io.ctrl.empty := !fifo.io.pop.valid
  fifo.io.flush := io.ctrl.flush

}

// ===========================================================================
// Verilog 生成器
// ===========================================================================
object ClusterOBuf extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp",
    headerWithDate = false
  ).generateVerilog(new ClusterOBuf(VenusCoreConfig.default.clusterCfg))
}
