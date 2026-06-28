package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * ClusterTop（Cluster 顶层封装）
 * ----------
 * 单个 Cluster 的顶层封装：
 * - 对外只暴露 ClusterTopIo（uOP / SFU 配置 / DMA / 控制状态）
 * - 内部例化并连接：
 *   - ClusterCtrl + ClusterIBuf + ClusterWBuf（控制 + 输入/权重缓存）
 *   - ClusterMacDp + ClusterPEGroup + ClusterSFU + ClusterOBuf（计算 + 后处理 + 输出缓存）
 *
 * 数据流：
 * - ActDMA → ClusterIBuf（输入激活写入）
 * - ClusterIBuf → ClusterMacDp → PELane(ClusterPEGroup) → ClusterSFU（计算与后处理）
 * - ClusterSFU → ClusterOBuf → OutDMA（输出写回）
 *
 * 控制流：
 * ClusterCtrl 通过 IBufCtrlPort / WBufCtrlPort / MacDpCtrlPort /
 * SfuCtrlPort / OBufCtrlPort 驱动所有子模块，以及生成 DMA 命令。
 */
case class ClusterTop(cfg: ClusterConfig) extends Component with HasClusterTopIo {

  // 顶层 IO（聚合了 uOP / SFU 配置 / DMA / 控制状态）
  val io = ClusterTopIo(cfg)
  noIoPrefix()

  // ============================================================
  // 1. 子模块例化
  // ============================================================
  val ctrl = new ClusterCtrl(cfg)
  val ibuf = new ClusterIBuf(cfg)
  val wbuf = new ClusterWBuf(cfg)
  val mac_dp = new ClusterMacDp(cfg)
  val pe_group = new ClusterPEGroup(cfg)
  val sfu = new ClusterSFU(cfg)
  val obuf = new ClusterOBuf(cfg)

  // ============================================================
  // 2. 顶层 IO <-> ClusterCtrl
  // ============================================================

  // uOP / 全局控制（SFU 系数由 MacDp 通过 Flow 配置给 SFU）
  ctrl.io.uop_data <> io.uop_data
  ctrl.io.ctrl <> io.ctrl

  // DMA 命令：ClusterCtrl 产生 → 透传到顶层 DMA 端口
  io.act_dma_port.cmd << ctrl.io.act_dma_cmd
  io.wgt_dma_port.cmd << ctrl.io.wgt_dma_cmd
  io.out_dma_port.cmd << ctrl.io.out_dma_cmd

  // ============================================================
  // 3. ClusterCtrl 与 ClusterIBuf / ClusterWBuf 的连接
  // ============================================================

  // 控制端口：ClusterCtrl master → IBUF/WBUF slave
  ibuf.io.ctrl <> ctrl.io.ibuf_ctrl
  wbuf.io.ctrl <> ctrl.io.wbuf_ctrl

  // ActDMA.data → IBUF.act_stream（激活数据写入 IBUF）
  ibuf.io.act_stream << io.act_dma_port.data

  // WgtDMA.data → WBuf.wgt_stream（权重数据写入 WBUF）
  wbuf.io.wgt_stream << io.wgt_dma_port.data

  // MacDp 统一 IBUF 读口：MacDp master → ClusterIBuf.rd_port slave
  mac_dp.io.ibuf_rd <> ibuf.io.rd_port

  // MacDp WBuf 读端：MacDp master → WBuf.rd_port(i) slave
  for (i <- 0 until cfg.laneNum) {
    mac_dp.io.wbuf_rd(i) <> wbuf.io.rd_port(i)
  }

  // ============================================================
  // 4. MacDp → PELane(ClusterPEGroup) → SFU（计算与后处理）
  // ============================================================

  // MacDp 生成的 Lane 控制 + MAC 输入 → ClusterPEGroup
  for (i <- 0 until cfg.laneNum) {
    pe_group.io.lane_ctrl(i) <> mac_dp.io.lane_ctrl(i)
    pe_group.io.lane_mac_in(i) <> mac_dp.io.lane_mac_out(i)
  }

  // PEGroup 聚合后的 SFU 输入向量 → ClusterSFU
  sfu.io.acc_in << pe_group.io.sfu_acc_out

  // MacDp 行控制：ClusterCtrl master → MacDp.ctrl slave
  mac_dp.io.ctrl <> ctrl.io.mac_dp_ctrl

  // SFU 系数 Flow：MacDp.sfu_coe_cfg Flow → SFU.coe_cfg Flow
  mac_dp.io.sfu_coe >> sfu.io.coe
  // ============================================================
  // 5. ClusterCtrl 与 ClusterSFU / ClusterOBuf + OutDMA 的连接
  // ============================================================

  // SFU 控制端口：ClusterCtrl master → SFU.ctrl slave
  sfu.io.ctrl <> ctrl.io.sfu_ctrl


  // OBuf 控制端口：ClusterCtrl master → OBuf.ctrl slave
  obuf.io.ctrl <> ctrl.io.obuf_ctrl

  // SFU 输出 → OBUF 输入
  obuf.io.in_stream << sfu.io.out_data

  // OBUF.out_stream → OutDMA.data（输出写回数据通路）
  // 语义：Cluster 作为“源头”，往 DMA 写回主存。
  io.out_dma_port.data << obuf.io.out_stream
}

/** 顶层 Verilog 生成器 */
object ClusterTop extends App {
  val clCfg = VenusCoreConfig.default.clusterCfg
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = true,
    headerWithDate = false,
    anonymSignalPrefix = "tmp",
    keepAll = true
  ).generateVerilog(new ClusterTop(clCfg))
}
