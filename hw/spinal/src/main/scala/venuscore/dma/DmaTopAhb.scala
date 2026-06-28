package venuscore.dma

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.ahblite._
import venuscore.common._
import venuscore.config._

/**
 * DmaTopAhb
 * ---------------------------------------------------------------------
 * 作用：集中式 DMA Top（AHB 后端）。
 *
 * 功能概览：
 * - 聚合多路 DMA 通道（每 cluster：IFM/WGT 读 + OFM 写；Ctrl：uOP 读）；
 * - Round-robin 仲裁命令 → 送入 `DmaAGU` 生成 mem 请求；
 * - `DmaEngineAhb` 负责 AHB 时序、burst 组织与读写数据通路；
 * - 对读返回使用 `StreamDemux` 按通道分发；对写数据使用 `StreamMux` 按通道选路。
 *
 * 设计要点（时序/资源）：
 * - Arbiter 的 `output/chosen` 为组合逻辑：这里将 (cmd,id) 打包后 `m2sPipe()` 打拍，
 *   切断“仲裁→AGU→引擎”的长组合路径。
 */
case class DmaTopAhb(cfg: DmaConfig) extends Component {
  val cluster_num = cfg.clusterNum

  // --------------------------------------------------------------------------
  // IO 定义
  // --------------------------------------------------------------------------
  val io = new Bundle {
    val ahb = master(AhbLite3Master(cfg.ahbLite3Cfg))
    // profiling/debug taps（只读）：每个 beat 完成时产生 1 拍脉冲
    val dbg_rd_fire = out(Bool())
    val dbg_wr_fire = out(Bool())
    // Cluster 侧接口
    val wgt_rd_vec = Vec(slave(DmaFetchPort(cfg)), cluster_num)
    val ifm_rd_vec = Vec(slave(DmaFetchPort(cfg)), cluster_num)
    val ofm_wr_vec = Vec(slave(DmaStorePort(cfg)), cluster_num)
    // Ctrl 侧接口
    val instr_rd = slave(DmaFetchPort(cfg))
  }
  noIoPrefix()

  // --------------------------------------------------------------------------
  // 1. 实例化所有通道 (Dumb Mode - 纯 FIFO)
  // --------------------------------------------------------------------------
  val wgt_fetchs = Array.tabulate(cluster_num)(_ => DmaFetch(cfg))
  val ifm_fetchs = Array.tabulate(cluster_num)(_ => DmaFetch(cfg))
  val instr_fetch = DmaFetch(cfg)
  val ofm_stores = Array.tabulate(cluster_num)(_ => DmaStore(cfg))

  (wgt_fetchs zip io.wgt_rd_vec).foreach { case (m, p) => m.io.port <> p }
  (ifm_fetchs zip io.ifm_rd_vec).foreach { case (m, p) => m.io.port <> p }
  instr_fetch.io.port <> io.instr_rd
  (ofm_stores zip io.ofm_wr_vec).foreach { case (m, p) => m.io.port <> p }

  // --------------------------------------------------------------------------
  // 2. 准备聚合流 (Stream Aggregation)
  // --------------------------------------------------------------------------
  val all_cmd_streams = new scala.collection.mutable.ArrayBuffer[Stream[DmaCmd]]()
  val is_write_table = new scala.collection.mutable.ArrayBuffer[Bool]()

  // Group 1: Wgt (Read)
  wgt_fetchs.foreach { f =>
    all_cmd_streams += f.io.ag_cmd
    is_write_table += False
  }
  // Group 2: Ifm (Read)
  ifm_fetchs.foreach { f =>
    all_cmd_streams += f.io.ag_cmd
    is_write_table += False
  }
  // Group 3: Instr (Read)
  all_cmd_streams += instr_fetch.io.ag_cmd
  is_write_table += False

  // Group 4: Store (Write)
  ofm_stores.foreach { s =>
    all_cmd_streams += s.io.ag_cmd
    is_write_table += True
  }

  // --------------------------------------------------------------------------
  // 3. 核心模块实例化
  // --------------------------------------------------------------------------
  val agu = DmaAGU(cfg)
  val engine = DmaEngineAhb(cfg)

  io.ahb <> engine.io.ahb
  io.dbg_rd_fire := engine.io.dbg_rd_fire
  io.dbg_wr_fire := engine.io.dbg_wr_fire

  // --------------------------------------------------------------------------
  // 4. 命令仲裁 (Arbiter + Pipeline + Manual Lock Logic)
  // --------------------------------------------------------------------------
  val arbiter = StreamArbiterFactory.roundRobin.build(DmaCmd(cfg), all_cmd_streams.length)
  (arbiter.io.inputs, all_cmd_streams).zipped.foreach(_ << _)

  // === 时序切分：仲裁输出打拍 ===
  // 问题：Arbiter 的 output/chosen 为组合逻辑，直接驱动 AGU 会导致长路径。
  // 解决：将 (Cmd, ChosenID) 打包，经一级寄存器（m2sPipe）再送给 AGU，并用打拍后的 ID 做 lock。

  // 定义一个临时 Bundle 携带 ID
  case class ArbOutBundle() extends Bundle {
    val cmd = DmaCmd(cfg)
    val id = UInt(log2Up(all_cmd_streams.length) bits)
  }

  val arb_out_stage_in = Stream(ArbOutBundle())
  val arb_out_stage_out = Stream(ArbOutBundle())

  // 1. 打包数据与ID
  arb_out_stage_in.valid := arbiter.io.output.valid
  arb_out_stage_in.payload.cmd := arbiter.io.output.payload
  arb_out_stage_in.payload.id := arbiter.io.chosen // 同步捕获当前仲裁结果
  arbiter.io.output.ready := arb_out_stage_in.ready

  // 2. 插入 Pipeline (寄存器) 切断关键路径
  arb_out_stage_out << arb_out_stage_in.m2sPipe()

  // 3. 连接到 AGU
  agu.io.cmd_in.valid := arb_out_stage_out.valid
  agu.io.cmd_in.payload := arb_out_stage_out.payload.cmd
  arb_out_stage_out.ready := agu.io.cmd_in.ready

  // 4. 锁定逻辑 (使用打拍后的 ID)
  val chosen_reg = Reg(UInt(log2Up(all_cmd_streams.length) bits))
  val staged_id = arb_out_stage_out.payload.id // 取出经过寄存器的 ID

  when(agu.io.cmd_in.fire) {
    chosen_reg := staged_id
  }

  val current_ch_id = UInt(log2Up(all_cmd_streams.length) bits)

  when(agu.io.busy) {
    current_ch_id := chosen_reg
  } otherwise {
    current_ch_id := staged_id
  }

  agu.io.cmd_ch := current_ch_id
  // === 时序切分结束 ===

  // --------------------------------------------------------------------------
  // 5. 属性查表 (Metadata Lookup)
  // --------------------------------------------------------------------------
  val is_write_vec = Vec(is_write_table)
  agu.io.cmd_is_write := is_write_vec(current_ch_id)

  engine.io.memReq << agu.io.mem_req

  // --------------------------------------------------------------------------
  // 6. 写数据路由 (Write Data Mux)
  // --------------------------------------------------------------------------
  val store_data_streams = Vec(ofm_stores.map(_.io.ag_data))
  val store_id_offset = cluster_num * 2 + 1
  val engine_processing_id = engine.io.rdDataChId
  val current_store_idx = (engine_processing_id - U(store_id_offset)).resized

  engine.io.wrData << StreamMux(current_store_idx, store_data_streams)

  // --------------------------------------------------------------------------
  // 7. 读数据分发 (Read Data Demux)
  // --------------------------------------------------------------------------
  val fetch_rsps = new scala.collection.mutable.ArrayBuffer[Stream[Bits]]()
  wgt_fetchs.foreach(fetch_rsps += _.io.ag_rsp)
  ifm_fetchs.foreach(fetch_rsps += _.io.ag_rsp)
  fetch_rsps += instr_fetch.io.ag_rsp

  val demux_outputs = StreamDemux(engine.io.rdData.map(_.fragment), engine.io.rdDataChId, all_cmd_streams.length)

  for (i <- 0 until fetch_rsps.length) {
    fetch_rsps(i) << demux_outputs(i)
  }

  for (i <- fetch_rsps.length until all_cmd_streams.length) {
    demux_outputs(i).ready := True
  }
}

object DmaTopAhb extends App {
  val fpgaCfg = VenusCoreConfig.default.dmaCfg

  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new DmaTopAhb(fpgaCfg)).printPruned()
}
