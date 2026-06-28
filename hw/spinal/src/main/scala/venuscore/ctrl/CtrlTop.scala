package venuscore.ctrl

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba3.apb._
import venuscore.config._
import venuscore.common._
import venuscore.cluster._

/**
 * CtrlTop
 * -------
 * VenusCore NPU 控制器顶层 (Architecture v0.5)。
 *
 * 架构数据流：
 * RegBlock (配置) -> CtrlUopFetch (DMA抓取 & 解码) -> Stream(uOP) -> CtrlScheduler (动态分发) -> Clusters
 *
 * 职责：
 * 1. 实例化 RegBlock, UopFetch, Scheduler, IntCtrl。
 * 2. 桥接寄存器控制信号到内部逻辑。
 * 3. 路由中断与状态反馈。
 */
case class CtrlTop(cfg: CtrlConfig) extends Component {

  val io = new Bundle {
    // --- SoC 侧接口 ---
    val apb = slave(Apb3(cfg.apb3Config)) // 寄存器配置总线
    val npu_int = out Bool() // 全局中断线

    // --- 存储侧接口 ---
    val instr_dma = master(DmaFetchPort(cfg.dmaConfig)) // 指令读取 DMA

    // --- DMA Profiling taps（只读，不参与控制）---
    // 说明：用于统计 DMA 读/写累计字节数；由 DMA 子系统输出 "每个 beat 完成" 脉冲。
    val dma_dbg_rd_fire = in Bool()
    val dma_dbg_wr_fire = in Bool()

    // --- Cluster 侧接口 (支持多 Cluster) ---
    // uOP 数据流：Scheduler -> Cluster
    val uop_data = Vec(master(Stream(ClusterUop(cfg.clusterConfig))), cfg.clusterNum)
    // 控制与状态：Scheduler <-> Cluster
    val ctrl = Vec(master(ClusterCtrlPort(cfg.clusterConfig)), cfg.clusterNum)
  }

  noIoPrefix()

  // ==========================================================================
  // 1. 子模块实例化
  // ==========================================================================
  val reg_block = CtrlRegBlock(cfg)
  val uop_fetch = CtrlUopFetch(cfg)
  val scheduler = CtrlScheduler(cfg)
  val int_ctrl = InterruptCtrl(cfg)

  // ==========================================================================
  // 2. SoC 侧连接 (APB & IRQ)
  // ==========================================================================

  // 2.1 APB 连接
  reg_block.io.apb <> io.apb

  // 2.2 中断链
  // 路径: Scheduler (原始脉冲) -> RegBlock (状态记录/屏蔽) -> IntCtrl (最终汇总) -> IO
  reg_block.io.npu_int_raw := scheduler.io.npu_int_raw

  int_ctrl.io.int_status := reg_block.io.npu_int_status
  int_ctrl.io.int_en := reg_block.io.npu_int_ctrl
  io.npu_int := int_ctrl.io.npu_int

  // ==========================================================================
  // 3. 控制路径 (RegBlock -> Fetch & Scheduler)
  // ==========================================================================

  // 3.1 启动桥接逻辑
  // RegBlock 输出 start 为 1 拍脉冲，正好对应 Stream valid 语义
  // 将寄存器配置打包成 FetchReq 发送给 UopFetch
  uop_fetch.io.fetch_req.valid := reg_block.io.npu_ctrl.start
  uop_fetch.io.fetch_req.payload.uop_base := reg_block.io.uop_base
  // 注意：RegBlock Count 是 16位，内核逻辑统一使用 32位
  uop_fetch.io.fetch_req.payload.uop_count := reg_block.io.uop_count

  // 3.2 侧带控制信号 (Abort / Soft Reset)
  // 这些信号优先级高，直接广播给所有子模块
  uop_fetch.io.abort := reg_block.io.npu_ctrl.abort
  uop_fetch.io.soft_reset := reg_block.io.npu_ctrl.soft_reset

  scheduler.io.npu_ctrl := reg_block.io.npu_ctrl
  scheduler.io.uop_count := reg_block.io.uop_count // Scheduler 也需要总数做记分板

  // 3.3 状态反馈 (Status Logic)
  // 全局 Busy 由 Scheduler 决定（它涵盖了 Fetch、Dispatch 和 Cluster 执行的全过程）
  // UopFetch 的 busy/error 仅供调试或作为 Scheduler 内部参考（当前架构由 Scheduler 统管顶层 Status）
  reg_block.io.npu_status := scheduler.io.npu_status

  // ==========================================================================
  // 4. 数据路径 (Fetch -> Scheduler -> Cluster)
  // ==========================================================================

  // 4.1 DMA 接口
  uop_fetch.io.dma <> io.instr_dma

  // 4.2 Fetch -> Scheduler
  // UopFetch 解码出的 ClusterUop 流送入 Scheduler
  scheduler.io.uop_in << uop_fetch.io.uop_out

  // 4.3 Scheduler -> Clusters
  // Scheduler 内部包含分发逻辑 (Demux)，直接通过 Vector 接口连接外部
  for (i <- 0 until cfg.clusterNum) {
    // Stream 接口 (uOP)
    io.uop_data(i) << scheduler.io.cl_uops(i)

    // Ctrl 接口 (Enable/Flush/Busy/Done/Status)
    // ClusterCtrlPort 定义了 Master/Slave 方向，可以直接连接
    io.ctrl(i) <> scheduler.io.cl_ctrl(i)
  }

  // ==========================================================================
  // 5. 辅助信号 (Debug & Version)
  // ==========================================================================

  // 版本号: v0.5.0
  reg_block.io.npu_debug.version := B(0x00050000, 32 bits)

  // --------------------------------------------------------------------------
  // NPU_DEBUG0 / NPU_DEBUG1：用于 FPGA 上板调试的 DFT 观测信号
  //
  // 约定：
  // - NPU_DEBUG0：运行周期计数（profiling）
  // - NPU_DEBUG1：状态快照（死锁定位）
  //
  // 说明：这些信号只读、不参与功能逻辑（不回馈控制），可放心用于软件轮询排查“卡在哪”。
  // --------------------------------------------------------------------------

  // DEBUG0：从 “任务进入 busy” 开始计数，busy 期间每拍 +1，任务结束后保持不变
  val run_cycles_reg = Reg(UInt(32 bits)) init (0)
  val busy_now = scheduler.io.npu_status.busy
  val busy_d = RegNext(busy_now) init (False)
  val run_enter = busy_now && !busy_d

  // 统一清零口径：
  // - soft_reset
  // - start 一次任务（busy 0->1）
  // - 软件显式清零（NPU_DEBUG_CTRL.CLR_COUNTERS）
  val debug_clr = reg_block.io.debug_clr
  val clear_counters = reg_block.io.npu_ctrl.soft_reset || run_enter || debug_clr

  when(clear_counters) {
    run_cycles_reg := 0
  } elsewhen (busy_now) {
    run_cycles_reg := (run_cycles_reg + 1).resized
  }

  reg_block.io.npu_debug.debug0 := run_cycles_reg.asBits

  // DEBUG1：状态/握手/中断/错误快照
  val cl_busy_bits = Bits(cfg.clusterNum bits)
  val cl_done_bits = Bits(cfg.clusterNum bits)
  val cl_err_bits = Bits(cfg.clusterNum bits)
  for (i <- 0 until cfg.clusterNum) {
    cl_busy_bits(i) := scheduler.io.cl_ctrl(i).cl_busy
    cl_done_bits(i) := scheduler.io.cl_ctrl(i).cl_done
    cl_err_bits(i) := scheduler.io.cl_ctrl(i).cl_status.orR
  }

  val sched_state_u4 = scheduler.io.dbg_state.resized
  val fetch_state_u4 = uop_fetch.io.dbg_state.resized
  val curr_opcode_u4 = Bits(4 bits)
  curr_opcode_u4 := scheduler.io.npu_status.curr_opcode.asBits.resized

  // bit[11:0]：低位标志，便于脚本直接读 bit 定位
  //  [0]  scheduler_busy
  //  [1]  scheduler_error
  //  [2]  sync_hold（SYNC 屏障阻塞后续 uOP 下发）
  //  [3]  uop_fetch_busy
  //  [4]  uop_fetch_uop_out_valid
  //  [5]  uop_fetch_uop_out_ready
  //  [6]  any_cluster_busy
  //  [7]  any_cluster_done_pulse（本拍任意 cluster cl_done=1）
  //  [8]  any_cluster_error（本拍任意 cluster cl_status!=0）
  //  [9]  irq_out（最终对外中断线）
  //  [10] int_status_done（寄存器保持态）
  //  [11] int_status_error（寄存器保持态）
  val flags_u12 = Bits(12 bits)
  flags_u12 := 0
  flags_u12(0) := scheduler.io.npu_status.busy
  flags_u12(1) := scheduler.io.npu_status.error
  flags_u12(2) := scheduler.io.dbg_sync_hold
  flags_u12(3) := uop_fetch.io.busy
  flags_u12(4) := uop_fetch.io.uop_out.valid
  flags_u12(5) := uop_fetch.io.uop_out.ready
  flags_u12(6) := cl_busy_bits.orR
  flags_u12(7) := cl_done_bits.orR
  flags_u12(8) := cl_err_bits.orR
  flags_u12(9) := io.npu_int
  flags_u12(10) := reg_block.io.npu_int_status.done_int_status
  flags_u12(11) := reg_block.io.npu_int_status.error_int_status

  // NPU_DEBUG1[31:0] 打包格式：
  //   [31:24] error_code（来自 NPU_STATUS）
  //   [23:20] curr_opcode[3:0]
  //   [19:16] scheduler_state（SchedState，编码见 CtrlScheduler）
  //   [15:12] uop_fetch_state（CtrlUopFetch.State，编码见 CtrlUopFetch）
  //   [11:0]  flags_u12
  reg_block.io.npu_debug.debug1 :=
    scheduler.io.npu_status.error_code ##
      curr_opcode_u4 ##
      sched_state_u4 ##
      fetch_state_u4 ##
      flags_u12

  // --------------------------------------------------------------------------
  // NPU_DEBUG2..8：profiling counters（busy=1 期间计数，done 后保持）
  //
  // 计数窗口：NPU_STATUS.BUSY=1 期间
  // 清零：soft_reset / start 一次任务 / NPU_DEBUG_CTRL.CLR_COUNTERS
  // --------------------------------------------------------------------------

  val word_bytes_u32 = U(cfg.dmaConfig.wordBytes, 32 bits)

  // DEBUG2：CLUSTER_BUSY_CYCLES
  val cl_busy_cycles_reg = Reg(UInt(32 bits)) init (0)
  when(clear_counters) {
    cl_busy_cycles_reg := 0
  } elsewhen (busy_now && cl_busy_bits.orR) {
    cl_busy_cycles_reg := (cl_busy_cycles_reg + 1).resized
  }
  reg_block.io.npu_debug.debug2 := cl_busy_cycles_reg.asBits

  // DEBUG3：STALL_UOPFETCH_CYCLES（uop_fetch busy 且暂时没有 uOP 输出）
  val stall_uopfetch_cycles_reg = Reg(UInt(32 bits)) init (0)
  val stall_uopfetch = uop_fetch.io.busy && !uop_fetch.io.uop_out.valid
  when(clear_counters) {
    stall_uopfetch_cycles_reg := 0
  } elsewhen (busy_now && stall_uopfetch) {
    stall_uopfetch_cycles_reg := (stall_uopfetch_cycles_reg + 1).resized
  }
  reg_block.io.npu_debug.debug3 := stall_uopfetch_cycles_reg.asBits

  // DEBUG4：STALL_DMA_OR_BUF_CYCLES
  // 口径：任务处于 RUN 且未被 SYNC 屏障阻塞；fetch 已经准备好 1 条 uOP，但 scheduler/cluster 不接收；
  //       同时 cluster 不 busy，通常代表 “等待 DMA/片上 buffer/内部 ready”。
  val stall_dma_or_buf_cycles_reg = Reg(UInt(32 bits)) init (0)
  val stall_dma_or_buf =
    scheduler.io.dbg_is_run && !scheduler.io.dbg_sync_hold &&
      uop_fetch.io.uop_out.valid && !uop_fetch.io.uop_out.ready &&
      !cl_busy_bits.orR
  when(clear_counters) {
    stall_dma_or_buf_cycles_reg := 0
  } elsewhen (busy_now && stall_dma_or_buf) {
    stall_dma_or_buf_cycles_reg := (stall_dma_or_buf_cycles_reg + 1).resized
  }
  reg_block.io.npu_debug.debug4 := stall_dma_or_buf_cycles_reg.asBits

  // DEBUG5/6：DMA_RD_BYTES / DMA_WR_BYTES
  val dma_rd_bytes_reg = Reg(UInt(32 bits)) init (0)
  val dma_wr_bytes_reg = Reg(UInt(32 bits)) init (0)
  when(clear_counters) {
    dma_rd_bytes_reg := 0
    dma_wr_bytes_reg := 0
  } elsewhen (busy_now) {
    when(io.dma_dbg_rd_fire) {
      dma_rd_bytes_reg := (dma_rd_bytes_reg + word_bytes_u32).resized
    }
    when(io.dma_dbg_wr_fire) {
      dma_wr_bytes_reg := (dma_wr_bytes_reg + word_bytes_u32).resized
    }
  }
  reg_block.io.npu_debug.debug5 := dma_rd_bytes_reg.asBits
  reg_block.io.npu_debug.debug6 := dma_wr_bytes_reg.asBits

  // DEBUG7：UOP_EXEC_CNT（统计 uOP 被 scheduler 接收的次数）
  val uop_exec_cnt_reg = Reg(UInt(32 bits)) init (0)
  when(clear_counters) {
    uop_exec_cnt_reg := 0
  } elsewhen (busy_now && uop_fetch.io.uop_out.fire) {
    uop_exec_cnt_reg := (uop_exec_cnt_reg + 1).resized
  }
  reg_block.io.npu_debug.debug7 := uop_exec_cnt_reg.asBits

  // DEBUG8：TILE_DONE_CNT（统计 cluster done 事件次数；多 cluster 会累加）
  val tile_done_cnt_reg = Reg(UInt(32 bits)) init (0)
  val done_cnt_u32 = CountOne(cl_done_bits).resize(32)
  when(clear_counters) {
    tile_done_cnt_reg := 0
  } elsewhen (busy_now && cl_done_bits.orR) {
    tile_done_cnt_reg := (tile_done_cnt_reg + done_cnt_u32).resized
  }
  reg_block.io.npu_debug.debug8 := tile_done_cnt_reg.asBits
}

object CtrlTop {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl",
      defaultConfigForClockDomains = ClockDomainConfig(
        resetKind = SYNC,
        resetActiveLevel = LOW
      )
    ).generateVerilog(CtrlTop(VenusCoreConfig.default.ctrlCfg))
  }
}
