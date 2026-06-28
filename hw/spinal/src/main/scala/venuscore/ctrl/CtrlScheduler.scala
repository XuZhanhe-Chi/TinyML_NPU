package venuscore.ctrl

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._
import venuscore.cluster._

/**
 * CtrlScheduler
 * -------------
 * 功能：
 * 1) 从上游获取 uOP 流；
 * 2) 动态分发：选择空闲（ready）的 Cluster 下发 uOP；
 * 3) 记分板：统计已分发数量 / 已完成数量，判断任务结束；
 * 4) 产生状态与中断脉冲；
 *
 * SYNC 同步屏障语义（sync=1）：
 * - 当某条 uOP 的 sync=1 且被成功分发后：
 * 在 “该条 uOP 及其之前所有 uOP 全部完成” 之前，不允许继续下发后续 uOP。
 */
case class CtrlScheduler(cfg: CtrlConfig) extends Component {
  val io = new Bundle {
    // --- 上游：寄存器块控制 ---
    val npu_ctrl = in(NpuCtrlBus(cfg)) // start / abort / soft_reset
    val uop_count = in UInt (16 bits) // 本次任务总 uOP 数

    // --- 上游：寄存器块状态反馈 ---
    val npu_status = out(NpuStatusBus()) // busy / error / debug
    val npu_int_raw = out(NpuIntRawBus()) // done/error 原始中断脉冲

    // --- 数据流输入：来自 UopFetch 的 uOP ---
    val uop_in = slave(Stream(ClusterUop(cfg.clusterConfig)))

    // --- 下游：Cluster 接口 ---
    val cl_uops = Vec(master(Stream(ClusterUop(cfg.clusterConfig))), cfg.clusterNum)
    val cl_ctrl = Vec(master(ClusterCtrlPort(cfg.clusterConfig)), cfg.clusterNum)

    // --- 调试/DFT 观测口（只读，不参与控制）---
    // 说明：SpinalHDL 不允许在父模块直接读取子模块内部 directionless 信号，
    // 因此将关键信号通过显式 IO 端口导出，供 CtrlTop 打包到 NPU_DEBUG0/1。
    val dbg_state = out(Bits(4 bits)) // 调度器主状态机编码
    val dbg_sync_hold = out(Bool()) // SYNC 屏障是否阻塞后续 uOP 下发
    val dbg_sync_wait = out(Bool()) // 是否处于 SYNC 等待阶段（屏障已拉起）
    val dbg_is_run = out(Bool()) // 任务是否处于 RUN 阶段（用于 profiling 归因口径）
  }

  noIoPrefix()

  // ==========================================================================
  // 1. 主状态机与记分板计数器
  // ==========================================================================
  object SchedState extends SpinalEnum {
    val IDLE, RUN, DRAIN, DONE, ABORT = newElement()
  }

  val state = RegInit(SchedState.IDLE)

  // 记分板计数器
  val cnt_dispatched = Reg(UInt(16 bits)) init (0) // 已分发到 Cluster 的 uOP 数
  val cnt_finished = Reg(UInt(16 bits)) init (0) // 已完成（累计 cl_done）的 uOP 数
  val reg_total_count = Reg(UInt(16 bits)) init (0) // 锁存本次任务总数

  val reg_busy = RegInit(False)
  val reg_error = RegInit(False)
  val reg_error_code = Reg(Bits(8 bits)) init (0)

  // 当前正在分发的 Opcode（用于观察/调试）
  val reg_curr_opcode = Reg(ClusterUopOpcode()) init (ClusterUopOpcode.NOP)

  // ==========================================================================
  // 2. SYNC 屏障跟踪寄存器
  // ==========================================================================
  // 当 sync_wait_reg=1 时：禁止继续分发后续 uOP，
  // 直到 cnt_finished 达到 sync_target_done_reg。
  val sync_wait_reg = RegInit(False)
  val sync_target_done_reg = Reg(UInt(16 bits)) init (0)

  // ==========================================================================
  // 3. 完成/错误监控（汇聚各 Cluster 的 done/status）
  // ==========================================================================
  val monitorArea = new Area {
    // 收集 done 脉冲
    val done_bits = Bits(cfg.clusterNum bits)
    for (i <- 0 until cfg.clusterNum) {
      done_bits(i) := io.cl_ctrl(i).cl_done
    }
    // 当前周期完成了多少条（可能多个 Cluster 同时 done）
    val current_cycle_done_cnt = CountOne(done_bits)

    // 收集错误：cl_status != 0 视为错误
    val error_bits = Bits(cfg.clusterNum bits)
    for (i <- 0 until cfg.clusterNum) {
      error_bits(i) := io.cl_ctrl(i).cl_status.orR
    }
    val any_cluster_error = error_bits.orR

    // 简单错误码聚合：锁存一个非零 status（此处用覆盖式优先，满足“有码可见”即可）
    val captured_error_code = Bits(8 bits)
    captured_error_code := 0
    for (i <- 0 until cfg.clusterNum) {
      when(io.cl_ctrl(i).cl_status.orR) {
        captured_error_code := io.cl_ctrl(i).cl_status
      }
    }
  }

  // 组合出“本周期如果把 done 计入后”的完成总数（用于比较/屏障释放）
  val done_cnt_u16 = monitorArea.current_cycle_done_cnt.resize(16)
  val finished_total_now = (cnt_finished + done_cnt_u16).resize(16)

  // 屏障保持条件：已进入屏障 且 完成数还没达到目标
  val sync_hold = sync_wait_reg && (finished_total_now < sync_target_done_reg)

  // 一旦达到目标完成数，立即释放屏障（不额外插入空泡）
  when(sync_wait_reg && (finished_total_now >= sync_target_done_reg)) {
    sync_wait_reg := False
  }

  // ==========================================================================
  // 4. 动态分发逻辑
  //    策略：任意 ready 的 Cluster 都可接收；若多个 ready，按最低 index 优先。
  //    同时受 SYNC 屏障（sync_hold）限制。
  // ==========================================================================
  val dispatchArea = new Area {
    // 收集各 Cluster 的 ready（来自下游 Stream ready）
    val cls_ready_bits = Bits(cfg.clusterNum bits)
    for (i <- 0 until cfg.clusterNum) {
      cls_ready_bits(i) := io.cl_uops(i).ready
    }

    // 当前是否存在任意 Cluster 正在执行（用于禁止多 cluster 并行下发）
    val cl_busy_bits = Bits(cfg.clusterNum bits)
    for (i <- 0 until cfg.clusterNum) {
      cl_busy_bits(i) := io.cl_ctrl(i).cl_busy
    }
    val any_cluster_busy = cl_busy_bits.orR

    // 是否存在任意可用 Cluster
    val any_ready = cls_ready_bits.orR

    // 选择目标 Cluster（优先编码：取最低位 1）
    val target_oh = OHMasking.first(cls_ready_bits)

    // 默认：不上游反压、下游不发 valid
    io.uop_in.ready := False
    for (i <- 0 until cfg.clusterNum) {
      io.cl_uops(i).valid := False
      io.cl_uops(i).payload := io.uop_in.payload
    }

    // RUN 状态下：需要满足 “有 Cluster ready” 且 “未被 sync 屏障阻塞”
    when(state === SchedState.RUN) {
      when(any_ready && !sync_hold) {
        io.uop_in.ready := True
        for (i <- 0 until cfg.clusterNum) {
          io.cl_uops(i).valid := io.uop_in.valid && target_oh(i)
        }
      }
    }

    // 更新当前 opcode（仅用于调试观察）
    when(io.uop_in.fire) {
      reg_curr_opcode := io.uop_in.payload.opcode
    }
  }

  // ==========================================================================
  // 5. 主状态机（任务流转）
  // ==========================================================================
  switch(state) {
    // --------------------------------------------------------
    // IDLE：等待 start
    // --------------------------------------------------------
    is(SchedState.IDLE) {
      reg_busy := False

      when(io.npu_ctrl.soft_reset) {
        reg_error := False
        reg_error_code := 0
        sync_wait_reg := False
        sync_target_done_reg := 0
      } elsewhen (io.npu_ctrl.start) {
        // 启动新任务
        cnt_dispatched := 0
        cnt_finished := 0
        reg_total_count := io.uop_count
        reg_busy := True
        reg_error := False
        reg_error_code := 0

        // 清除屏障状态
        sync_wait_reg := False
        sync_target_done_reg := 0

        when(io.uop_count === 0) {
          state := SchedState.DONE
        } otherwise {
          state := SchedState.RUN
        }
      }
    }

    // --------------------------------------------------------
    // RUN：持续分发 uOP
    // --------------------------------------------------------
    is(SchedState.RUN) {
      // 1) 分发计数：每 fire 一次表示成功分发一条 uOP
      when(io.uop_in.fire) {
        cnt_dispatched := cnt_dispatched + 1
      }

      // 2) 完成计数：累计所有 Cluster 的 done 脉冲
      cnt_finished := cnt_finished + done_cnt_u16

      // 3) SYNC 屏障：当 sync=1 的 uOP 成功分发后，设置屏障目标
      //    目标完成数 = 分发该条后的累计 dispatched（即 cnt_dispatched + 1）
      when(io.uop_in.fire && io.uop_in.payload.sync) {
        sync_wait_reg := True
        sync_target_done_reg := (cnt_dispatched + 1).resize(16)
      }

      // 4) 所有 uOP 都已分发：转入 DONE 或 DRAIN
      when(io.uop_in.fire && (cnt_dispatched + 1 === reg_total_count)) {
        when(finished_total_now === reg_total_count) {
          state := SchedState.DONE
        } otherwise {
          state := SchedState.DRAIN
        }
      }
    }

    // --------------------------------------------------------
    // DRAIN：uOP 已发完，等待全部完成
    // --------------------------------------------------------
    is(SchedState.DRAIN) {
      cnt_finished := cnt_finished + done_cnt_u16
      when(finished_total_now === reg_total_count) {
        state := SchedState.DONE
      }
    }

    // --------------------------------------------------------
    // DONE：完成脉冲一拍
    // --------------------------------------------------------
    is(SchedState.DONE) {
      reg_busy := False
      // 清理屏障状态，避免影响下一次任务
      sync_wait_reg := False
      sync_target_done_reg := 0
      state := SchedState.IDLE
    }

    // --------------------------------------------------------
    // ABORT：中止（错误或上层 abort）
    // --------------------------------------------------------
    is(SchedState.ABORT) {
      reg_busy := False
      sync_wait_reg := False
      sync_target_done_reg := 0
      state := SchedState.IDLE
    }
  }

  // ==========================================================================
  // 6. abort / reset 优先级与错误捕捉
  // ==========================================================================
  // abort 优先级最高
  when(io.npu_ctrl.abort) {
    state := SchedState.ABORT
  }

  // soft_reset 强制回到 IDLE，并清状态
  when(io.npu_ctrl.soft_reset) {
    state := SchedState.IDLE
    reg_busy := False
    reg_error := False
    reg_error_code := 0
    sync_wait_reg := False
    sync_target_done_reg := 0
  }

  // 错误捕捉：RUN/DRAIN 期间任意 Cluster 报错则中止
  when((state === SchedState.RUN || state === SchedState.DRAIN) && monitorArea.any_cluster_error) {
    reg_error := True
    // 仅首次出错时锁存错误码
    when(!reg_error) {
      reg_error_code := monitorArea.captured_error_code
    }
    state := SchedState.ABORT
  }

  // ==========================================================================
  // 7. Cluster 控制输出
  // ==========================================================================
  for (i <- 0 until cfg.clusterNum) {
    // 非 IDLE 即使能 Cluster
    io.cl_ctrl(i).cl_enable := (state =/= SchedState.IDLE)
    // ABORT 或外部 abort 时 flush
    io.cl_ctrl(i).cl_flush := (state === SchedState.ABORT) || io.npu_ctrl.abort
    // 固定分配 Cluster ID（按端口下标）
    io.cl_ctrl(i).cl_id := B(i, io.cl_ctrl(i).cl_id.getWidth bits)
  }

  // ==========================================================================
  // 8. NPU 顶层状态与中断输出
  // ==========================================================================
  io.npu_status.busy := reg_busy
  io.npu_status.error := reg_error
  io.npu_status.curr_opcode := reg_curr_opcode
  io.npu_status.error_code := reg_error_code

  // done_int_raw：DONE 状态打一拍脉冲
  io.npu_int_raw.done_int_raw := (state === SchedState.DONE)
  // error_int_raw：任意 cluster_error 的原始指示（可按需再做脉冲化/屏蔽）
  io.npu_int_raw.error_int_raw := monitorArea.any_cluster_error

  // ==========================================================================
  // 9. Debug taps（导出给 CtrlTop）
  // ==========================================================================
  io.dbg_state := state.asBits.resized
  io.dbg_sync_hold := sync_hold
  io.dbg_sync_wait := sync_wait_reg
  io.dbg_is_run := (state === SchedState.RUN)
}

object CtrlScheduler {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl",
      headerWithDate = false
    ).generateVerilog(CtrlScheduler(VenusCoreConfig.default.ctrlCfg))
  }
}
