package venuscore.cluster

import spinal.core._
import spinal.lib._

/**
  * ClusterScheduler（调度器：tile 级状态机）
  *
  * - 仅管理 `ClState` 状态转移（不做重计算/不做 DMA 地址生成）；
  * - 消费各模块的 done/ready/busy 标志，决定下一状态；
  * - 对上层 Ctrl 暴露 `cl_busy/cl_done/cl_status`。
  */
final class ClusterScheduler(val top: ClusterCtrl) extends Area {
  import top._

  // =========================================================
  // Area 1: Tile FSM（顶层执行流控制）
  // =========================================================
  val fsm_area: Area = new Area {
    val state_next = ClState()
    state_next := state_reg

    // 默认：不产生行推进脉冲。
    // Backend 将其作为 1 拍 strobe，保证每行只推进一次 y_out。
    advance_row_pulse := False

    when(cl_flush) {
      state_next := ClState.IDLE
    } otherwise {
      switch(state_reg) {
        is(ClState.IDLE) {
          when(tile_start_pulse && cl_enable) {
            // 精简策略：
            // - AvgPool 无 WBuf：IDLE -> LOAD_ACT
            // - 其他算子：IDLE -> LOAD_WGT -> LOAD_ACT
            // - 保留 PW IBUF 乒乓预取（见 Backend / ActDMA）。
            when(tile_is_avg_pulse) {
              state_next := ClState.LOAD_ACT
            } elsewhen (wgt_reuse_next && act_reuse_next) {
              state_next := ClState.COMPUTE
            } elsewhen (wgt_reuse_next) {
              state_next := ClState.LOAD_ACT
            } otherwise {
              state_next := ClState.LOAD_WGT
            }
          }
        }
        is(ClState.LOAD_WGT) {
          when(wgt_load_done) {
            when(act_reuse_reg) {
              state_next := ClState.COMPUTE
            } otherwise {
              state_next := ClState.LOAD_ACT
            }
          }
        }
        is(ClState.LOAD_ACT) {
          when(act_load_done) {
            state_next := ClState.COMPUTE
          }
        }
        is(ClState.COMPUTE) {
          when(compute_done) {
            state_next := ClState.SFU_DRAIN
          }
        }
        is(ClState.SFU_DRAIN) {
          when(row_sfu_done && is_last_row) {
            state_next := ClState.DRAIN_OBUF
          } elsewhen (row_sfu_done) {
            // 行结束但非最后一行：
            // - PW：若下一行已在另一个 IBUF bank 预取就绪，则直接 COMPUTE 以减少 bubble；
            // - AVG/rolling(DW/CONV)：若下一行已预取就绪，则直接 COMPUTE；
            //   否则若仍在预取 inflight，则留在 SFU_DRAIN 等待（避免进入 LOAD_ACT 后又空转）；
            // - 其他算子：回到 LOAD_ACT 触发下一行加载。
            val can_skip_load =
              (is_pw_uop && pw_next_row_ready) ||
                (is_pool_uop && avg_next_row_ready) ||
                (use_rolling && roll_next_row_ready)

            val should_wait_prefetch =
              (is_pool_uop && avg_pf_inflight && !avg_next_row_ready) ||
                (use_rolling && roll_pf_inflight && !roll_next_row_ready)

            when(can_skip_load) {
              state_next := ClState.COMPUTE
              advance_row_pulse := True
            } elsewhen (should_wait_prefetch) {
              state_next := ClState.SFU_DRAIN
            } otherwise {
              state_next := ClState.LOAD_ACT
              advance_row_pulse := True
            }
          }
        }
        is(ClState.DRAIN_OBUF) {
          when(drain_done) {
            state_next := ClState.DONE
          }
        }
        is(ClState.DONE) {
          state_next := ClState.IDLE
        }
        is(ClState.ERROR) {
          when(cl_flush) {
            state_next := ClState.IDLE
          }
        }
      }
    }

    state_reg := state_next

    // 对上暴露 cluster 状态
    io.ctrl.cl_busy := (state_reg =/= ClState.IDLE) && (state_reg =/= ClState.ERROR)
    io.ctrl.cl_done := (state_reg === ClState.DONE)
    io.ctrl.cl_status := 0 // 保留：可用于记录详细错误码
  }

  // =========================================================
  // Area 9：兼容性钩子（给回归脚本/XMR 使用）
  // =========================================================
  val compat_area: Area = new Area {
    // 回归 TB 可能通过 XMR 读取这些寄存器名。
    when(cl_flush) {
      avg_row0_ready_reg := False
      avg_next_row0_prefetched_reg := False
      avg_next_row0_y_out_reg := 0
    }
  }
}
