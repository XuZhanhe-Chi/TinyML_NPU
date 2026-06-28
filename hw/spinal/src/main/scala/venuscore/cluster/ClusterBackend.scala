package venuscore.cluster

import spinal.core._
import spinal.lib._

/**
  * ClusterBackend（后端：数据/循环侧控制）
  *
  * Area 划分（按职责拆分，便于阅读与时序收敛）：
  * - Area 3：行循环 + IBUF 行映射（rowId 旋转/rolling）
  * - Area 4：WgtDMA + WBuf 加载控制（coeff/weight）
  * - Area 5：ActDMA + IBuf 加载控制（PW 乒乓也在这里）
  * - Area 6：MacDp 行级控制 + kernel-group 循环
  * - Area 7：SFU drain 控制（行结束握手）
  * - Area 8：OBuf + OutDMA 写回（含写回收尾保护）
  */
final class ClusterBackend(val top: ClusterCtrl) extends Area {
  import top._

  // 当前 kernel group index：需要被 MacDp 控制与 OutDMA 同时使用。
  // 放在 Backend 顶层以避免跨 Area 访问字段带来的结构类型问题。
  val kernel_group_idx_reg =
    Reg(UInt(cfg.wbufAddrWidth bits)) init (0)

  // AVG2x2：跟踪“当前 y_out”所需的 row0/row1 是否就绪。
  // 说明：row0 可能来自“下一行 row0 预取”（先写入 rowId_top），在 advance_row_pulse 时旋转进 rowId_mid。
  val avg_row1_ready_reg = Reg(Bool()) init (False)
  val avg_next_row1_prefetched_reg = Reg(Bool()) init (False)
  val avg_next_row1_y_out_reg = Reg(UInt(8 bits)) init (0)

  // rolling CONV/DW（stride=1）：跟踪“下一行需要的新 bottom 行”是否已预取到位。
  val roll_bot_ready_reg = Reg(Bool()) init (False)
  val roll_next_bot_prefetched_reg = Reg(Bool()) init (False)
  val roll_next_bot_y_out_reg = Reg(UInt(8 bits)) init (0)

  // Backend 局部状态按 tile 复位（共享寄存器由 Frontend 在 tile_start 时统一复位）。
  when(cl_flush || (tile_start_pulse && (state_reg === ClState.IDLE))) {
    avg_row1_ready_reg := False
    avg_next_row1_prefetched_reg := False
    avg_next_row1_y_out_reg := 0
    roll_bot_ready_reg := False
    roll_next_bot_prefetched_reg := False
    roll_next_bot_y_out_reg := 0
  }

  // =========================================================
  // Area 3: 行循环与 IBUF 行映射（rolling line buffer 控制）
  // =========================================================
  val loop_ctrl_area: Area = new Area {
    // 在 Scheduler 发出 pulse 时，进入下一行（保证一行只前进一次）
    when(advance_row_pulse && cl_enable && !cl_flush) {
      val addrWidth = cfg.dmaConfig.addrWidth
      val in_row_bytes_addr = in_row_bytes.resize(addrWidth bits)
      val fi_addr_tile_base = uop_reg.precalc.fi_addr_tile.resize(addrWidth bits)
      val fi_last_row_addr = uop_reg.precalc.fi_last_row_addr.resize(addrWidth bits)

      def satAdd(base: UInt, inc: UInt): UInt = {
        val sum = (base + inc).resize(addrWidth)
        (sum > fi_last_row_addr) ? fi_last_row_addr | sum
      }

      val next_y_out = (y_out_reg + 1).resized

      y_out_reg := y_out_reg + 1
      out_row_offset_bytes_reg :=
        (out_row_offset_bytes_reg + row_stride_bytes).resized

      when(is_pw_uop) {
        pw_row_offset_bytes_reg :=
          (pw_row_offset_bytes_reg + pw_stride_row_bytes).resized
      }

      when(is_pool_uop) {
        val hit_row0 =
          avg_next_row0_prefetched_reg && (avg_next_row0_y_out_reg === next_y_out)
        val hit_row1 =
          avg_next_row1_prefetched_reg && (avg_next_row1_y_out_reg === next_y_out)

        val off2 = (in_row_bytes_addr |<< 1).resized
        val row0_base_next = (avg_row0_base_addr_reg + off2).resized
        val row1_base_next = satAdd(row0_base_next, in_row_bytes_addr)

        avg_row0_offset_bytes_reg :=
          (avg_row0_offset_bytes_reg + off2).resized
        avg_row0_base_addr_reg := row0_base_next
        avg_row1_base_addr_reg := row1_base_next
        avg_row0_ready_reg := hit_row0
        avg_row1_ready_reg := hit_row1

        when(hit_row0) {
          // AVG2x2：将“预取的 next-row0（rowId_top）”旋转到 rowId_mid，供下一行作为 row0 使用。
          val tmp = rowid_mid_reg
          rowid_mid_reg := rowid_top_reg
          rowid_top_reg := tmp
        }

        // 行前进后，清空“下一行预取”的 bookkeeping（不允许跨行复用旧标记）。
        avg_next_row0_prefetched_reg := False
        avg_next_row0_y_out_reg := 0
        avg_next_row1_prefetched_reg := False
        avg_next_row1_y_out_reg := 0
      }

      when(is_conv_uop || is_dw_uop) {
        when(tile_cfg_reg.stride === U(2)) {
          val step2 = (in_row_bytes_addr |<< 1).resized
          val step1 = in_row_bytes_addr
          val step =
            (tile_cfg_reg.pad_top && (y_out_reg === 0)) ? step1 | step2
          s2_row0_addr_reg := satAdd(s2_row0_addr_reg, step)
        } elsewhen (use_rolling) {
          val off2 = (in_row_bytes_addr |<< 1).resized
          val off3 = (off2 + in_row_bytes_addr).resized
          when(y_out_reg === 0) {
            val init_addr =
              tile_cfg_reg.pad_top ?
                satAdd(fi_addr_tile_base, off2) |
                satAdd(fi_addr_tile_base, off3)
            roll_bot_addr_reg := init_addr
          } otherwise {
            roll_bot_addr_reg := satAdd(roll_bot_addr_reg, in_row_bytes_addr)
          }
        }
      }

      when(!is_pw_uop && !is_pool_uop && !(is_conv_uop || is_dw_uop)) {
        val step =
          (tile_cfg_reg.stride === U(2)) ?
            (in_row_bytes_addr |<< 1) |
            in_row_bytes_addr
        seq_row_addr_reg := satAdd(seq_row_addr_reg, step.resized)
      }

      when(use_rolling) {
        val next_bot =
          (rowid_bot_reg === U(2, cfg.ibufRowIdWidth bits)) ?
            U(0, cfg.ibufRowIdWidth bits) |
            (rowid_bot_reg + 1).resized
        rowid_top_reg := rowid_mid_reg
        rowid_mid_reg := rowid_bot_reg
        rowid_bot_reg := next_bot.resized

        val hit_roll =
          roll_next_bot_prefetched_reg && (roll_next_bot_y_out_reg === next_y_out)
        roll_bot_ready_reg := hit_roll
        roll_next_bot_prefetched_reg := False
        roll_next_bot_y_out_reg := 0
      }

      when(is_pw_uop && pw_next_row_ready) {
        val old_idx = pw_rd_bank_reg.asUInt
        pw_rd_bank_reg := !pw_rd_bank_reg
        pw_bank_valid_reg(old_idx) := False
      }
    }

    val last_row_val =
      (tile_cfg_reg.h_tile.resize(y_out_reg.getWidth) - 1).resized

    is_last_row := (y_out_reg === last_row_val)
  }

  // =========================================================
  // Area 4: WgtDMA + WBuf 加载控制（Coeff + Weight 两阶段）
  // =========================================================
  val wgt_dma_area: Area = new Area {
    val wgt_cmd = wgt_dma_stream

    wgt_cmd.valid := False
    wgt_cmd.payload.base := 0
    wgt_cmd.payload.length := 0
    wgt_cmd.payload.stride := 0
    wgt_cmd.payload.repeat := 1

    io.wbuf_ctrl.load_start := False
    io.wbuf_ctrl.load_len_words := 0
    io.wbuf_ctrl.load_kernel_cnt := 0
    io.wbuf_ctrl.load_base_addr := 0

    val coe_len_words =
      U(2, io.wbuf_ctrl.load_len_words.getWidth bits)
    val coe_dma_len =
      (total_kernel_cnt |<< 1)
        .resize(wgt_cmd.payload.length.getWidth)

    val wgt_len_words = UInt(io.wbuf_ctrl.load_len_words.getWidth bits)
    when(is_dw_uop) {
      wgt_len_words := U(3, wgt_len_words.getWidth bits)
    } otherwise {
      wgt_len_words := tile_cfg_reg.c4_in.resize(wgt_len_words.getWidth)
      when(tile_cfg_reg.kernel_taps === U(4)) {
        wgt_len_words := (tile_cfg_reg.c4_in.resize(wgt_len_words.getWidth) |<< 2).resized
      } elsewhen (tile_cfg_reg.kernel_taps === U(9)) {
        val c4_in_u = tile_cfg_reg.c4_in.resize(wgt_len_words.getWidth)
        wgt_len_words := ((c4_in_u |<< 3) + c4_in_u).resized
      }
    }

    val wgt_base_addr =
      (tile_cfg_reg.c4_out << 1)
        .resize(io.wbuf_ctrl.load_base_addr.getWidth)

    val load_phase_wgt_reg = Reg(Bool()) init (False)
    val started_reg = Reg(Bool()) init (False)

    wgt_load_done := False

    when(state_reg === ClState.LOAD_WGT && cl_enable) {
      when(wgt_reuse_reg) {
        wgt_load_done := True
        started_reg := False
        load_phase_wgt_reg := False
      } elsewhen (is_pool_uop) {
        wgt_load_done := True
        started_reg := False
        load_phase_wgt_reg := False
      } otherwise {
        when(!started_reg) {
          io.wbuf_ctrl.load_start := True
          io.wbuf_ctrl.load_kernel_cnt := total_kernel_cnt
          wgt_cmd.valid := True

          when(!load_phase_wgt_reg) {
            io.wbuf_ctrl.load_base_addr := 0
            io.wbuf_ctrl.load_len_words := coe_len_words
            wgt_cmd.payload.base := uop_reg.coe_addr
            wgt_cmd.payload.length := coe_dma_len
          } otherwise {
            io.wbuf_ctrl.load_base_addr := wgt_base_addr
            io.wbuf_ctrl.load_len_words := wgt_len_words
            wgt_cmd.payload.base := uop_reg.w_addr
            wgt_cmd.payload.length :=
              uop_reg.precalc.wgt_dma_len_words.resize(wgt_cmd.payload.length.getWidth)
          }

          when(wgt_cmd.ready) {
            started_reg := True
          }
        }

        when(io.wbuf_ctrl.load_done) {
          started_reg := False
          when(!load_phase_wgt_reg) {
            load_phase_wgt_reg := True
          } otherwise {
            wgt_load_done := True
            load_phase_wgt_reg := False
          }
        }
      }
    } otherwise {
      started_reg := False
      load_phase_wgt_reg := False
    }

    when((state_reg === ClState.LOAD_WGT) && wgt_load_done && !is_pool_uop && cl_enable) {
      wgt_key_reg := wgt_key_pending_reg
      wgt_valid_reg := True
    }
  }

  // =========================================================
  // Area 5: ActDMA（两级地址预计算 + IBUF 行加载）
  // =========================================================
  val act_dma_area: Area = new Area {
    val act_cmd = act_dma_stream

    pw_next_row_ready := False

    act_cmd.valid := False
    act_cmd.payload.base := 0
    act_cmd.payload.length := 0
    act_cmd.payload.repeat := 0
    act_cmd.payload.stride := 0

    io.ibuf_ctrl.load_start := False
    io.ibuf_ctrl.load_row_id := 0
    io.ibuf_ctrl.load_len_words := 0

    io.ibuf_ctrl.cfg_w_tile :=
      in_row_pixels.resize(io.ibuf_ctrl.cfg_w_tile.getWidth)

    act_load_done := False

    val addrWidth = cfg.dmaConfig.addrWidth
    val dma_line_len =
      in_row_pixels.resize(act_cmd.payload.length.getWidth)
    val line_words =
      ibuf_line_words.resize(io.ibuf_ctrl.load_len_words.getWidth)

    val in_load_act = (state_reg === ClState.LOAD_ACT)
    val in_prefetch_window =
      (state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN)

    final class SharedLineLoader extends Area {
      val active_reg = Reg(Bool()) init (False)
      val dma_sent_reg = Reg(Bool()) init (False)
      val ibuf_started_reg = Reg(Bool()) init (False)
      val ibuf_done_reg = Reg(Bool()) init (False)

      val row_id_reg =
        Reg(UInt(io.ibuf_ctrl.load_row_id.getWidth bits)) init (0)
      val base_addr_reg =
        Reg(UInt(addrWidth bits)) init (0)

      def start(rowId: UInt, baseAddr: UInt): Unit = {
        active_reg := True
        dma_sent_reg := False
        ibuf_started_reg := False
        ibuf_done_reg := False
        row_id_reg := rowId.resized
        base_addr_reg := baseAddr.resized
      }

      def abort(): Unit = {
        active_reg := False
        dma_sent_reg := False
        ibuf_started_reg := False
        ibuf_done_reg := False
      }

      val ibuf_done_pulse =
        io.ibuf_ctrl.load_done &&
          active_reg && ibuf_started_reg
      when(ibuf_done_pulse) {
        ibuf_done_reg := True
      }

      val want_ibuf_start =
        active_reg && !ibuf_started_reg &&
          !io.ibuf_ctrl.load_busy && !io.ibuf_ctrl.load_done

      val done = active_reg && dma_sent_reg && ibuf_done_reg
    }

    val loader = new SharedLineLoader

    object LineTask extends SpinalEnum {
      val NONE, LOAD_ACT, AVG_PF_ROW0, AVG_PF_ROW1, ROLL_PF_BOT = newElement()
    }
    val task_reg = Reg(LineTask()) init (LineTask.NONE)

    val pw_req_inflight_reg = Reg(Bool()) init (False)
    val pw_req_row_reg = Reg(UInt(8 bits)) init (0)
    val pw_req_bank_reg = Reg(Bool()) init (False) // False->bank0(row_id=0), True->bank1(row_id=1)

    val row_idx_reg = Reg(UInt(2 bits)) init (0)

    // 默认：不声明“下一行已就绪”；只有在本 Area 成功预取/命中时才置位。
    avg_next_row_ready := False
    avg_pf_inflight := False
    roll_next_row_ready := False
    roll_pf_inflight := False

    when(is_pw_uop) {
      task_reg := LineTask.NONE
      val y_out_val = y_out_reg

      def pwBankHasRow(bank: Bool, row: UInt): Bool = {
        val idx = bank.asUInt
        pw_bank_valid_reg(idx) && (pw_bank_row_reg(idx) === row)
      }

      val next_row = (y_out_val + 1).resized
      val next_bank = !pw_rd_bank_reg
      val req_complete = pw_req_inflight_reg && loader.done

      val next_ready_mem = pwBankHasRow(next_bank, next_row)
      val next_ready_now =
        req_complete && (pw_req_row_reg === next_row) && (pw_req_bank_reg === next_bank)
      pw_next_row_ready :=
        !is_last_row && (next_ready_mem || next_ready_now)

      val cur_row = y_out_val.resized
      val load_act_bank =
        (y_out_val === 0) ? False | (!pw_rd_bank_reg)
      val cur_ready_mem = pwBankHasRow(load_act_bank, cur_row)
      val cur_ready_now =
        req_complete && (pw_req_row_reg === cur_row) && (pw_req_bank_reg === load_act_bank)

      when(in_load_act && (cur_ready_mem || cur_ready_now)) {
        act_load_done := True
        pw_rd_bank_reg := load_act_bank
        pw_bank_valid_reg((!load_act_bank).asUInt) := False
      }

      val need_load_cur =
        in_load_act && !(cur_ready_mem || cur_ready_now)
      val need_prefetch_next =
        in_prefetch_window && !is_last_row && !pw_next_row_ready

      val pw_ctrl_active =
        (state_reg === ClState.LOAD_ACT) || (state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN)

      when(cl_flush || !cl_enable || !pw_ctrl_active) {
        pw_req_inflight_reg := False
        loader.abort()
        task_reg := LineTask.NONE
      } otherwise {
        when(!pw_req_inflight_reg) {
          when(need_load_cur) {
            pw_req_inflight_reg := True
            pw_req_row_reg := cur_row
            pw_req_bank_reg := load_act_bank
            pw_bank_valid_reg(load_act_bank.asUInt) := False
          } elsewhen (need_prefetch_next) {
            pw_req_inflight_reg := True
            pw_req_row_reg := next_row
            pw_req_bank_reg := next_bank
            pw_bank_valid_reg(next_bank.asUInt) := False
          }
        }

        val pw_can_start_req =
          pw_req_inflight_reg && !loader.active_reg
        when(pw_can_start_req) {
          val row_offset_bytes =
            (pw_req_row_reg === next_row) ?
              (pw_row_offset_bytes_reg + pw_stride_row_bytes).resized |
              pw_row_offset_bytes_reg
          val base_addr =
            (uop_reg.precalc.fi_addr_tile.resize(addrWidth) + row_offset_bytes).resized
          val row_id =
            pw_req_bank_reg.asUInt.resize(io.ibuf_ctrl.load_row_id.getWidth bits)
          loader.start(row_id, base_addr)
          task_reg := LineTask.LOAD_ACT
        }

        when(req_complete) {
          pw_req_inflight_reg := False
          pw_bank_valid_reg(pw_req_bank_reg.asUInt) := True
          pw_bank_row_reg(pw_req_bank_reg.asUInt) := pw_req_row_reg
          loader.abort()
          task_reg := LineTask.NONE
        }
      }
    } otherwise {
      val y_out_val = y_out_reg
      val has_top_pad = tile_cfg_reg.pad_top

      val next_y_out = (y_out_val + 1).resized
      val avg_row0_pf_done =
        avg_next_row0_prefetched_reg && (avg_next_row0_y_out_reg === next_y_out)
      val avg_row1_pf_done =
        avg_next_row1_prefetched_reg && (avg_next_row1_y_out_reg === next_y_out)
      val roll_pf_done =
        roll_next_bot_prefetched_reg && (roll_next_bot_y_out_reg === next_y_out)

      avg_next_row_ready :=
        is_pool_uop && !is_last_row && avg_row0_pf_done && avg_row1_pf_done
      roll_next_row_ready :=
        use_rolling && !is_last_row && roll_pf_done

      val prefetch_can_start =
        in_prefetch_window &&
          (task_reg === LineTask.NONE) &&
          !loader.active_reg &&
          !io.ibuf_ctrl.load_busy &&
          !is_last_row

      val will_start_avg_pf =
        prefetch_can_start &&
          is_pool_uop &&
          ((!avg_row0_pf_done && ((state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN))) ||
            (!avg_row1_pf_done && (state_reg === ClState.SFU_DRAIN)))

      val roll_pf_window =
        (state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN)
      val roll_pf_enable = Bool(cfg.enableRollingPrefetch)
      val will_start_roll_pf =
        prefetch_can_start &&
          roll_pf_enable &&
          use_rolling &&
          !roll_pf_done &&
          roll_pf_window

      avg_pf_inflight :=
        (loader.active_reg &&
          ((task_reg === LineTask.AVG_PF_ROW0) || (task_reg === LineTask.AVG_PF_ROW1))) ||
          will_start_avg_pf
      roll_pf_inflight :=
        ((roll_pf_enable && loader.active_reg && (task_reg === LineTask.ROLL_PF_BOT))) ||
          will_start_roll_pf

      when(!cl_enable || cl_flush) {
        row_idx_reg := 0
        loader.abort()
        task_reg := LineTask.NONE
      } otherwise {
        val fi_addr_tile_base = uop_reg.precalc.fi_addr_tile.resize(addrWidth)
        val fi_last_row_addr = uop_reg.precalc.fi_last_row_addr.resize(addrWidth)
        val in_row_bytes_addr = in_row_bytes.resize(addrWidth)

        def satAddAddr(base: UInt, inc: UInt): UInt = {
          val sum = (base + inc).resize(addrWidth)
          (sum > fi_last_row_addr) ? fi_last_row_addr | sum
        }

        val loader_idle = !loader.active_reg
        val task_idle = task_reg === LineTask.NONE

        // -------------------------------------------------------
        // loader 完成（统一收口：避免不同分支重复写 done/busy）
        // -------------------------------------------------------
        when(loader.done) {
          switch(task_reg) {
            is(LineTask.AVG_PF_ROW0) {
              avg_next_row0_prefetched_reg := True
              avg_next_row0_y_out_reg := next_y_out
            }
            is(LineTask.AVG_PF_ROW1) {
              avg_next_row1_prefetched_reg := True
              avg_next_row1_y_out_reg := next_y_out
            }
            is(LineTask.ROLL_PF_BOT) {
              roll_next_bot_prefetched_reg := True
              roll_next_bot_y_out_reg := next_y_out
            }
            is(LineTask.LOAD_ACT) {
              when(is_pool_uop) {
                val loaded_row0 = loader.row_id_reg === rowid_mid_reg.resized
                val loaded_row1 = loader.row_id_reg === rowid_bot_reg.resized
                avg_row0_ready_reg := avg_row0_ready_reg || loaded_row0
                avg_row1_ready_reg := avg_row1_ready_reg || loaded_row1

                val row0_ready_next = avg_row0_ready_reg || loaded_row0
                val row1_ready_next = avg_row1_ready_reg || loaded_row1
                when(row0_ready_next && row1_ready_next) {
                  act_load_done := True
                }
              } otherwise {
                val use_rolling_conv = use_rolling
                val rows_to_load = UInt(2 bits)
                when(use_rolling_conv) {
                  when(y_out_val === 0) {
                    val first_rows =
                      has_top_pad ? kernel_h_minus1 | kernel_h
                    rows_to_load := first_rows.resize(rows_to_load.getWidth)
                  } otherwise {
                    rows_to_load := 1
                  }
                } elsewhen ((is_conv_uop || is_dw_uop) && (tile_cfg_reg.stride === U(2))) {
                  rows_to_load := kernel_h.resize(rows_to_load.getWidth)
                } otherwise {
                  rows_to_load := 1
                }

                when(row_idx_reg === (rows_to_load - 1)) {
                  act_load_done := True
                  row_idx_reg := 0
                } otherwise {
                  row_idx_reg := row_idx_reg + 1
                }
              }
            }
            default {
              // 无（保留）
            }
          }

          loader.abort()
          task_reg := LineTask.NONE
        }

        // -------------------------------------------------------
        // 触发 LOAD_ACT（最高优先级）
        // -------------------------------------------------------
        when((state_reg === ClState.LOAD_ACT) && task_idle && loader_idle) {
          when(act_reuse_reg) {
            act_load_done := True
            row_idx_reg := 0
          } elsewhen (is_pool_uop) {
            val need_row0 = !avg_row0_ready_reg
            val need_row1 = !avg_row1_ready_reg

            when(!need_row0 && !need_row1) {
              act_load_done := True
            } otherwise {
              val base_row0_addr = avg_row0_base_addr_reg
              val base_row1_addr = avg_row1_base_addr_reg

              val load_row_id = UInt(io.ibuf_ctrl.load_row_id.getWidth bits)
              val base_addr_next = UInt(addrWidth bits)
              when(need_row0) {
                load_row_id := rowid_mid_reg.resized
                base_addr_next := base_row0_addr
              } otherwise {
                load_row_id := rowid_bot_reg.resized
                base_addr_next := base_row1_addr
              }

              loader.start(load_row_id, base_addr_next)
              task_reg := LineTask.LOAD_ACT
            }
          } elsewhen (use_rolling && (y_out_val =/= 0) && roll_bot_ready_reg) {
            act_load_done := True
            roll_bot_ready_reg := False
            row_idx_reg := 0
          } otherwise {
            val use_rolling_conv = use_rolling
            val rows_to_load = UInt(2 bits)
            when(use_rolling_conv) {
              when(y_out_val === 0) {
                val first_rows =
                  has_top_pad ? kernel_h_minus1 | kernel_h
                rows_to_load := first_rows.resize(rows_to_load.getWidth)
              } otherwise {
                rows_to_load := 1
              }
            } elsewhen ((is_conv_uop || is_dw_uop) && (tile_cfg_reg.stride === U(2))) {
              rows_to_load := kernel_h.resize(rows_to_load.getWidth)
            } otherwise {
              rows_to_load := 1
            }

            val load_row_id = UInt(io.ibuf_ctrl.load_row_id.getWidth bits)
            load_row_id := rowid_mid_reg.resized

            val base_addr_next = UInt(addrWidth bits)
            base_addr_next := seq_row_addr_reg

            when(use_rolling_conv) {
              when(y_out_val === 0) {
                val addr0 = fi_addr_tile_base
                val addr1 = satAddAddr(fi_addr_tile_base, in_row_bytes_addr)
                val addr2 = satAddAddr(fi_addr_tile_base, (in_row_bytes_addr |<< 1).resized)
                when(has_top_pad) {
                  when(row_idx_reg === 0) {
                    load_row_id := rowid_mid_reg.resized
                    base_addr_next := addr0
                  } otherwise {
                    load_row_id := rowid_bot_reg.resized
                    base_addr_next := addr1
                  }
                } otherwise {
                  switch(row_idx_reg) {
                    is(U(0)) {
                      load_row_id := rowid_top_reg.resized
                      base_addr_next := addr0
                    }
                    is(U(1)) {
                      load_row_id := rowid_mid_reg.resized
                      base_addr_next := addr1
                    }
                    default {
                      load_row_id := rowid_bot_reg.resized
                      base_addr_next := addr2
                    }
                  }
                }
              } otherwise {
                load_row_id := rowid_bot_reg.resized
                base_addr_next := roll_bot_addr_reg
              }
            } elsewhen ((is_conv_uop || is_dw_uop) && (tile_cfg_reg.stride === U(2))) {
              val row0 = s2_row0_addr_reg
              val row1 = satAddAddr(row0, in_row_bytes_addr)
              val row2 = satAddAddr(row0, (in_row_bytes_addr |<< 1).resized)

              switch(row_idx_reg) {
                is(U(0)) {
                  load_row_id := rowid_top_reg.resized
                  base_addr_next := row0
                }
                is(U(1)) {
                  load_row_id := rowid_mid_reg.resized
                  base_addr_next := row1
                }
                default {
                  load_row_id := rowid_bot_reg.resized
                  base_addr_next := row2
                }
              }

              when(has_top_pad && (y_out_val === 0)) {
                when(row_idx_reg === 0) {
                  base_addr_next := row0
                } elsewhen (row_idx_reg === 1) {
                  base_addr_next := row0
                } otherwise {
                  base_addr_next := row1
                }
              }
            }

            loader.start(load_row_id, base_addr_next)
            task_reg := LineTask.LOAD_ACT
          }
        }

        // -------------------------------------------------------
        // 触发预取（best-effort，低优先级；不应阻塞主流程）
        // -------------------------------------------------------
        when(prefetch_can_start) {
          when(is_pool_uop) {
            val off2 = (in_row_bytes_addr |<< 1).resized
            val row0_next_addr =
              satAddAddr(avg_row0_base_addr_reg.resized, off2)
            val row1_next_addr =
              satAddAddr(row0_next_addr, in_row_bytes_addr)

            // row0 预取既可在 COMPUTE 发起，也可在 SFU_DRAIN 作为兜底发起：
            // 对于“计算窗口很短”的 avgpool（例如 w_tile 很小），COMPUTE 可能来不及踢出预取，
            // 因此允许在 SFU_DRAIN 继续尝试，以减少下一行的 bubble。
            when(!avg_row0_pf_done && ((state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN))) {
              loader.start(rowid_top_reg.resized, row0_next_addr)
              task_reg := LineTask.AVG_PF_ROW0
            } elsewhen (!avg_row1_pf_done && (state_reg === ClState.SFU_DRAIN)) {
              loader.start(rowid_bot_reg.resized, row1_next_addr)
              task_reg := LineTask.AVG_PF_ROW1
            }
          } elsewhen (roll_pf_enable && use_rolling && !roll_pf_done && (state_reg === ClState.SFU_DRAIN)) {
            val off2 = (in_row_bytes_addr |<< 1).resized
            val off3 = (off2 + in_row_bytes_addr).resized
            val next_bot_addr = UInt(addrWidth bits)
            when(y_out_val === 0) {
              next_bot_addr :=
                has_top_pad ?
                  satAddAddr(fi_addr_tile_base, off2) |
                  satAddAddr(fi_addr_tile_base, off3)
            } otherwise {
              next_bot_addr := satAddAddr(roll_bot_addr_reg, in_row_bytes_addr)
            }

            val next_bot_row_id =
              (rowid_bot_reg === U(2, cfg.ibufRowIdWidth bits)) ?
                U(0, cfg.ibufRowIdWidth bits) |
                (rowid_bot_reg + 1).resized

            loader.start(next_bot_row_id.resized, next_bot_addr)
            task_reg := LineTask.ROLL_PF_BOT
          }
        }
      }
    }

    val loader_ctrl_active =
      cl_enable && ((state_reg === ClState.LOAD_ACT) || (state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN))

    when(!loader_ctrl_active || cl_flush) {
      loader.abort()
    } otherwise {
      val dma_can_accept_now = act_cmd.ready
      val dma_issue_now = loader.active_reg && !loader.dma_sent_reg && dma_can_accept_now
      val issue_ibuf = loader.want_ibuf_start && (loader.dma_sent_reg || dma_issue_now)
      when(issue_ibuf) {
        io.ibuf_ctrl.load_start := True
        io.ibuf_ctrl.load_row_id := loader.row_id_reg
        io.ibuf_ctrl.load_len_words := line_words
        loader.ibuf_started_reg := True
      }

      val allow_dma =
        loader.active_reg && !loader.dma_sent_reg &&
          (loader.ibuf_started_reg || issue_ibuf) &&
          (state_reg =/= ClState.IDLE)
      when(allow_dma) {
        act_cmd.valid := True
        act_cmd.payload.base :=
          loader.base_addr_reg.resize(act_cmd.payload.base.getWidth)
        act_cmd.payload.length := dma_line_len
        act_cmd.payload.repeat :=
          tile_cfg_reg.c4_in.resize(act_cmd.payload.repeat.getWidth)
        act_cmd.payload.stride :=
          fi_stride_byte.resize(act_cmd.payload.stride.getWidth)
        when(act_cmd.ready) {
          loader.dma_sent_reg := True
        }
      }
    }

    when((state_reg === ClState.LOAD_ACT) && act_load_done && cl_enable) {
      act_key_reg := act_key_pending_reg
      act_valid_reg := True
    }
  }

  // =========================================================
  // Area 6: MacDp 控制（行级 compute + kernel group 循环）
  // =========================================================
  val pe_group_area: Area = new Area {
    val in_compute = state_reg === ClState.COMPUTE
    val in_compute_d = RegNext(in_compute) init (False)
    val compute_enter = in_compute && !in_compute_d

    // 保持 MacDp 在 SFU_DRAIN 期间仍处于 enable：
    // 对于极短行（例如 w_tile=1），`row_done` 可能在 COMPUTE 结束后才到达；
    // 如果过早关掉 enable，会导致内部流水无法自然 drain，进而影响行完成握手。
    io.mac_dp_ctrl.enable :=
      cl_enable && ((state_reg === ClState.COMPUTE) || (state_reg === ClState.SFU_DRAIN))

    io.mac_dp_ctrl.row_len :=
      tile_cfg_reg.w_tile.resize(io.mac_dp_ctrl.row_len.getWidth)

    val op_bits = Bits(3 bits)
    switch(uop_reg.opcode) {
      is(ClusterUopOpcode.PWCONV) {
        op_bits := B"001"
      }
      is(ClusterUopOpcode.CONV2D) {
        op_bits := B"000"
      }
      is(ClusterUopOpcode.DWCONV) {
        op_bits := B"010"
      }
      is(ClusterUopOpcode.AVGPOOL) {
        op_bits := B"011"
      }
      is(ClusterUopOpcode.MAXPOOL) {
        op_bits := B"101"
      }
      default {
        op_bits := B"000"
      }
    }
    io.mac_dp_ctrl.op_mode := op_bits
    io.mac_dp_ctrl.stride := tile_cfg_reg.stride.resize(io.mac_dp_ctrl.stride.getWidth)

    val macdp_c4_in =
      UInt(io.mac_dp_ctrl.c4_in.getWidth bits)
    when(is_dw_uop || is_pool_uop) {
      macdp_c4_in := 1
    } otherwise {
      macdp_c4_in :=
        tile_cfg_reg.c4_in.resize(macdp_c4_in.getWidth)
    }
    io.mac_dp_ctrl.c4_in := macdp_c4_in

    when(uop_reg.opcode === ClusterUopOpcode.PWCONV) {
      val pw_row_id =
        pw_rd_bank_reg.asUInt.resize(io.mac_dp_ctrl.row_mid.getWidth bits)
      io.mac_dp_ctrl.row_top := pw_row_id
      io.mac_dp_ctrl.row_mid := pw_row_id
      io.mac_dp_ctrl.row_bot := pw_row_id
    } otherwise {
      io.mac_dp_ctrl.row_top := rowid_top_reg
      io.mac_dp_ctrl.row_mid := rowid_mid_reg
      io.mac_dp_ctrl.row_bot := rowid_bot_reg
    }

    io.mac_dp_ctrl.ibuf_valid_w := ibuf_valid_width
    io.mac_dp_ctrl.pad_left := tile_cfg_reg.pad_left
    io.mac_dp_ctrl.pad_right := tile_cfg_reg.pad_right

    io.mac_dp_ctrl.pad_top_en :=
      (y_out_reg === 0) && tile_cfg_reg.pad_top

    val pad_top_u = tile_cfg_reg.pad_top ? U(1, 16 bits) | U(0, 16 bits)
    val y_out_global_u =
      (tile_cfg_reg.y_index.resize(16 bits) + y_out_reg.resize(16 bits)).resized
    val y_out_scaled_u = UInt(16 bits)
    when(tile_cfg_reg.stride === U(2)) {
      y_out_scaled_u := (y_out_global_u |<< 1).resized
    } otherwise {
      y_out_scaled_u := y_out_global_u
    }
    val y_max_u = UInt(16 bits)
    val has_y_halo_shift =
      use_rolling && (tile_cfg_reg.y_index =/= 0)
    when(has_y_halo_shift) {
      y_max_u := (y_out_global_u + kernel_h_minus2.resize(16 bits)).resized
    } otherwise {
      y_max_u := (y_out_scaled_u + kernel_h_minus1.resize(16 bits) - pad_top_u).resized
    }
    io.mac_dp_ctrl.pad_bot_en :=
      tile_cfg_reg.pad_bot && (y_max_u >= uop_reg.geom.ifm_h.resize(16 bits))

    val kernel_group_active_reg = Reg(Bool()) init (False)
    val kernel_group_all_done_reg = Reg(Bool()) init (False)

    val last_group_idx =
      (tile_cfg_reg.c4_out - 1)
        .resize(kernel_group_idx_reg.getWidth)

    when(compute_enter) {
      kernel_group_active_reg := False
      kernel_group_all_done_reg := False
      kernel_group_idx_reg := 0
    } elsewhen (!in_compute || cl_flush) {
      kernel_group_idx_reg := 0
      kernel_group_active_reg := False
      kernel_group_all_done_reg := False
    }

    val start_pulse = Bool()
    start_pulse := False

    when(in_compute && cl_enable) {
      // 不能在 SFU 仍有上一个 kernel-group 尾部结果在途时启动下一组：
      // MacDp 的 CFG_SFU 会覆盖共享量化参数寄存器，稳定污染“上一组每行最后一个输出 word”。
      // 这里要求 SFU 先完全排空，再进入下一组的 start/CFG_SFU。
      when(
        !kernel_group_active_reg &&
          !kernel_group_all_done_reg &&
          (tile_cfg_reg.c4_out =/= 0) &&
          !io.sfu_ctrl.busy
      ) {
        start_pulse := True
        kernel_group_active_reg := True
      }

      when(io.mac_dp_ctrl.row_done && kernel_group_active_reg) {
        kernel_group_active_reg := False
        when(kernel_group_idx_reg === last_group_idx) {
          kernel_group_all_done_reg := True
        } otherwise {
          kernel_group_idx_reg := (kernel_group_idx_reg + 1).resized
        }
      }
    }

    io.mac_dp_ctrl.start := start_pulse

    val is_switching_next =
      io.mac_dp_ctrl.row_done && kernel_group_active_reg &&
        (kernel_group_idx_reg =/= last_group_idx)

    io.mac_dp_ctrl.kernel_group_idx :=
      is_switching_next ?
        (kernel_group_idx_reg + 1) |
        kernel_group_idx_reg

    // WBUF 中 weight 数据起始地址（coeff 占用 2 * C4_OUT words）
    io.mac_dp_ctrl.wbuf_wgt_base :=
      (tile_cfg_reg.c4_out << 1)
        .resize(io.mac_dp_ctrl.wbuf_wgt_base.getWidth bits)

    compute_done :=
      in_compute && kernel_group_all_done_reg

    // kernel_group_idx_reg 暴露给 OutDMA 使用（当前 group index）
  }

  // =========================================================
  // Area 7: SFU 控制（行级结果 drain 到 OBuf）
  // =========================================================
  val sfu_area: Area = new Area {
    io.sfu_ctrl.enable :=
      (cl_enable &&
        ((state_reg === ClState.COMPUTE) ||
          (state_reg === ClState.SFU_DRAIN))) ||
        io.sfu_ctrl.busy

    io.sfu_ctrl.act_type := uop_reg.act_type
    io.sfu_ctrl.trunc_shift := is_avg_uop
    io.sfu_ctrl.pool_ties_even := is_avg_uop && (uop_reg.qmode === ClusterQMode.Q4)

    val row_target = row_out_words
    val sfu_cnt = Reg(UInt(16 bits)) init (0)

    val in_compute = state_reg === ClState.COMPUTE
    val in_compute_d = RegNext(in_compute) init (False)
    val compute_enter = in_compute && !in_compute_d

    when(state_reg === ClState.LOAD_ACT || state_reg === ClState.IDLE || compute_enter) {
      sfu_cnt := 0
    } elsewhen (io.sfu_ctrl.valid) {
      sfu_cnt := sfu_cnt + 1
    }

    row_sfu_done :=
      (state_reg === ClState.SFU_DRAIN) &&
        (sfu_cnt >= row_target) &&
        (row_target =/= 0)
  }

  // =========================================================
  // Area 8: OBuf + OutDMA（使用 Y_INDEX + FO_STRIDE 做全局 OFM 坐标转换）
  // =========================================================
  val obuf_outdma_area: Area = new Area {
    val out_cmd = out_dma_stream

    out_cmd.valid := False
    out_cmd.payload.base := 0
    out_cmd.payload.length := 0
    out_cmd.payload.repeat := 0
    out_cmd.payload.stride := 0

    io.obuf_ctrl.flush :=
      tile_start_pulse && (state_reg === ClState.IDLE)

    val addrWidth = cfg.dmaConfig.addrWidth
    val cmd_fifo =
      StreamFifo(UInt(addrWidth bits), depth = cfg.obufDepthWords)
    cmd_fifo.io.flush := io.obuf_ctrl.flush || cl_flush

    val current_group_idx = kernel_group_idx_reg

    val group_offset_cur_reg = Reg(UInt(addrWidth bits)) init (0)
    val group_offset_lat_reg = Reg(UInt(addrWidth bits)) init (0)

    val row_done_d =
      RegNext(io.mac_dp_ctrl.row_done && cl_enable) init (False)

    val in_compute = state_reg === ClState.COMPUTE
    val in_compute_d = RegNext(in_compute) init (False)
    val compute_enter = in_compute && !in_compute_d

    val last_group_idx =
      (tile_cfg_reg.c4_out - 1).resize(current_group_idx.getWidth)

    when(compute_enter || (state_reg === ClState.IDLE) || cl_flush) {
      group_offset_cur_reg := 0
      group_offset_lat_reg := 0
    }

    when(io.mac_dp_ctrl.row_done && cl_enable) {
      group_offset_lat_reg := group_offset_cur_reg
      when(current_group_idx =/= last_group_idx) {
        group_offset_cur_reg :=
          (group_offset_cur_reg + fo_plane_stride_bytes).resize(addrWidth)
      }
    }

    cmd_fifo.io.push.valid := False
    cmd_fifo.io.push.payload := 0
    when(row_done_d) {
      val base = uop_reg.precalc.fo_addr_tile.resize(addrWidth)
      val addr_sum =
        (base + out_row_offset_bytes_reg + group_offset_lat_reg)
          .resize(addrWidth)

      cmd_fifo.io.push.valid := True
      cmd_fifo.io.push.payload := addr_sum
    }

    if (GenerationFlags.simulation) {
      when(cmd_fifo.io.push.valid && !cmd_fifo.io.push.ready) {
        assert(False, "OutDMA cmd_fifo overflow: push while not ready")
      }
    }

    when(cmd_fifo.io.pop.valid && cl_enable) {
      out_cmd.valid := True
      out_cmd.payload.base := cmd_fifo.io.pop.payload
      out_cmd.payload.length :=
        tile_cfg_reg.w_tile.resize(out_cmd.payload.length.getWidth)
      out_cmd.payload.repeat := 1
      out_cmd.payload.stride := 0

      cmd_fifo.io.pop.ready := out_cmd.ready
    } otherwise {
      cmd_fifo.io.pop.ready := False
    }

    when(cl_flush) {
      cmd_fifo.io.pop.ready := False
    }

    // 写回收尾保护（drain grace）：
    // OutDMA 的全局写回路径通常包含写数据 FIFO / AHB 尾部；即使 OBuf 已空、命令也已发完，
    // 最后几个 word 仍可能在途。这里在 DRAIN_OBUF 额外保留若干拍，避免尾部丢失/下游短暂停顿。
    val DRAIN_GRACE_CYCLES = 4
    val drain_grace_cnt =
      Reg(UInt(log2Up(DRAIN_GRACE_CYCLES + 1) bits)) init (0)

    val drain_prereq =
      io.obuf_ctrl.empty && (cmd_fifo.io.occupancy === 0)

    when((state_reg =/= ClState.DRAIN_OBUF) || cl_flush) {
      drain_grace_cnt := 0
    } elsewhen (!drain_prereq) {
      drain_grace_cnt := 0
    } elsewhen (drain_grace_cnt =/= U(DRAIN_GRACE_CYCLES)) {
      drain_grace_cnt := drain_grace_cnt + 1
    }

    drain_done :=
      (state_reg === ClState.DRAIN_OBUF) &&
        drain_prereq &&
        (drain_grace_cnt === U(DRAIN_GRACE_CYCLES))
  }
}
