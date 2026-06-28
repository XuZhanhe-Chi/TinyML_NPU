package venuscore.cluster

import spinal.core._
import spinal.lib._

/**
  * ClusterFrontend（前端：uOP 接收/解码）
  *
  * - 仅在 Cluster 处于 IDLE 时接收 uOP stream；
  * - 锁存 1 条 uOP，并解码写入 `uop_reg` + `tile_cfg_reg`；
  * - 产生 `tile_start_pulse` 并完成 tile 级循环状态复位。
  */
final class ClusterFrontend(val top: ClusterCtrl) extends Area {
  import top._

  // =========================================================
  // Area 2：uOP 锁存/解码 + tile 初始化
  // =========================================================
  val uop_area: Area = new Area {
    val uop_stream = io.uop_data
    val uop_latched = Reg(ClusterUop(cfg)) init (ClusterUop.resetValue(cfg))
    val uop_latched_valid = Reg(Bool()) init (False)

    // 仅在 IDLE 拉取下一条 uOP（1-deep latch）
    val can_accept_uop = (state_reg === ClState.IDLE) && !uop_latched_valid
    uop_stream.ready := can_accept_uop

    val uop_fire = uop_stream.valid && uop_stream.ready
    when(uop_fire) {
      uop_latched := uop_stream.payload
      uop_latched_valid := True
    }

    when(cl_flush) {
      uop_latched_valid := False
    }

    val decode_fire = uop_latched_valid

    tile_start_pulse := False
    act_reuse_next := False
    wgt_reuse_next := False

    when(decode_fire) {
      val u = uop_latched

      val is_cfg = u.opcode === ClusterUopOpcode.CFG
      val is_nop = u.opcode === ClusterUopOpcode.NOP
      val is_conv = u.opcode === ClusterUopOpcode.CONV2D
      val is_pw = u.opcode === ClusterUopOpcode.PWCONV
      val is_dw = u.opcode === ClusterUopOpcode.DWCONV
      val is_matmul = u.opcode === ClusterUopOpcode.MATMUL
      val is_avg = u.opcode === ClusterUopOpcode.AVGPOOL
      val is_max = u.opcode === ClusterUopOpcode.MAXPOOL
      val is_pool = is_avg || is_max

      val is_compute_uop_next =
        is_conv || is_pw || is_dw || is_matmul || is_pool

      // uOP decode 完成，释放 latch（单拍缓存）
      uop_latched_valid := False

      when(is_cfg) {
        // CFG uOP：当前版本保留入口，不做额外动作
      } elsewhen (is_nop) {
        // NOP：忽略
      } elsewhen (is_compute_uop_next) {
        // -------------------------
        // 锁存 compute uOP
        // -------------------------
        tile_start_pulse := True
        // 兼容既有命名：MAXPOOL 也走 "pool" 路径
        tile_is_avg_pulse := is_pool
        uop_reg := u

        // Tile 几何配置
        tile_cfg_reg.h_tile := u.h_tile
        tile_cfg_reg.w_tile := u.w_tile
        tile_cfg_reg.c4_in := u.c4_in
        tile_cfg_reg.c4_out := u.c4_out
        tile_cfg_reg.y_index := u.y_index
        tile_cfg_reg.stride := u.stride
        tile_cfg_reg.pad_top := u.top_pad
        tile_cfg_reg.pad_bot := u.bot_pad
        tile_cfg_reg.pad_left := u.left_pad
        tile_cfg_reg.pad_right := u.right_pad
        tile_cfg_reg.fo_stride := u.fo_stride

        // kernel_taps 推导：Conv/DW=9, PW/FC=1, AvgPool(2x2)=4
        val next_kernel_taps = UInt(4 bits)
        when(is_conv || is_dw) {
          next_kernel_taps := 9
        } elsewhen (is_pw || is_matmul) {
          next_kernel_taps := 1
        } elsewhen (is_pool) {
          next_kernel_taps := 4
        } otherwise {
          next_kernel_taps := 1
        }
        tile_cfg_reg.kernel_taps := next_kernel_taps

        val act_key_next =
          u.precalc.fi_addr_tile.asBits ##
            u.precalc.fi_stride_bytes.asBits ##
            u.precalc.ibuf_line_words.asBits ##
            u.c4_in.asBits ##
            u.stride.asBits ##
            u.top_pad.asBits ##
            u.bot_pad.asBits ##
            u.left_pad.asBits ##
            u.right_pad.asBits ##
            next_kernel_taps.asBits ##
            u.opcode.asBits

        val wgt_key_next =
          u.coe_addr.asBits ##
            u.w_addr.asBits ##
            u.c4_in.asBits ##
            u.c4_out.asBits ##
            next_kernel_taps.asBits ##
            u.qmode.asBits ##
            u.opcode.asBits

        val act_reuse_hit =
          !is_pool &&
            act_valid_reg &&
            (act_key_reg === act_key_next) &&
            (u.h_tile === U(1, u.h_tile.getWidth bits))
        val wgt_reuse_hit =
          wgt_valid_reg && (wgt_key_reg === wgt_key_next)

        act_reuse_reg := act_reuse_hit
        wgt_reuse_reg := wgt_reuse_hit
        act_reuse_next := act_reuse_hit
        wgt_reuse_next := wgt_reuse_hit
        act_key_pending_reg := act_key_next
        wgt_key_pending_reg := wgt_key_next

        cfg_valid_reg := True

        // -------------------------
        // tile 级循环状态初始化
        // -------------------------
        y_out_reg := 0
        pw_row_offset_bytes_reg := 0
        out_row_offset_bytes_reg := 0
        avg_row0_offset_bytes_reg := 0

        when(!act_reuse_next) {
          avg_row0_ready_reg := False
          avg_next_row0_prefetched_reg := False
          avg_next_row0_y_out_reg := 0
        }

        // PW ping-pong init：始终从 bank0 开始读
        pw_rd_bank_reg := False
        when(!act_reuse_next) {
          pw_bank_valid_reg(0) := False
          pw_bank_valid_reg(1) := False
          pw_bank_row_reg(0) := 0
          pw_bank_row_reg(1) := 0
        }

        // ActDMA 行地址累加器初始化（使用 u.precalc 的 tile base）
        val fi_addr_tile_u =
          u.precalc.fi_addr_tile.resize(cfg.dmaConfig.addrWidth bits)
        s2_row0_addr_reg := fi_addr_tile_u
        roll_bot_addr_reg := fi_addr_tile_u
        seq_row_addr_reg := fi_addr_tile_u
        // AVGPOOL 行基址初始化（row0/row1）
        val addrWidth = cfg.dmaConfig.addrWidth
        val fi_addr_tile_base = fi_addr_tile_u.resize(addrWidth)
        val fi_last_row_addr = u.precalc.fi_last_row_addr.resize(addrWidth)
        val in_row_bytes_addr = u.precalc.in_row_bytes.resize(addrWidth)

        def satAddAddr(base: UInt, inc: UInt): UInt = {
          val sum = (base + inc).resize(addrWidth)
          (sum > fi_last_row_addr) ? fi_last_row_addr | sum
        }

        avg_row0_base_addr_reg := fi_addr_tile_base
        avg_row1_base_addr_reg := satAddAddr(fi_addr_tile_base, in_row_bytes_addr)

        // rolling CONV/DW：初始化 y_out=0 时 bottom 行地址
        val next_is_rolling =
          (is_conv || is_dw) && (u.stride === U(1)) && (next_kernel_taps === U(9))
        when(next_is_rolling) {
          val off1 = in_row_bytes_addr
          val off2 = (in_row_bytes_addr |<< 1).resized
          val bottom_y0 =
            u.top_pad ? satAddAddr(fi_addr_tile_base, off1) | satAddAddr(fi_addr_tile_base, off2)
          roll_bot_addr_reg := bottom_y0.resized
        }

        // IBUF 行 ID 初始化：AvgPool 使用 (mid/bot/top) 作为 (row0/row1/next-row0)
        when(is_avg) {
          rowid_mid_reg := 0
          rowid_bot_reg := 1
          rowid_top_reg := 2
        } otherwise {
          rowid_top_reg := 0
          rowid_mid_reg := 1
          rowid_bot_reg := 2
        }
      }
    }

    // IBUF 配置：c4_in / 行宽度有效部分
    io.ibuf_ctrl.cfg_c4 :=
      tile_cfg_reg.c4_in.resize(io.ibuf_ctrl.cfg_c4.getWidth)
    io.ibuf_ctrl.cfg_valid := cfg_valid_reg
  }
}
