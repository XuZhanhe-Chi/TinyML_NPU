package venuscore.ctrl

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._
import venuscore.cluster._

/**
 * CtrlUopFetch
 * ------------
 * 功能：
 * 1. 通过 fetch_req Stream 接收任务请求；
 * 2. 通过 DMA 从主存抓取 32B 定长 uOP 序列；
 * 3. 按最新 ISA 解码（W0/W1/W2/W3~W7）并计算 Param Block 偏移；
 * 4. 输出解码后的 ClusterUop 到下游 FIFO。
 */
case class CtrlUopFetch(cfg: CtrlConfig) extends Component {
  val io = new Bundle {
    // ---- Control Interface ----
    // fetch_req.valid 作为 start 信号，payload 中给出 base + count
    val fetch_req = slave(Stream(CtrlFetchReq(cfg)))

    // 侧带控制信号（可用于立即打断当前抓取）
    val abort = in Bool()
    val soft_reset = in Bool()

    // Status to CtrlCore
    val busy = out Bool()
    val error = out Bool()

    // ---- DMA Interface (Read Only) ----
    val dma = master(DmaFetchPort(cfg.dmaConfig))

    // ---- Downstream Interface ----
    val uop_out = master(Stream(ClusterUop(cfg.clusterConfig)))

    // ---- 调试/DFT 观测口（只读，不参与控制）----
    // 说明：用于把内部状态机导出到 CtrlTop，便于软件在 FPGA 上定位卡点。
    val dbg_state = out(Bits(4 bits)) // 抓取/解码状态机编码
  }

  noIoPrefix()

  // ========================================================================
  // 通用信号
  // ========================================================================
  val dma_data_done = Bool()
  val geom_done = Bool()
  val fifo_push_fire = Bool()

  // uOP 隐含几何信息输出寄存器（由 geometryArea 填充）
  val geom_ifm_h_reg = Reg(UInt(12 bits)) init (0)
  val geom_ifm_w_reg = Reg(UInt(12 bits)) init (0)

  // uOP 预计算常量输出寄存器（由 geometryArea 在 DONE 时统一填充）
  val precalc_in_row_pixels_reg = Reg(UInt(12 bits)) init (0)
  val precalc_in_row_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val precalc_ibuf_line_words_reg =
    Reg(UInt(cfg.clusterConfig.ibufAddrWidth bits)) init (0)
  val precalc_fi_stride_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val precalc_row_out_words_reg = Reg(UInt(16 bits)) init (0)
  val precalc_row_stride_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val precalc_fo_plane_stride_bytes_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  private val wbuf_ctrl_width =
    cfg.clusterConfig.wbufAddrWidth + log2Up(cfg.clusterConfig.laneNum)
  val precalc_total_kernel_cnt_reg =
    Reg(UInt(wbuf_ctrl_width bits)) init (0)
  val precalc_wgt_dma_len_words_reg =
    Reg(UInt(cfg.dmaConfig.maxWordWidth bits)) init (0)
  val precalc_fi_addr_tile_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val precalc_fo_addr_tile_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)
  val precalc_fi_last_row_addr_reg =
    Reg(UInt(cfg.dmaConfig.addrWidth bits)) init (0)

  object State extends SpinalEnum {
    val IDLE, REQ_DMA, WAIT_DATA, GEOM, PUSH_FIFO = newElement()
  }

  // ========================================================================
  // Area 1: Control Logic & State Machine
  // ========================================================================
  val ctrlArea = new Area {
    val state = RegInit(State.IDLE)

    val cnt_uops_processed = Reg(UInt(16 bits)) init (0)
    // 锁存 fetch_req 中的 payload
    val reg_uop_count = Reg(UInt(16 bits)) init (0)
    val reg_curr_dma_addr = Reg(UInt(cfg.addrWidth bits)) init (0)

    val reg_error = RegInit(False)
    val is_busy = state =/= State.IDLE

    // 默认不接收请求，仅在 IDLE 时 ready
    io.fetch_req.ready := False

    switch(state) {
      is(State.IDLE) {
        io.fetch_req.ready := True

        when(io.soft_reset) {
          reg_error := False
        } elsewhen (io.fetch_req.valid) {
          // 握手成功，锁存参数
          reg_curr_dma_addr := io.fetch_req.payload.uop_base
          reg_uop_count := io.fetch_req.payload.uop_count
          cnt_uops_processed := 0

          when(io.fetch_req.payload.uop_count > 0) {
            state := State.REQ_DMA
          }
        }
      }

      is(State.REQ_DMA) {
        // 发出 DMA cmd
        when(io.dma.cmd.fire) {
          state := State.WAIT_DATA
        }
      }

      is(State.WAIT_DATA) {
        // 等待 6 个 word 全部返回
        when(dma_data_done) {
          state := State.GEOM
        }
      }

      is(State.GEOM) {
        // 计算 uOP 隐含的通用几何参数（减少 ClusterCtrl 的组合逻辑）
        when(geom_done) {
          state := State.PUSH_FIFO
        }
      }

      is(State.PUSH_FIFO) {
        // 推送解码后的 uOP 到 FIFO
        when(fifo_push_fire) {
          cnt_uops_processed := cnt_uops_processed + 1
          reg_curr_dma_addr := reg_curr_dma_addr + 32 // 每条 uOP 32B

          when(cnt_uops_processed === reg_uop_count - 1) {
            state := State.IDLE
          } otherwise {
            state := State.REQ_DMA
          }
        }
      }
    }

    // 异步 Abort/Reset
    when(io.abort || io.soft_reset) {
      state := State.IDLE
    }

    io.busy := is_busy
    io.error := reg_error
  }

  // Debug taps：导出状态机编码（避免父模块直接读取内部 directionless 信号导致层级违规）
  io.dbg_state := ctrlArea.state.asBits.resized

  // ========================================================================
  // Area 2: DMA Fetch Logic
  // ========================================================================
  val dmaFetchArea = new Area {
    val word_buffer = Vec(Reg(Bits(32 bits)), 8)
    val cnt_words = Reg(UInt(3 bits))

    io.dma.cmd.valid := (ctrlArea.state === State.REQ_DMA)
    io.dma.cmd.payload.base := ctrlArea.reg_curr_dma_addr
    io.dma.cmd.payload.length := 8 // 8 words = 32 bytes
    io.dma.cmd.payload.repeat := 1
    io.dma.cmd.payload.stride := 0

    io.dma.data.ready := (ctrlArea.state === State.WAIT_DATA)

    when(ctrlArea.state === State.REQ_DMA) {
      cnt_words := 0
    } elsewhen (io.dma.data.fire) {
      word_buffer(cnt_words) := io.dma.data.payload
      cnt_words := cnt_words + 1
    }

    dma_data_done :=
      (ctrlArea.state === State.WAIT_DATA) &&
        (io.dma.data.fire && cnt_words === 7)
  }

  // ========================================================================
  // Area 3: Decode & Param Block Address Calculation
  // ========================================================================
  val decodeArea = new Area {
    val raw = dmaFetchArea.word_buffer
    val decoded = ClusterUop(cfg.clusterConfig)

    // --- 3.1 原始字段提取（按最新 ISA 版本）---
    val w0 = raw(0)
    val w1 = raw(1)
    val w2 = raw(2)
    val w3 = raw(3)
    val w4 = raw(4)
    val w5 = raw(5)
    val w6 = raw(6)
    val w7 = raw(7)

    // W0: OPCODE / ACT / FLAG / STRIDE / PAD / H_TILE / W_TILE
    val opcode = w0(3 downto 0).asUInt
    val act_type = w0(6 downto 4).asUInt
    val first_flag = w0(7)
    val last_flag = w0(8)
    val stride = w0(10 downto 9).asUInt
    val pad_top = w0(11)
    val pad_bot = w0(12)
    val pad_left = w0(13)
    val pad_right = w0(14)
    // 按 ISA：H_TILE=W0[22:15]（8bit），W_TILE=W0[30:23]（8bit）
    val h_tile = w0(22 downto 15).asUInt
    val w_tile = w0(30 downto 23).asUInt
    val sync = w0(31)

    // W1: C4_IN / C4_OUT / Y_INDEX / QMODE
    val c4_in = w1(9 downto 0).asUInt
    val c4_out = w1(19 downto 10).asUInt
    val y_index = w1(29 downto 20).asUInt
    val qmode = w1(31 downto 30).asUInt

    // W2: FI_STRIDE / FO_STRIDE (各 16bit)
    val fi_stride = w2(15 downto 0).asUInt
    val fo_stride = w2(31 downto 16).asUInt

    // W3–W5: 地址
    val param_addr = w3.asUInt
    val fi_addr = w4.asUInt
    val fo_addr = w5.asUInt

    // W6: IFM_W/IFM_H（显式输入空间尺寸）
    val ifm_w = w6(15 downto 0).asUInt
    val ifm_h = w6(31 downto 16).asUInt

    // W7: DMA 预计算提示字段（word 计数）
    val actdma_line_words = w7(15 downto 0).asUInt
    val outdma_line_words = w7(31 downto 16).asUInt

    // --- 3.2 字段映射到 ClusterUop ---

    import CtrlIsaOpCode._

    decoded.opcode := ClusterUopOpcode.RESERVED
    switch(opcode) {
      is(CMD_NOP) {
        decoded.opcode := ClusterUopOpcode.NOP
      }
      is(CMD_CONV3x3) {
        decoded.opcode := ClusterUopOpcode.CONV2D
      }
      is(CMD_PW1x1) {
        decoded.opcode := ClusterUopOpcode.PWCONV
      }
      is(CMD_DW3x3) {
        decoded.opcode := ClusterUopOpcode.DWCONV
      }
      is(CMD_AVGPOOL) {
        decoded.opcode := ClusterUopOpcode.AVGPOOL
      }
      is(CMD_MAXPOOL) {
        decoded.opcode := ClusterUopOpcode.MAXPOOL
      }
      is(CMD_MATMUL) {
        decoded.opcode := ClusterUopOpcode.MATMUL
      }
      // MAXPOOL（CMD_MAXPOOL）如果未来实现，再在此扩展
    }

    decoded.act_type := ClusterActType.NONE
    switch(act_type) {
      is(U(1, 3 bits)) {
        decoded.act_type := ClusterActType.RELU
      }
      is(U(2, 3 bits)) {
        decoded.act_type := ClusterActType.RELU6
      }
    }

    decoded.first_flag := first_flag
    decoded.last_flag := last_flag
    decoded.stride := stride
    decoded.sync := sync

    decoded.top_pad := pad_top
    decoded.bot_pad := pad_bot
    decoded.left_pad := pad_left
    decoded.right_pad := pad_right

    decoded.h_tile := h_tile.resized
    decoded.w_tile := w_tile.resized
    decoded.c4_in := c4_in.resized
    decoded.c4_out := c4_out.resized

    decoded.y_index := y_index.resized
    // ---- qmode: 2-bit → 枚举 ----
    decoded.qmode := ClusterQMode.Q8
    switch(qmode) {
      is(U"00") {
        decoded.qmode := ClusterQMode.Q8
      }
      is(U"01") {
        decoded.qmode := ClusterQMode.Q4
      }
      is(U"10") {
        decoded.qmode := ClusterQMode.Q2
      }
      is(U"11") {
        decoded.qmode := ClusterQMode.RESERVED
      }
    }

    decoded.fi_stride := fi_stride.resized
    decoded.fo_stride := fo_stride.resized

    decoded.fi_addr := fi_addr.resized
    decoded.fo_addr := fo_addr.resized

    // --- 3.3 参数块地址计算（coe_addr / w_addr）---
    // 约定：
    //   Cout_tile ≈ C4_OUT * 4
    //   每个输出通道量化参数大小 Q_BYTES（实现内部常量，如 8B）
    //   Q_SIZE = Cout_tile * Q_BYTES = C4_OUT * 32
    //
    // 这里先按 INT8 路径：Q_BYTES = 8 → 每个 c4 group = 32B
    // 后续若根据 qmode 做不同 Q_BYTES，可在此扩展。
    val q_size_bytes = UInt(32 bits)
    // C4_OUT * 32 = c4_out << 5
    q_size_bytes := (c4_out.resize(32) << 5).resized

    val weight_base_raw = param_addr + q_size_bytes
    // ALIGN_BYTES = 16：权重按 16B 对齐
    val weight_base_aligned = (weight_base_raw + 15) & ~U(15, 32 bits)

    decoded.coe_addr := param_addr.resized
    decoded.w_addr := weight_base_aligned.resized

    // geom：由 geometryArea 计算并填充
    decoded.geom.ifm_h := geom_ifm_h_reg.resized
    decoded.geom.ifm_w := geom_ifm_w_reg.resized

    // precalc：由 geometryArea 在 DONE 时统一填充
    decoded.precalc.in_row_pixels := precalc_in_row_pixels_reg.resized
    decoded.precalc.in_row_bytes := precalc_in_row_bytes_reg.resized
    decoded.precalc.ibuf_line_words := precalc_ibuf_line_words_reg.resized
    decoded.precalc.fi_stride_bytes := precalc_fi_stride_bytes_reg.resized
    decoded.precalc.row_out_words := precalc_row_out_words_reg.resized
    decoded.precalc.row_stride_bytes := precalc_row_stride_bytes_reg.resized
    decoded.precalc.fo_plane_stride_bytes :=
      precalc_fo_plane_stride_bytes_reg.resized
    decoded.precalc.total_kernel_cnt := precalc_total_kernel_cnt_reg.resized
    decoded.precalc.wgt_dma_len_words := precalc_wgt_dma_len_words_reg.resized
    decoded.precalc.fi_addr_tile := precalc_fi_addr_tile_reg.resized
    decoded.precalc.fo_addr_tile := precalc_fo_addr_tile_reg.resized
    decoded.precalc.fi_last_row_addr := precalc_fi_last_row_addr_reg.resized
  }

  // ========================================================================
  // Area 3.5: uOP 通用几何/预计算（多周期，避免大组合逻辑）
  // ========================================================================
  val geometryArea = new Area {
    import CtrlIsaOpCode._

    // 复用 DMA buffer 里的原始 word
    val raw = dmaFetchArea.word_buffer
    val w0 = raw(0)
    val w1 = raw(1)
    val w2 = raw(2)
    val w4 = raw(4)
    val w5 = raw(5)
    val w6 = raw(6)
    val w7 = raw(7)

    val opcode_raw = w0(3 downto 0).asUInt
    val stride_raw = w0(10 downto 9).asUInt
    // 按 ISA：H_TILE=W0[22:15]（8bit），W_TILE=W0[30:23]（8bit）
    val w_tile = w0(30 downto 23).asUInt

    val fi_stride_words = w2(15 downto 0).asUInt
    val fo_stride_words = w2(31 downto 16).asUInt

    // W1: C4_IN / C4_OUT / Y_INDEX
    val c4_in = w1(9 downto 0).asUInt
    val c4_out = w1(19 downto 10).asUInt
    val y_index = w1(29 downto 20).asUInt

    // W4/W5: FI/FO base（byte address）
    val fi_addr_raw = w4.asUInt
    val fo_addr_raw = w5.asUInt

    // W6: IFM_W/IFM_H
    val ifm_w_raw = w6(15 downto 0).asUInt
    val ifm_h_raw = w6(31 downto 16).asUInt

    // W7: DMA 预计算提示字段（word 计数）
    val actdma_line_words_raw = w7(15 downto 0).asUInt
    val outdma_line_words_raw = w7(31 downto 16).asUInt

    val is_conv3 = opcode_raw === CMD_CONV3x3
    val is_dw3 = opcode_raw === CMD_DW3x3

    // 注意：stride 在 uOP 里编码为实际步长值（常用 1/2），这里显式指定常量位宽，避免比较出现隐式扩展/截断问题
    val stride_is_2 = stride_raw === U(2, stride_raw.getWidth bits)

    val in_geom = ctrlArea.state === State.GEOM
    val in_geom_d = RegNext(in_geom) init (False)
    val geom_enter = in_geom && !in_geom_d

    object GeomState extends SpinalEnum {
      val IDLE, DONE = newElement()
    }

    val geom_state_reg = RegInit(GeomState.IDLE)

    // 输出寄存器：IFM_H/IFM_W（单位：像素，不含 padding）
    val ifm_h_reg = geom_ifm_h_reg
    val ifm_w_reg = geom_ifm_w_reg

    // 状态输出给 ctrlArea
    geom_done := (ctrlArea.state === State.GEOM) && (geom_state_reg === GeomState.DONE)

    when(!in_geom) {
      geom_state_reg := GeomState.IDLE
    } elsewhen (geom_enter) {
      // 直接锁存 ISA W6 给出的 IFM 尺寸，避免硬件反推（除法/候选枚举）
      ifm_w_reg := ifm_w_raw.resized
      ifm_h_reg := ifm_h_raw.resized
      geom_state_reg := GeomState.DONE
    }

    when(in_geom) {
      switch(geom_state_reg) {
        is(GeomState.IDLE) {
          // 等待 geom_enter
        }

        is(GeomState.DONE) {
          // 保持输出稳定，等待 ctrlArea 离开 GEOM
          //
          // 同时在 DONE 状态统一填充 “precalc” 一组常量，确保：
          // - 依赖 IFM_W/H 的字段使用最终值；
          // - 下游在 PUSH_FIFO 阶段拿到一致的一组参数。
          val addr_width = cfg.dmaConfig.addrWidth

          // 1) IBUF/ActDMA：输入一行像素/字节数
          precalc_in_row_pixels_reg := geom_ifm_w_reg.resized
          val in_row_bytes_next =
            (geom_ifm_w_reg.resize(addr_width) |<< 2).resize(addr_width)
          precalc_in_row_bytes_reg := in_row_bytes_next

          // 2) ActDMA 行读取长度（word）：直接使用 ISA W7 的提示字段，避免乘法链路
          precalc_ibuf_line_words_reg :=
            actdma_line_words_raw.resize(cfg.clusterConfig.ibufAddrWidth bits)

          // 3) FI/FO stride（word→byte）
          precalc_fi_stride_bytes_reg :=
            (fi_stride_words.resize(addr_width) |<< 2).resize(addr_width)

          val row_stride_bytes_next =
            (w_tile.resize(addr_width) |<< 2).resize(addr_width)
          precalc_row_stride_bytes_reg := row_stride_bytes_next

          val fo_plane_stride_bytes_next =
            (fo_stride_words.resize(addr_width) |<< 2).resize(addr_width)
          precalc_fo_plane_stride_bytes_reg := fo_plane_stride_bytes_next

          // 4) WgtDMA：kernel 总数（= c4_out * laneNum）
          val total_kernel_cnt_next =
            (c4_out.resize(wbuf_ctrl_width) * U(cfg.clusterConfig.laneNum, wbuf_ctrl_width bits))
              .resize(wbuf_ctrl_width)
          precalc_total_kernel_cnt_reg := total_kernel_cnt_next

          // 4.1) WgtDMA：权重阶段 DMA length（word 数）
          // 说明：
          // - 该值仅在需要加载权重的算子上有效（CONV/PW/DW/MATMUL）。
          // - 预先计算后下发到 ClusterCtrl，可避免每个 cluster 内综合乘法器。
          val wgt_len_words = UInt(cfg.dmaConfig.maxWordWidth bits)
          wgt_len_words := 0

          when(is_dw3) {
            // DW3x3：每 kernel 固定 3 word
            wgt_len_words := 3
          } elsewhen (is_conv3) {
            // CONV3x3：每 kernel = c4_in * 9
            val c4_in_u = c4_in.resize(cfg.dmaConfig.maxWordWidth)
            wgt_len_words := ((c4_in_u |<< 3) + c4_in_u).resized
          } elsewhen (opcode_raw === CMD_PW1x1 || opcode_raw === CMD_MATMUL) {
            // PW1x1 / MATMUL：每 kernel = c4_in
            wgt_len_words := c4_in.resize(cfg.dmaConfig.maxWordWidth)
          }

          val wgt_dma_len_full =
            (wgt_len_words.resize(cfg.dmaConfig.maxWordWidth + wbuf_ctrl_width) *
              total_kernel_cnt_next.resize(cfg.dmaConfig.maxWordWidth + wbuf_ctrl_width))
              .resized
          precalc_wgt_dma_len_words_reg :=
            wgt_dma_len_full.resize(cfg.dmaConfig.maxWordWidth)

          // 5) SFU：每行输出 word 数：直接使用 ISA W7 的提示字段，避免乘法链路
          precalc_row_out_words_reg := outdma_line_words_raw.resize(16 bits)

          // 6) 预计算 tile base（把 Y_INDEX 对齐提前做掉）
          //    统一规则：fi_addr_tile 始终包含 y_index*stride（以及 rolling halo 的 -1 调整），
          //    ClusterCtrl 侧只需要做 tile 内行偏移的累加/饱和，不再叠加 y_index 偏移。
          val y_index_stride_u12 = UInt(12 bits)
          y_index_stride_u12 := y_index.resize(12 bits)
          when(stride_is_2) {
            y_index_stride_u12 := (y_index.resize(12 bits) |<< 1).resize(12 bits)
          }

          val in_row_bytes = in_row_bytes_next
          // stride=1 的 CONV/DW(3x3) 如果发生 Y_INDEX tiling（y_index>0），
          // tile 的第一行输出需要访问“上一行”的 halo（3x3 的窗口上沿）。
          // 如果 fi_addr_tile 从 y_index 行起始，会导致 tile 起始处（例如 row8）少取一行，输出不对。
          // 这里把 fi_addr_tile 的行基准调整为 (y_index - 1)（y_index=0 保持不变），以覆盖 halo。
          val fi_base_row_u12 = UInt(12 bits)
          fi_base_row_u12 := y_index_stride_u12
          when((is_conv3 || is_dw3) && !stride_is_2 && (y_index =/= 0)) {
            fi_base_row_u12 := (y_index_stride_u12 - 1).resized
          }

          val fi_y_offset =
            (fi_base_row_u12.resize(addr_width) * in_row_bytes).resize(addr_width)
          val fo_y_offset =
            (y_index.resize(addr_width) * row_stride_bytes_next).resize(addr_width)

          // fi_addr_tile：对齐到 tile 所在的输入起始行（y_index*stride），并包含 rolling halo 的 -1 调整。
          // 统一规则：无论 opcode/stride，均写入 fi_addr_raw + fi_y_offset，避免 Cluster 侧再叠加 y_index 偏移。
          precalc_fi_addr_tile_reg :=
            (fi_addr_raw.resize(addr_width) + fi_y_offset).resize(addr_width)

          precalc_fo_addr_tile_reg :=
            (fo_addr_raw.resize(addr_width) + fo_y_offset).resize(addr_width)

          // IFM last row address：fi_addr_raw + (ifm_h-1) * in_row_bytes
          // 用于 Cluster 侧做地址饱和（pad_bottom/越界），避免每个 cluster 综合乘法器。
          val last_row_idx_u12 =
            (geom_ifm_h_reg - 1).resized
          val last_row_off =
            (last_row_idx_u12.resize(addr_width) * in_row_bytes).resize(addr_width)
          precalc_fi_last_row_addr_reg :=
            (fi_addr_raw.resize(addr_width) + last_row_off).resize(addr_width)
        }
      }
    }

    when(io.abort || io.soft_reset) {
      geom_state_reg := GeomState.IDLE
    }
  }

  // ========================================================================
  // Area 4: Output FIFO
  // ========================================================================
  val outputArea = new Area {
    // 注意：
    // - 这里必须提供至少 1-deep 的缓存来正确处理下游 backpressure。
    // - 若 depth=0（纯组合透传），当 Scheduler/Cluster 拉低 ready 时，uOP 可能被覆盖导致“漏发 uOP”，
    //   表现为：实际执行的 uOP 数量小于 UOP_COUNT，进而出现 KWS top1 偏差/回归失败。
	    val fifo = StreamFifo(ClusterUop(cfg.clusterConfig), depth = 1)
	    // depth=1 时 StreamFifo 本身就是寄存器实现；不要访问内部 ram/logic，避免不同实现下出现空指针。

    fifo.io.push.valid := (ctrlArea.state === State.PUSH_FIFO)
    fifo.io.push.payload := decodeArea.decoded

    fifo_push_fire := fifo.io.push.fire

    io.uop_out << fifo.io.pop
  }
}

object CtrlUopFetch {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl",
      headerWithDate = false
    ).generateVerilog(CtrlUopFetch(VenusCoreConfig.default.ctrlCfg))
  }
}
