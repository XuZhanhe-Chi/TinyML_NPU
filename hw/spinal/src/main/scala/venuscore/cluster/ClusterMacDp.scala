package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * ClusterMacDp（MAC 数据通路控制器）
 * ---------------------------------------------------------------------
 * 作用：单个 Cluster 内部的 MAC 数据平面控制（Mac-DP 控制器）。
 *
 * 1) CFG_SFU 阶段（state = CFG_SFU）
 *    - 从 WBuf 中预取当前 tile 的量化参数（bias/scale/shift），配置 SFU。
 *    - Pooling（AVG2/AVG3）不依赖外部量化参数，直接配置 SFU 为“除以 4/8/9”。
 *
 * 2) RUN 阶段（state = RUN）
 *    - Stage 0 (S0)：只负责地址和控制：
 *      * 计算 IBUF 访问坐标 (row_id, x)、padding 区域标志、DW3x3 的 top/mid/bot 选择；
 *      * 计算 WBUF 权重地址（支持 CONV3x3 / PW1x1 / DW3x3 / AVG2x2 / AVG3x3）；
 *      * 打一拍寄存器后驱动 IBUF / WBUF 读端口；
 *      * 根据 tap_cnt / g_cnt / x_out_cnt 产生 start_pixel / pixel_done。
 *
 *    - 反压桥接（Backpressure Bridge）：
 *      * S0 仅把 ctrl + 读地址推入“小宽度 request FIFO”（depth = fifoDepth）；
 *      * IBUF/WBUF 读回的数据进入 1-deep 数据缓冲（FifoPayload），再送给 S1；
 *      * 使用“基于 credit 的限流”：ahead_cnt 表示 S0 领先“读请求发射”的 token 数，
 *      S0 最多领先 maxAhead 个 token，减少 ready 反压的组合路径。
 *
 *    - Stage 1 (S1)：数据分发与 Lane 控制：
 *      * 从 FIFO 中取出 activations/weights + ctrl；
 *      * 非 DW 模式：每个 lane 共用同一 4×int8 宽度的 scalar 向量；
 *      * DW3x3 模式：对 {top, mid, bot} 三行逐 lane 拆字节，按 {top, mid, bot, 0}
 *      形式重新拼装为 4-tap 向量，并在 top/bot 行 padding 时按行清零；
 *      * X 方向或（CONV/AVG 的）Y 方向 padding：整 tap 全零，但仍然发出 start_pixel/pixel_done；
 *      * 将最终的 act_vec / wgt_vec / elem_mask 和 LaneCtrlPort 发送到每个 Lane。
 *
 * 3) DRAIN 阶段（state = DRAIN）
 *    - 等待 FIFO 读空且所有 Lane busy 变为 0，发出 row_done 脉冲并返回 IDLE。
 */
case class ClusterMacDp(cfg: ClusterConfig) extends Component {

  // =====================================================================
  // 局部参数别名（Scala 常量）
  // =====================================================================
  private val lane_num = cfg.laneNum
  private val ibuf_word_width = cfg.ibufWordWidth
  private val wbuf_addr_width = cfg.wbufAddrWidth
  private val wbuf_word_width = cfg.wbufWordWidth
  private val row_id_width = cfg.ibufRowIdWidth
  private val linebuf_addr_width = cfg.lineBufAddrWidth
  private val coord_width = cfg.ibufCoordWidth

  // Request FIFO 深度与 credit 宽度
  private val fifoDepth = 4
  private val creditWidth = log2Up(fifoDepth + 1)
  // S0 最多领先“读请求发射”的 token 数（小于 FIFO 深度，保留余量）
  private val maxAheadVal = if (fifoDepth >= 3) fifoDepth - 2 else fifoDepth - 1
  private val maxAhead = U(maxAheadVal, creditWidth bits)

  // =====================================================================
  // 顶层 IO
  // =====================================================================
  val io = new Bundle {
    // 上游 ClusterCtrl 控制端口（包含 row_len, c4_in, padding, wbuf_wgt_base 等）
    val ctrl = slave(MacDpCtrlPort(cfg))

    // SFU 量化系数配置接口：一拍 valid + lane_num 路系数
    val sfu_coe = master(Flow(Vec(SfuCoe(cfg), lane_num)))

    // IBUF 读端口（行缓冲控制）
    val ibuf_rd = master(IBufRdPort(cfg))

    // WBUF 读端口（每个 lane 一路，便于 lane 级并行）
    val wbuf_rd = Vec(master(RamRdPort(wbuf_addr_width, wbuf_word_width)), lane_num)

    // Lane 控制端口（使能 / 累加 / pooling 模式 / busy 返回）
    val lane_ctrl = Vec(master(LaneCtrlPort(cfg)), lane_num)

    // Lane MAC 输入（每 lane 一路 Stream）
    val lane_mac_out = Vec(master(Stream(LaneMacIn(cfg))), lane_num)
  }
  noIoPrefix()

  // =====================================================================
  // 内部 Bundle 定义：S0→S1 的控制信息 + FIFO 负载
  // =====================================================================

  /**
   * PipeCtrlBundle：S0→S1 的像素级控制信息
   *   - S0 生成，在 S1 使用；
   *   - 携带一个 tap 对应的像素级控制信息（非通道级）。
   */
  case class PipeCtrlBundle() extends Bundle {
    val start_pixel = Bool() // 当前 tap 是否是该像素的第一个 tap（用于 Lane 清零 acc）
    val pixel_done = Bool() // 当前 tap 是否是该像素的最后一个 tap（用于 Lane 输出）
    val is_pool = Bool() // 是否是 AVGPOOL（2x2/3x3）
    val is_dw3 = Bool() // 是否是 DW3x3
    val lane_enable = Bool() // lane 使能（来自上游 ctrl.enable）
    val is_pad = Bool() // 整 tap padding（X 或 CONV/AVG 的 Y padding）
    val pad_top = Bool() // DW 模式下：top 行是否 Y padding（逐行清零）
    val pad_bot = Bool() // DW 模式下：bot 行是否 Y padding（逐行清零）
  }

  /**
   * FifoPayload：S0→S1 的 FIFO 负载（数据 + 控制）
   *   - S0 侧打包 IBUF/WBUF 数据 + PipeCtrl；
   *   - 经 FIFO 传递到 S1。
   */
  case class FifoPayload() extends Bundle {
    val ctrl = PipeCtrlBundle()
    val ibuf_data_scalar = Bits(ibuf_word_width bits) // 非 DW 模式：4×int8 作为一向量
    val ibuf_data_top = Bits(ibuf_word_width bits) // DW 模式：top 行
    val ibuf_data_mid = Bits(ibuf_word_width bits) // DW 模式：mid 行
    val ibuf_data_bot = Bits(ibuf_word_width bits) // DW 模式：bot 行
    val wbuf_data = Vec(Bits(wbuf_word_width bits), lane_num) // 每个 lane 各自的权重向量
  }

  /**
   * RdReq：S0→读接口的请求（小宽度，降低 FIFO 资源）
   *   - 携带 ctrl + IBUF/WBUF 地址信息；
   *   - 数据在读回后进入 1-deep 数据缓冲（FifoPayload）。
   */
  case class RdReq() extends Bundle {
    val ctrl = PipeCtrlBundle()
    val ibuf_x = UInt(coord_width bits)
    val ibuf_row = UInt(row_id_width bits)
    val ibuf_dw = Bool()
    val ibuf_row_top = UInt(row_id_width bits)
    val ibuf_row_mid = UInt(row_id_width bits)
    val ibuf_row_bot = UInt(row_id_width bits)
    val cgroup = UInt(coord_width bits)
    val wbuf_addr = UInt(wbuf_addr_width bits)
  }

  // =====================================================================
  // 状态机与主要寄存器
  // =====================================================================
  object MacDpState extends SpinalEnum {
    val IDLE, CFG_SFU, RUN, DRAIN = newElement()
  }

  val state_reg = Reg(MacDpState()) init (MacDpState.IDLE)
  val busy_reg = Reg(Bool()) init (False)
  val row_done_reg = Reg(Bool()) init (False)

  // ------------------------
  // Tile 几何与通道配置寄存器
  // ------------------------
  val row_len_reg = Reg(UInt(linebuf_addr_width bits)) init (0) // tile 内输出行长度（W_TILE）
  val stride_reg = Reg(UInt(2 bits)) init (1) // 空间 stride（1/2），由上游 ctrl 传入
  val c4_in_reg = Reg(UInt(coord_width bits)) init (0) // 输入通道 group 数（C4_IN）

  // 每 kernel 的 tap 数（1 / 3 / 4 / 9），来自 decodeTapCount
  val tap_count_reg = Reg(UInt(4 bits)) init (0)

  val kernel_group_idx_reg = Reg(UInt(wbuf_addr_width bits)) init (0) // 对应 Cout 子块在全局的 group idx
  val row_top_reg = Reg(UInt(row_id_width bits)) init (0) // IBUF 中 top 行 ID
  val row_mid_reg = Reg(UInt(row_id_width bits)) init (0) // IBUF 中 mid 行 ID
  val row_bot_reg = Reg(UInt(row_id_width bits)) init (0) // IBUF 中 bot 行 ID

  // Padding 相关配置
  val pad_left_reg = Reg(Bool()) init (False)
  val pad_right_reg = Reg(Bool()) init (False)
  val pad_top_en_reg = Reg(Bool()) init (False)
  val pad_bot_en_reg = Reg(Bool()) init (False)
  val ibuf_valid_w_reg = Reg(UInt(coord_width bits)) init (0) // 有效 IFM 宽度（用于 X 方向右侧 padding 判定）

  // 权重基地址（WBuf 地址空间内）
  val wgt_base_reg = Reg(UInt(wbuf_addr_width bits)) init (0)

  // ------------------------
  // S0 运行计数器
  // ------------------------
  val x_out_cnt = Reg(UInt(linebuf_addr_width bits)) init (0) // tile 内输出列计数
  val g_cnt = Reg(UInt(coord_width bits)) init (0) // 当前输入通道 group 下标
  val tap_cnt = Reg(UInt(4 bits)) init (0) // 当前 kernel 内 tap 下标

  // CFG_SFU 阶段计数器与临时缓冲（bias 低 word）
  val cfg_cnt = Reg(UInt(2 bits)) init (0)
  val bias_low_buf = Vec(Reg(Bits(32 bits)), lane_num)

  // op_mode 解码锁存（IDLE->RUN 前一次性解析）
  val is_conv3_reg = Reg(Bool()) init (False) // CONV3x3
  val is_pw1_reg = Reg(Bool()) init (False) // PW1x1
  val is_dw3_reg = Reg(Bool()) init (False) // DW3x3
  val is_avg2_reg = Reg(Bool()) init (False) // AVG2x2
  val is_avg3_reg = Reg(Bool()) init (False) // AVG3x3
  val is_max2_reg = Reg(Bool()) init (False) // MAX2x2
  val is_pool_reg = Reg(Bool()) init (False) // 是否 pooling（AVG2 / AVG3 / MAX2）

  // WBuf 中每个 kernel 的 tap 数，用于计算权重地址（wbuf_addr = base + offset）
  val tap_per_kernel_wbuf_reg = Reg(UInt(wbuf_addr_width bits)) init (1)

  // 别名（方便阅读）
  val is_conv3 = is_conv3_reg
  val is_pw1 = is_pw1_reg
  val is_dw3 = is_dw3_reg
  val is_avg2 = is_avg2_reg
  val is_avg3 = is_avg3_reg
  val is_max2 = is_max2_reg
  val is_pool = is_pool_reg
  val is_pool2 = is_avg2 || is_max2

  // credit：S0 相对 S1 领先的 token 数（0 ~ maxAhead）
  val ahead_cnt = Reg(UInt(creditWidth bits)) init (0)

  // ------------------------
  // 状态输出到上游 ctrl
  // ------------------------
  io.ctrl.busy := busy_reg
  io.ctrl.row_done := row_done_reg
  row_done_reg := False // 默认拉低，DRAIN 完成时打一拍高脉冲

  // ------------------------
  // 默认 SFU 配置输出（避免综合出 latch）
  // ------------------------
  io.sfu_coe.valid := False
  for (i <- 0 until lane_num) {
    io.sfu_coe.payload(i).bias := 0
    io.sfu_coe.payload(i).scale := 0
    io.sfu_coe.payload(i).shift := 0
  }

  // 任意一个 lane 忙则认为整个 MacDp 还没 drain 完
  val any_lane_busy = io.lane_ctrl.map(_.busy).orR

  // =====================================================================
  // op_mode → tap 数 解码函数
  // =====================================================================

  /**
   * decodeTapCount(op_mode)：tap 数解码
   * 000：CONV3x3 → 9 tap（3×3 卷积）
   * 001：PW1x1   → 1 tap（1×1 点卷积）
   * 010: DW3x3   → 3 tap（以列为粒度）
   * 011：AVG2x2  → 4 tap（2×2 平均池化）
   * 100：AVG3x3  → 9 tap（3×3 平均池化）
   */
  def decodeTapCount(op_mode: Bits): UInt = {
    val cnt = UInt(4 bits)
    cnt := 1
    switch(op_mode) {
      is(B"000") {
        cnt := 9
      } // CONV3x3
      is(B"001") {
        cnt := 1
      } // PW1x1
      is(B"010") {
        cnt := 3
      } // DW3x3（三列）
      is(B"011") {
        cnt := 4
      } // AVG2x2
      is(B"100") {
        cnt := 9
      } // AVG3x3
      is(B"101") {
        cnt := 4
      } // MAX2x2
    }
    cnt
  }

  // =====================================================================
  // Stage 0：地址生成 & padding 判定（S0）
  // =====================================================================
  val s0_stream = Stream(PipeCtrlBundle())

  // 基于 credit 的限流：
  // RUN 状态下，且 S0 尚未超过 maxAhead，才允许发射下一个 tap token。
  val s0_can_issue =
    (state_reg === MacDpState.RUN) && (ahead_cnt < maxAhead) && io.ctrl.enable
  s0_stream.valid := s0_can_issue

  // ------------------------
  // tap / group / pixel 边界判定
  // ------------------------
  val last_tap_in_group = tap_cnt === (tap_count_reg - 1)
  val last_group = g_cnt === (c4_in_reg - 1)
  val last_pixel = x_out_cnt === (row_len_reg - 1)

  // 每个像素的第一个 tap 与最后一个 tap
  val s0_is_first_tap = (g_cnt === 0) && (tap_cnt === 0)
  val s0_is_last_tap_pixel = last_tap_in_group && last_group

  // ------------------------
  // 1. kernel 高/宽索引 kh/kw （tap 内部坐标）
  // ------------------------
  val s0_kw = UInt(2 bits)
  val s0_kh = UInt(2 bits)
  s0_kw := 0
  s0_kh := 0

  when(is_conv3 || is_avg3) {
    s0_kh := (tap_cnt / 3).resize(2 bits)
    s0_kw := (tap_cnt % 3).resize(2 bits)
  } elsewhen (is_pool2) {
    s0_kh := (tap_cnt / 2).resize(2 bits)
    s0_kw := (tap_cnt % 2).resize(2 bits)
  } elsewhen (is_dw3) {
    s0_kw := tap_cnt.resize(2 bits) // 0,1,2 三列
    s0_kh := 0
  }

  // ------------------------
  // 2. X 方向 padding 判定
  // ------------------------
  val s0_x_base = UInt(coord_width bits)

  // AVG2：stride 固定为 2（x_out*2），其他算子 stride=1
  when(is_pool2) {
    s0_x_base := (x_out_cnt << 1).resize(coord_width bits)
  } otherwise {
    // CONV/DW/PW：根据 stride_reg 支持 stride=1/2
    when(stride_reg === U(2, stride_reg.getWidth bits)) {
      s0_x_base := (x_out_cnt << 1).resize(coord_width bits)
    } otherwise {
      s0_x_base := x_out_cnt.resize(coord_width bits)
    }
  }

  val s0_win_x = s0_x_base + s0_kw.resize(coord_width bits)
  val is_x_pad_left = s0_win_x < pad_left_reg.asUInt // 落在左侧 pad 区
  val s0_ibuf_x_raw = s0_win_x - pad_left_reg.asUInt // 去掉左 pad 偏移后坐标
  val is_x_pad_right = s0_ibuf_x_raw >= ibuf_valid_w_reg // 超出有效宽度 → 右侧 pad
  val is_x_pad = is_x_pad_left || is_x_pad_right

  val s0_ibuf_x = UInt(coord_width bits)
  s0_ibuf_x := is_x_pad ? U(0, coord_width bits) | s0_ibuf_x_raw

  // ------------------------
  // 3. Y 方向 padding 判定（CONV/AVG 使用 tap 行 kh）
  // ------------------------
  val is_y_pad_top = Bool()
  val is_y_pad_bot = Bool()
  is_y_pad_top := False
  is_y_pad_bot := False

  when(is_conv3 || is_avg3 || is_pool2) {
    val kh_for_pad = s0_kh
    // top 行：kh=0
    is_y_pad_top := pad_top_en_reg && (kh_for_pad === U(0, 2 bits))

    // bottom 行：Pool2 的 kh=1，AVG3 的 kh=2
    when(is_pool2) {
      is_y_pad_bot := pad_bot_en_reg && (kh_for_pad === U(1, 2 bits))
    } otherwise {
      is_y_pad_bot := pad_bot_en_reg && (kh_for_pad === U(2, 2 bits))
    }
  }

  val is_y_pad = is_y_pad_top || is_y_pad_bot

  // DW3x3：Y padding 不整 tap 清零，而是 top/bot 行逐行清零，数据流 {top, mid, bot, 0}
  val s0_pad_top_flag = Bool()
  val s0_pad_bot_flag = Bool()
  s0_pad_top_flag := False
  s0_pad_bot_flag := False
  when(is_dw3) {
    s0_pad_top_flag := pad_top_en_reg
    s0_pad_bot_flag := pad_bot_en_reg
  }

  // ------------------------
  // 4. IBUF 行选择（row_id）
  // ------------------------
  val s0_ibuf_row = UInt(row_id_width bits)
  s0_ibuf_row := row_mid_reg

  when(is_conv3 || is_avg3) {
    switch(s0_kh) {
      is(U(0, 2 bits)) {
        s0_ibuf_row := row_top_reg
      }
      is(U(1, 2 bits)) {
        s0_ibuf_row := row_mid_reg
      }
      default {
        s0_ibuf_row := row_bot_reg
      }
    }
  } elsewhen (is_pw1) {
    s0_ibuf_row := row_mid_reg
  } elsewhen (is_dw3) {
    s0_ibuf_row := row_mid_reg
  } elsewhen (is_pool2) {
    switch(s0_kh) {
      is(U(0, 2 bits)) {
        s0_ibuf_row := row_mid_reg
      }
      default {
        s0_ibuf_row := row_bot_reg
      }
    }
  }

  val s0_ibuf_dw = is_dw3 // IBUF DW 模式标志

  // ------------------------
  // 5. 通道 group 选择（C4_IN or DW group）
  // ------------------------
  val s0_cgroup = UInt(coord_width bits)
  when(is_dw3 || is_pool) {
    // DW 模式：kernel_group_idx 表示当前 Cout 子块的“channel group idx”
    // Pool 模式：同样使用 kernel_group_idx（每个 kernel group 对应一个 c4-group）
    s0_cgroup := kernel_group_idx_reg.resize(coord_width bits)
  } otherwise {
    // 普通 CONV/PW/AVG：按 g_cnt 遍历 C4_IN
    s0_cgroup := g_cnt
  }

  // ------------------------
  // 6. 整 tap padding 标志
  // ------------------------
  val s0_is_pad = Bool()
  when(is_dw3) {
    // DW：仅 X 方向 pad 触发整 tap 全零，Y padding 由 pad_top/ pad_bot 行级清零
    s0_is_pad := is_x_pad && !is_pw1
  } otherwise {
    // 非 DW：X 或 Y padding 任一成立则整 tap 全零，PW1x1 不受 padding
    s0_is_pad := (is_x_pad || is_y_pad) && !is_pw1
  }

  // ------------------------
  // 7. WBUF 权重地址计算
  // ------------------------
  val s0_tap_count_const = tap_per_kernel_wbuf_reg
  val s0_kernel_vol = c4_in_reg.resize(wbuf_addr_width bits) * s0_tap_count_const

  val s0_group_offset = kernel_group_idx_reg.resize(wbuf_addr_width bits) * s0_kernel_vol
  val s0_c4_offset = g_cnt.resize(wbuf_addr_width bits) * s0_tap_count_const
  val s0_relative_addr = s0_group_offset + s0_c4_offset + tap_cnt.resize(wbuf_addr_width bits)
  val s0_wbuf_addr = wgt_base_reg + s0_relative_addr

  // ------------------------
  // 8. S0 → S1 控制 payload
  // ------------------------
  s0_stream.payload.start_pixel := s0_is_first_tap
  s0_stream.payload.pixel_done := s0_is_last_tap_pixel
  s0_stream.payload.is_pool := is_pool
  s0_stream.payload.is_dw3 := is_dw3
  s0_stream.payload.lane_enable := io.ctrl.enable
  s0_stream.payload.is_pad := s0_is_pad
  s0_stream.payload.pad_top := s0_pad_top_flag
  s0_stream.payload.pad_bot := s0_pad_bot_flag

  val s0_fire = s0_stream.fire // S0 token 真正发射条件（valid && ready）

  // =====================================================================
  // CFG_SFU 阶段 WBUF 读址：与 RUN 阶段复用 WBUF
  // =====================================================================
  val in_cfg_mode = state_reg === MacDpState.CFG_SFU
  val cfg_group_base = (kernel_group_idx_reg << 1) // 每组量化参数占用 2 word
  val cfg_wbuf_addr = cfg_group_base + cfg_cnt.resize(wbuf_addr_width bits)
  val cfg_rd_en = in_cfg_mode && (cfg_cnt < 2) // 读取两个 word

  // =====================================================================
  // S0 → 读请求 FIFO（小宽度，降低资源）
  // =====================================================================
  val req_fifo = StreamFifo(RdReq(), depth = fifoDepth)
  private val SMALL_FIFO_MAX_DEPTH = 32
  if (fifoDepth <= SMALL_FIFO_MAX_DEPTH) {
    req_fifo.logic.ram.addAttribute("syn_ramstyle", "registers")
  }

  val req_push = Stream(RdReq())
  req_push.valid := s0_stream.valid
  req_push.payload.ctrl := s0_stream.payload
  req_push.payload.ibuf_x := s0_ibuf_x
  req_push.payload.ibuf_row := s0_ibuf_row
  req_push.payload.ibuf_dw := s0_ibuf_dw
  req_push.payload.ibuf_row_top := row_top_reg
  req_push.payload.ibuf_row_mid := row_mid_reg
  req_push.payload.ibuf_row_bot := row_bot_reg
  req_push.payload.cgroup := s0_cgroup
  req_push.payload.wbuf_addr := s0_wbuf_addr.resized
  req_fifo.io.push << req_push

  // FIFO 没满即可接受新的 S0 token
  s0_stream.ready := req_fifo.io.push.ready

  // ------------------------
  // S0 计数器更新：tap → group → pixel
  // ------------------------
  when(s0_fire) {
    when(last_tap_in_group) {
      tap_cnt := 0
      when(last_group) {
        g_cnt := 0
        when(last_pixel) {
          x_out_cnt := 0
          state_reg := MacDpState.DRAIN // 本行所有像素结束，进入 DRAIN 等待流水 flush
        } otherwise {
          x_out_cnt := x_out_cnt + 1
        }
      } otherwise {
        g_cnt := g_cnt + 1
      }
    } otherwise {
      tap_cnt := tap_cnt + 1
    }
  }

  // =====================================================================
  // 读请求发射 + 数据缓冲（1-deep）
  // =====================================================================
  val all_lanes_ready = io.lane_mac_out.map(_.ready).reduce(_ && _)

  val RETURN_FIFO_DEPTH = 4
  val ret_fifo = StreamFifo(FifoPayload(), depth = RETURN_FIFO_DEPTH)
  if (RETURN_FIFO_DEPTH <= SMALL_FIFO_MAX_DEPTH) {
    ret_fifo.logic.ram.addAttribute("syn_ramstyle", "registers")
  }

  val s1_stream = ret_fifo.io.pop
  s1_stream.ready := all_lanes_ready

  val s1_fire = s1_stream.fire

  // 读请求 pop（RUN 阶段），避免下拍数据进来时溢出
  val rd_req = req_fifo.io.pop
  val rd_fire = rd_req.fire
  val rd_fire_d1 = RegNext(rd_fire, init = False)
  val rd_ctrl_d1 = Reg(PipeCtrlBundle())

  // IBUF/WBUF 读返回为 1-cycle latency。发起本拍 rd_fire 后，下拍必然会向 ret_fifo push 1 个 beat。
  // 因此这里要求 ret_fifo 当前至少保留 1 个空槽，避免“本拍看起来 all_lanes_ready，
  // 但下拍 lane/SFU 因输出背压突然停住”时，旧版 1-deep buffer 发生静默覆盖。
  val ret_fifo_has_spare =
    ret_fifo.io.occupancy < U(RETURN_FIFO_DEPTH - 1, ret_fifo.io.occupancy.getWidth bits)
  rd_req.ready := (!in_cfg_mode) && ret_fifo_has_spare

  when(rd_fire) {
    rd_ctrl_d1 := rd_req.payload.ctrl
  }

  // ------------------------
  // 驱动 WBUF 读端口（CFG_SFU & RUN 复用）
  // ------------------------
  for (i <- 0 until lane_num) {
    io.wbuf_rd(i).rd_en := in_cfg_mode ? cfg_rd_en | rd_fire
    io.wbuf_rd(i).rd_addr := (in_cfg_mode ? cfg_wbuf_addr | rd_req.payload.wbuf_addr).resized
  }

  // ------------------------
  // 驱动 IBUF 读端口（只在 RUN 使用）
  // ------------------------
  io.ibuf_rd.rd_en := rd_fire
  io.ibuf_rd.rd_dw_mode := rd_req.payload.ibuf_dw
  io.ibuf_rd.rd_row_id := rd_req.payload.ibuf_row
  io.ibuf_rd.rd_row_top := rd_req.payload.ibuf_row_top
  io.ibuf_rd.rd_row_mid := rd_req.payload.ibuf_row_mid
  io.ibuf_rd.rd_row_bot := rd_req.payload.ibuf_row_bot
  io.ibuf_rd.rd_cgroup := rd_req.payload.cgroup
  io.ibuf_rd.rd_x := rd_req.payload.ibuf_x

  // ------------------------
  // 读出 IBUF/WBUF 数据并进入 1-deep 缓冲
  // ------------------------
  val data_in = FifoPayload()
  data_in.ctrl := rd_ctrl_d1
  data_in.ibuf_data_scalar := io.ibuf_rd.rd_data
  data_in.ibuf_data_top := io.ibuf_rd.rd_data_top
  data_in.ibuf_data_mid := io.ibuf_rd.rd_data_mid
  data_in.ibuf_data_bot := io.ibuf_rd.rd_data_bot
  for (i <- 0 until lane_num) {
    data_in.wbuf_data(i) := io.wbuf_rd(i).rd_data
  }

  ret_fifo.io.push.valid := rd_fire_d1
  ret_fifo.io.push.payload := data_in

  // 仿真保护：返回 FIFO 不允许溢出。
  if (GenerationFlags.simulation) {
    when(rd_fire_d1 && !ret_fifo.io.push.ready) {
      assert(False, "ClusterMacDp return FIFO overflow: read data arrives while FIFO is full")
    }
  }

  // ------------------------
  // credit 计数：S0 产生 / 读请求发射
  // ------------------------
  when(s0_fire && !rd_fire) {
    ahead_cnt := ahead_cnt + 1
  } elsewhen (!s0_fire && rd_fire) {
    when(ahead_cnt =/= 0) {
      ahead_cnt := ahead_cnt - 1
    }
  }

  if (GenerationFlags.simulation) {
    when(ahead_cnt > maxAhead) {
      assert(False, "ClusterMacDp credit overflow: ahead_cnt > maxAhead")
    }
  }

  // ------------------------
  // 根据 IBUF 读出的 3 行数据，逐 lane 拆 byte
  // ------------------------
  val s1_top_bytes = s1_stream.payload.ibuf_data_top.subdivideIn(cfg.dataWidth bits)
  val s1_mid_bytes = s1_stream.payload.ibuf_data_mid.subdivideIn(cfg.dataWidth bits)
  val s1_bot_bytes = s1_stream.payload.ibuf_data_bot.subdivideIn(cfg.dataWidth bits)

  val s1_is_pad = s1_stream.payload.ctrl.is_pad
  val s1_is_dw3 = s1_stream.payload.ctrl.is_dw3
  val s1_pad_top = s1_stream.payload.ctrl.pad_top
  val s1_pad_bot = s1_stream.payload.ctrl.pad_bot

  // ------------------------
  // 对每个 lane 生成 act_vec / wgt_vec + lane_ctrl
  // ------------------------
  for (i <- 0 until lane_num) {
    val lane_act = Bits(ibuf_word_width bits)

    when(s1_is_dw3) {
      // DW3x3：每个 lane 从 top/mid/bot 中取一个 byte，按 {top, mid, bot, 0} 拼成一向量
      val a_top_raw = s1_top_bytes(i)
      val a_mid_raw = s1_mid_bytes(i)
      val a_bot_raw = s1_bot_bytes(i)

      val a_top = s1_pad_top ? B(0, cfg.dataWidth bits) | a_top_raw
      val a_mid = a_mid_raw
      val a_bot = s1_pad_bot ? B(0, cfg.dataWidth bits) | a_bot_raw

      lane_act := B(0, cfg.dataWidth bits) ## a_bot ## a_mid ## a_top
    } otherwise {
      // 非 DW：直接使用 scalar 向量（4×int8）作为 act_vec
      lane_act := s1_stream.payload.ibuf_data_scalar
    }

    // 整 tap padding：act_vec / wgt_vec 全零，但仍然发 pixel 控制信号
    val pad_act = is_max2_reg ? B(0x80808080L, ibuf_word_width bits) | B(0, ibuf_word_width bits)
    val final_act = s1_is_pad ? pad_act | lane_act
    val final_wgt = s1_is_pad ? B(0, wbuf_word_width bits) | s1_stream.payload.wbuf_data(i)

    // Lane MAC 输入：所有 lane 共用同一个 valid（对齐 ctrl）
    io.lane_mac_out(i).valid := s1_stream.valid
    io.lane_mac_out(i).payload.act_vec := final_act
    io.lane_mac_out(i).payload.wgt_vec := final_wgt
    // Pool 模式：每个 lane 只对一个 byte 累加（避免 4 个 byte 混加）。
    // 非 Pool：elem_mask 全 1，表示 SIMD4 全启用。
    io.lane_mac_out(i).payload.elem_mask :=
      s1_stream.payload.ctrl.is_pool ? (B(1, cfg.simdNum bits) |<< i) | B((1 << cfg.simdNum) - 1, cfg.simdNum bits)

    // Lane 控制信号：start_pixel / pixel_done 不再被 padding 屏蔽。
    //
    // 重要：当 `s1_stream.valid=0` 时，禁止从 FIFO payload 推导 lane enable。
    // 原因：valid=0 时 payload 属于 don't-care，若直接使用会导致 `lane_ctrl.enable` 变成 X，
    // X 会进一步传播到 PELane.ready、PEGroup mask/valid，最终让 SFU/OBuf/OutDMA 写出 X 数据。
    //
    // 策略：只要 MacDp 处于 enable（ClusterCtrl 会把 `ctrl.enable` 维持到 SFU_DRAIN），
    // 或 MacDp 内部仍 busy，就保持 lane enable 为真。
    // 这可以避免“极短行”场景下过早关 lane：lane 的最终结果可能在最后一个输入 beat 之后数拍才出现。
    io.lane_ctrl(i).enable := io.ctrl.enable || busy_reg
    io.lane_ctrl(i).accum_enable := True
    io.lane_ctrl(i).start_pixel := s1_stream.payload.ctrl.start_pixel
    io.lane_ctrl(i).pixel_done := s1_stream.payload.ctrl.pixel_done
    io.lane_ctrl(i).mode_avgpool := is_avg2_reg || is_avg3_reg
    io.lane_ctrl(i).mode_maxpool := is_max2_reg
  }

  // =====================================================================
  // 主状态机：IDLE → CFG_SFU → RUN → DRAIN
  // =====================================================================
  switch(state_reg) {
    is(MacDpState.IDLE) {
      // 进入 IDLE 时清空计数与 credit
      x_out_cnt := 0
      g_cnt := 0
      tap_cnt := 0
      cfg_cnt := 0
      ahead_cnt := 0

      when(io.ctrl.enable && io.ctrl.start && !busy_reg) {
        busy_reg := True
        row_len_reg := io.ctrl.row_len
        stride_reg := io.ctrl.stride
        c4_in_reg := io.ctrl.c4_in
        row_top_reg := io.ctrl.row_top
        row_mid_reg := io.ctrl.row_mid
        row_bot_reg := io.ctrl.row_bot
        kernel_group_idx_reg := io.ctrl.kernel_group_idx

        // 权重基地址
        wgt_base_reg := io.ctrl.wbuf_wgt_base

        // Padding 信息
        pad_left_reg := io.ctrl.pad_left
        pad_right_reg := io.ctrl.pad_right
        pad_top_en_reg := io.ctrl.pad_top_en
        pad_bot_en_reg := io.ctrl.pad_bot_en
        ibuf_valid_w_reg := io.ctrl.ibuf_valid_w

        // op_mode 解码锁存
        is_conv3_reg := io.ctrl.op_mode === B"000"
        is_pw1_reg := io.ctrl.op_mode === B"001"
        is_dw3_reg := io.ctrl.op_mode === B"010"
        is_avg2_reg := io.ctrl.op_mode === B"011"
        is_avg3_reg := io.ctrl.op_mode === B"100"
        is_max2_reg := io.ctrl.op_mode === B"101"
        is_pool_reg :=
          (io.ctrl.op_mode === B"011") || (io.ctrl.op_mode === B"100") || (io.ctrl.op_mode === B"101")

        // tap 数量（CONV3/AVG3=9, PW1=1, DW3=3, AVG2=4）
        val tap_cnt_tmp = decodeTapCount(io.ctrl.op_mode)
        tap_count_reg := tap_cnt_tmp
        tap_per_kernel_wbuf_reg := tap_cnt_tmp.resize(wbuf_addr_width bits)

        // 下一步进入 CFG_SFU（先配置 SFU，再进入 RUN）
        state_reg := MacDpState.CFG_SFU
      }
    }

    is(MacDpState.CFG_SFU) {
      when(is_pool_reg) {
        // -------------------------------------------------------------
        // 池化：AVG2x2 / AVG3x3
        //  - 不依赖外部量化参数；
        //  - AVG 的 /4(/8) 在 PELane 内部做算术右移（trunc，匹配软件行为模型）；
        //  - SFU 只负责 activation/saturate（以及 pool 的固定 shift），这里配置为：
        //      其中 bias=0, scale=1, shift=2(AVG2) / 3(AVG3)
        //    并在 SFU 侧开启 trunc-shift bypass，保证语义是 trunc 而不是 rounding。
        // -------------------------------------------------------------
        io.sfu_coe.valid := True
        for (i <- 0 until lane_num) {
          io.sfu_coe.payload(i).bias := S(0, io.sfu_coe.payload(i).bias.getWidth bits)
          io.sfu_coe.payload(i).scale := U(1, io.sfu_coe.payload(i).scale.getWidth bits)
          val pool_shift = UInt(io.sfu_coe.payload(i).shift.getWidth bits)
          pool_shift := U(0)
          when(is_avg2_reg) {
            pool_shift := U(2)
          } elsewhen (is_avg3_reg) {
            pool_shift := U(3)
          }
          io.sfu_coe.payload(i).shift := pool_shift
        }

        cfg_cnt := 0
        state_reg := MacDpState.RUN
      } otherwise {
        // -------------------------------------------------------------
        // CONV / PW / DW：从 WBUF 读取量化参数（2×32bit word）
        //  - cfg_cnt = 0：发出第 1 个读请求；
        //  - cfg_cnt = 1：latch 低 word（bias），发出第 2 个读请求；
        //  - cfg_cnt = 2：读取高 word（scale+shift），并下发 SFU 配置。
        // -------------------------------------------------------------
        switch(cfg_cnt) {
          is(U(0, 2 bits)) {
            cfg_cnt := 1
          }

          is(U(1, 2 bits)) {
            // 上一拍 WBUF 的 rd_data 即为低 word（bias）
            for (i <- 0 until lane_num) {
              bias_low_buf(i) := io.wbuf_rd(i).rd_data
            }
            cfg_cnt := 2
          }

          is(U(2, 2 bits)) {
            io.sfu_coe.valid := True
            for (i <- 0 until lane_num) {
              val high_word = io.wbuf_rd(i).rd_data
              val low_word = bias_low_buf(i)

              io.sfu_coe.payload(i).bias := low_word.asSInt.resized
              io.sfu_coe.payload(i).scale := high_word(15 downto 0).asUInt.resized
              io.sfu_coe.payload(i).shift := high_word(21 downto 16).asUInt
            }
            cfg_cnt := 0
            state_reg := MacDpState.RUN
          }
        }
      }
    }

    is(MacDpState.RUN) {
      // RUN 状态下，实际的数据流由 S0/S1 + FIFO + credit 控制；
      // 但若输入配置非法（例如 row_len/c4_in 为 0），应尽快收敛到 DRAIN，避免死锁。
      when((row_len_reg === 0) || (c4_in_reg === 0)) {
        state_reg := MacDpState.DRAIN
      }
    }

    is(MacDpState.DRAIN) {
      // 只有当 request FIFO、读返回流水、s1 缓冲以及 lane 尾部流水都排空后，才能结束当前行。
      // - `no_inflight_reads` 解决“最后一笔延迟读请求尚未入 FIFO”导致的假空；
      // - `!any_lane_busy` 保证最后一个 pixel_done 产生的 acc_out 已真正被 SFU 接走。
      // 两者缺一都会让 row_done 过早，从而污染每行尾部 word。
      val no_inflight_reads = !req_fifo.io.pop.valid && !rd_fire && !rd_fire_d1
      when(!s1_stream.valid && no_inflight_reads && !any_lane_busy) {
        busy_reg := False
        row_done_reg := True
        state_reg := MacDpState.IDLE
      }
    }
  }
}

// =======================================================================
// 单独生成 Verilog 的便捷对象
// =======================================================================
object ClusterMacDp extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = false,
    headerWithDate = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new ClusterMacDp(VenusCoreConfig.default.clusterCfg)).printPruned()
}
