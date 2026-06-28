package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * ClusterSFU（优化的流水线版本）
 * -------------------------------------
 * 优化点：
 * 1. 将 s2mPipe 替换为 m2sPipe (stage)，强制在 Payload 路径插入寄存器，切断组合逻辑链。
 * 2. 在 Serializer 和 Adder 之间增加一级打拍，隔离 mac_dp RAM 的读取延迟。
 * 3. 保持完整的反压 (Back-pressure) 支持。
 */
class ClusterSFU(cfg: ClusterConfig) extends Component {

  assert(cfg.laneNum == 4, "ClusterSFU designed for SIMD=4")

  val sumWidth = cfg.accWidth.max(cfg.biasWidth) + 1
  val mulResWidth = sumWidth + cfg.scaleWidth

  val io = new Bundle {
    val ctrl = slave(SfuCtrlPort(cfg))
    val coe = slave(Flow(Vec(SfuCoe(cfg), cfg.laneNum)))
    val acc_in = slave Stream (SfuAccIn(cfg))
    val out_data = master Stream (Bits(cfg.obufWordWidth bits))
  }
  noIoPrefix()

  // pool 快路径开关：
  // - 默认 avgpool2x2 的语义保持 sum >> 2（trunc）；
  // - 对选定层，编译器复用 QMODE=Q4 作为 opt-in 标记，切到 signed round(sum / 4), ties-to-even；
  // - 如果仍走“串行化 + 量化”通道，会导致每个 32-bit 输出 word 需要 4 拍（按 lane 串行），吞吐仅 1/4；
  // - 因此在 trunc_shift=1 时启用 pool 快路径：4 lane 并行做平均 + 饱和，再打包输出。
  val use_pool_fast = io.ctrl.trunc_shift

  // 输出统一走 out_stream，再接到 io.out_data（避免多处驱动）
  val out_stream = Stream(Bits(cfg.obufWordWidth bits))
  io.out_data << out_stream

  // 参数寄存器
  val params = Reg(Vec(SfuCoe(cfg), cfg.laneNum))
  when(io.coe.valid) {
    params := io.coe.payload
  }

  // valid 在文件末尾统一给出（避免多处赋值）
  // 说明：
  // - ClusterCtrl 用该信号做“输出 word 计数”，用于判断一行 SFU 是否完成；
  // - 因为 OBuf 可能 backpressure，必须用 fire 而不是 valid，否则会提前计数导致行切换错误。
  io.ctrl.valid := out_stream.fire

  // ===========================================================================
  // 1. 串行化器 (Serializer)
  // ===========================================================================

  case class SerialPayload() extends Bundle {
    val acc = SInt(cfg.accWidth bits)
    val param = SfuCoe(cfg)
    val mask = Bool()
    val last = Bool()
  }

  // 现有实现把 `io.acc_in` 当作“保持 4 拍稳定的向量”，只在最后一拍给 ready。
  // 这会导致上游（PEGroup/PELane）在 3/4 的时间被 SFU 反压，avgpool 这类轻算子吞吐会很差。
  //
  // 改进：增加一个“向量缓冲寄存器”，一次握手接收完整 4-lane 向量，再在 SFU 内部串行化。
  // 这样：
  // - 上游只需保持 payload 1 拍；
  // - SFU 可以在后台用 4 拍完成串行化/量化/饱和；
  // - MAC-DP 可以与 SFU 流水并行（avgpool 利用率明显提升）。
  val vec_buf_valid = Reg(Bool()) init (False)
  val vec_buf_lanes = Reg(Vec(SInt(cfg.accWidth bits), cfg.laneNum))
  val vec_buf_mask = Reg(Bits(cfg.laneNum bits)) init (0)

  val serial_stream = Stream(SerialPayload())
  val lane_cnt = Reg(UInt(2 bits)) init (0)
  val is_last_lane = lane_cnt === 3

  // 当向量缓冲为空时，允许接收新的 acc_in 向量
  io.acc_in.ready := use_pool_fast ? out_stream.ready | !vec_buf_valid

  when(io.acc_in.fire && !use_pool_fast) {
    vec_buf_valid := True
    vec_buf_lanes := io.acc_in.lanes
    vec_buf_mask := io.acc_in.mask
    lane_cnt := 0
  }

  serial_stream.valid := vec_buf_valid && !use_pool_fast
  serial_stream.payload.acc := vec_buf_lanes(lane_cnt)
  serial_stream.payload.mask := vec_buf_mask(lane_cnt)
  serial_stream.payload.param := params(lane_cnt)
  serial_stream.payload.last := is_last_lane

  when(serial_stream.fire) {
    when(is_last_lane) {
      // 该向量 4 lane 已全部串行化，释放缓冲
      vec_buf_valid := False
      lane_cnt := 0
    } otherwise {
      lane_cnt := lane_cnt + 1
    }
  }

  // ===========================================================================
  // 2. 计算流水线 (Calculation Pipeline)
  // ===========================================================================

  // --- 阶段 0：缓冲 / 隔离 ---
  // 关键优化：在计算开始前先打一拍。
  // 目的：切断从 mac_dp (RAM) -> Mux -> Adder 的路径。
  // 使用 m2sPipe 切断 Payload 路径。
  val s0_stream = serial_stream.m2sPipe()

  // --- 阶段 1：加 bias ---
  case class S1Payload() extends Bundle {
    val sum = SInt(sumWidth bits)
    val param = SfuCoe(cfg)
    val mask = Bool()
    val last = Bool()
  }

  val s1_stream = s0_stream.map { in =>
    val out = S1Payload()
    val op_acc = in.mask ? in.acc | S(0)
    out.sum := (op_acc + in.param.bias).resized
    out.param := in.param
    out.mask := in.mask
    out.last := in.last
    out
  }.stage() // 使用 stage() 等同于 m2sPipe()，确保 Adder 结果存入寄存器

  // --- 阶段 2：乘 scale ---
  case class S2Payload() extends Bundle {
    val prod = SInt(mulResWidth bits)
    val shift = UInt(cfg.shiftWidth bits)
    val mask = Bool()
    val last = Bool()
  }

  val s2_stream = s1_stream.map { in =>
    val out = S2Payload()
    out.prod := (in.sum * (U(0, 1 bits) ## in.param.scale).asSInt).resized
    out.shift := in.param.shift
    out.mask := in.mask
    out.last := in.last
    out
  }.stage() // 切断 Mul 路径

  // --- 阶段 3：移位 + 激活 ---
  // 说明：
  // - 默认行为：rounding right shift（匹配量化的“带舍入”需求）。
  // - trunc_shift=1：让 SFU 对 shift 采用“纯算术右移”（trunc），跳过 rounding 的 bias 加法。
  //   当前 avgpool 使用并行 fast path，因此这里仍保留 trunc_shift 语义，避免影响非 pool 路径。
  //
  // 时序优化：将 “Shift + Activation” 拆成两级流水，避免变量移位 + 比较链路落在同一拍。
  // 进一步时序优化：rounding 所需的 bias 生成（可变左移）与最终右移拆开两拍，避免同拍出现两个 barrel shifter。
  case class S3prePayload() extends Bundle {
    val prodPre = SInt(mulResWidth bits)
    val shift = UInt(cfg.shiftWidth bits)
    val mask = Bool()
    val last = Bool()
  }

  val s3pre_stream = s2_stream.map { in =>
    val out = S3prePayload()

    val prodPre = SInt(mulResWidth bits)
    when(io.ctrl.trunc_shift) {
      // trunc：不做 rounding bias，加法链路最短
      prodPre := in.prod
    } otherwise {
      when(in.shift === 0) {
        prodPre := in.prod
      } otherwise {
        val bias = (U(1, mulResWidth bits) << (in.shift - 1)).asSInt

        val addend = SInt(mulResWidth bits)
        when(in.prod.msb) { // negative
          addend := (bias - 1).resized
        } otherwise { // non-negative
          addend := bias.resized
        }
        prodPre := (in.prod + addend).resized
      }
    }

    out.prodPre := prodPre
    out.shift := in.shift
    out.mask := in.mask
    out.last := in.last
    out
  }.stage()

  case class S3shiftPayload() extends Bundle {
    val shifted = SInt(mulResWidth bits)
    val mask = Bool()
    val last = Bool()
  }

  val s3shift_stream = s3pre_stream.map { in =>
    val out = S3shiftPayload()

    val shifted = SInt(mulResWidth bits)
    shifted := (in.shift === 0) ? in.prodPre | (in.prodPre >> in.shift)

    out.shifted := shifted
    out.mask := in.mask
    out.last := in.last
    out
  }.stage()

  case class S3aPayload() extends Bundle {
    val activated = SInt(mulResWidth bits)
    val mask = Bool()
    val last = Bool()
  }

  val s3a_stream = s3shift_stream.map { in =>
    val out = S3aPayload()

    val activated = SInt(mulResWidth bits)
    switch(io.ctrl.act_type) {
      is(ClusterActType.RELU) {
        activated := in.shifted.max(0)
      }
      is(ClusterActType.RELU6) {
        activated := in.shifted.max(0)
      }
      default {
        activated := in.shifted
      }
    }

    out.activated := activated
    out.mask := in.mask
    out.last := in.last
    out
  }.stage()

  // --- 阶段 4：饱和裁剪 ---
  case class S3Payload() extends Bundle {
    val resByte = Bits(8 bits)
    val last = Bool()
  }

  val s3_stream = s3a_stream.map { in =>
    val out = S3Payload()

    val sat_res = Bits(8 bits)
    val max_int8 = S(127, mulResWidth bits)
    val min_int8 = S(-128, mulResWidth bits)

    when(in.activated > max_int8) {
      sat_res := S(127, 8 bits).asBits
    } elsewhen (in.activated < min_int8) {
      sat_res := S(-128, 8 bits).asBits
    } otherwise {
      sat_res := in.activated(7 downto 0).asBits
    }

    out.resByte := in.mask ? sat_res | B(0, 8 bits)
    out.last := in.last
    out
  }.stage()

  // ===========================================================================
  // 4. 解串器 (Deserializer)
  // ===========================================================================

  val collect_buf = Vec(Reg(Bits(8 bits)), cfg.laneNum)
  val collect_cnt = Reg(UInt(2 bits)) init 0

  out_stream.valid := False
  out_stream.payload := B(0)
  s3_stream.ready := True // Default ready

  when(s3_stream.valid) {
    collect_buf(collect_cnt) := s3_stream.payload.resByte

    when(s3_stream.payload.last) {
      // 这里的组合逻辑只剩下拼接，非常快
      val full_word = s3_stream.payload.resByte ## collect_buf(2) ## collect_buf(1) ## collect_buf(0)

      out_stream.valid := True
      out_stream.payload := full_word

      // 只有下游 ready，我们才能消耗 s3 的数据
      s3_stream.ready := out_stream.ready

      when(out_stream.ready) {
        collect_cnt := 0
      }
    } otherwise {
      collect_cnt := collect_cnt + 1
    }
  }

  // ===========================================================================
  // 5. Busy 指示（给 ClusterCtrl 做 enable/busy 保持）
  // ===========================================================================
  // 注意：busy 不应只看 acc_in.valid；否则 SFU 内部仍有待处理数据时可能被错误关断。
  io.ctrl.busy :=
    vec_buf_valid || serial_stream.valid ||
      s0_stream.valid || s1_stream.valid || s2_stream.valid ||
      s3shift_stream.valid || s3a_stream.valid || s3_stream.valid || io.out_data.valid

  // ===========================================================================
  // 6. Pool 快路径（并行处理 4 lane）
  // ===========================================================================
  // 约定：
  // - 默认 avgpool2x2 语义为 trunc：sum >> 2，然后饱和到 int8；
  // - io.ctrl.pool_ties_even=1 时，切到 signed round(sum / 4), ties-to-even。
  // 这里不走 bias/scale，直接在 fast path 完成“平均 + 饱和”。
  when(use_pool_fast) {
    out_stream.valid := io.acc_in.valid
    // ready 已在上面通过 io.acc_in.ready 绑定到 out_stream.ready

    val packed = Bits(cfg.obufWordWidth bits)
    val bytes = Vec(Bits(8 bits), cfg.laneNum)

    for (i <- 0 until cfg.laneNum) {
      val v = io.acc_in.lanes(i)
      val vWide = v.resize(cfg.accWidth + 2)
      val truncShifted = (vWide >> 2).resized
      val neg = vWide.msb
      val mag = UInt((cfg.accWidth + 2) bits)
      when(neg) {
        mag := (-vWide).asUInt
      } otherwise {
        mag := vWide.asUInt
      }
      val quo = mag >> 2
      val rem = mag(1 downto 0)
      val inc = Bool()
      inc := (rem > U(2, 2 bits)) || ((rem === U(2, 2 bits)) && quo(0))
      val roundedMag = UInt((cfg.accWidth + 2) bits)
      roundedMag := (quo + inc.asUInt).resized
      val shifted = SInt((cfg.accWidth + 2) bits)
      when(neg) {
        shifted := (-roundedMag.asSInt).resized
      } otherwise {
        shifted := roundedMag.asSInt.resized
      }
      val poolShifted = io.ctrl.pool_ties_even ? shifted | truncShifted

      val max_int8 = S(127, poolShifted.getWidth bits)
      val min_int8 = S(-128, poolShifted.getWidth bits)
      val sat = Bits(8 bits)

      when(poolShifted > max_int8) {
        sat := S(127, 8 bits).asBits
      } elsewhen (poolShifted < min_int8) {
        sat := S(-128, 8 bits).asBits
      } otherwise {
        sat := poolShifted(7 downto 0).asBits
      }

      // mask=0 的 lane 输出置 0（用于通道 padding/关闭 lane）
      bytes(i) := io.acc_in.mask(i) ? sat | B(0, 8 bits)
    }

    // 低字节对应 lane0，高字节对应 lane3，保持与原解串器一致
    packed := bytes(3) ## bytes(2) ## bytes(1) ## bytes(0)
    out_stream.payload := packed
  }
}

// ==============================
// Verilog 生成入口
// ==============================
object ClusterSFU extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new ClusterSFU(VenusCoreConfig.default.clusterCfg)).printPruned()
}
