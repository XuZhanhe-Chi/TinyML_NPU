package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._

/**
 * PELane: 单路 SIMD4 MAC 累加单元 (Stallable Version)
 * --------------------------------
 * 方案：全局暂停 (Global Stall)
 * 限制：不增加 FIFO 深度。
 * 逻辑：当下游不 Ready 且有输出时，冻结流水线所有寄存器，
 * 输入端立即停止接收 (Ready=0)。
 */
case class PELane(cfg: ClusterConfig) extends Component {

  assert(cfg.dataWidth == 8, "PELane currently supports 8-bit data width only")
  assert(cfg.simdNum == 4, "PELane currently designed for SIMD=4")

  val io = new Bundle {
    val ctrl = slave(LaneCtrlPort(cfg))
    val mac_in = slave(Stream(LaneMacIn(cfg)))
    val acc_out = master(Stream(SInt(cfg.accWidth bits)))
  }
  noIoPrefix()

  // ============================================================
  // 1. 全局暂停控制逻辑 (Global Stall Logic)
  // ============================================================

  // 输出端口寄存器
  val out_valid_reg = Reg(Bool()) init (False)
  val out_data_reg = Reg(SInt(cfg.accWidth bits)) init (0)

  // ★★★ 核心：判断是否堵塞 ★★★
  // 如果当前已经输出了 Valid 数据，但下游 Ready 没拉高，则必须暂停！
  val is_stalled = out_valid_reg && !io.acc_out.ready

  // 只有在【使能】且【未暂停】时，才允许输入握手
  io.mac_in.ready := io.ctrl.enable && !is_stalled

  val in_fire = io.mac_in.valid && io.mac_in.ready

  // 连线输出
  io.acc_out.valid := out_valid_reg
  io.acc_out.payload := out_data_reg

  // Busy 逻辑：只要有任何一级 valid，或者处于暂停状态，都算 busy
  // 注意：暂停期间 busy 必须为高，防止 ClusterMacDp 误判结束
  val s0_valid = Reg(Bool()) init (False)
  val s1_valid = Reg(Bool()) init (False)
  val s2_valid = Reg(Bool()) init (False)

  io.ctrl.busy := in_fire || s0_valid || s1_valid || s2_valid || out_valid_reg

  // ============================================================
  // Stage 0: 输入寄存 (带暂停保护)
  // ============================================================

  case class Stage0Payload() extends Bundle {
    val act_vec = Bits(cfg.simdNum * cfg.dataWidth bits)
    val wgt_vec = Bits(cfg.simdNum * cfg.dataWidth bits)
    val elem_mask = Bits(cfg.simdNum bits)
    val start_pixel = Bool()
    val accum_enable = Bool()
    val pixel_done = Bool()
    val mode_avgpool = Bool()
    val mode_maxpool = Bool()
  }

  val s0_payload = Reg(Stage0Payload())

  // ★★★ 所有的寄存器更新都必须在 !is_stalled 时进行 ★★★
  when(!is_stalled) {
    s0_valid := in_fire // 如果没暂停，Valid 随输入 fire 传递

    when(in_fire) {
      s0_payload.act_vec := io.mac_in.payload.act_vec
      s0_payload.wgt_vec := io.mac_in.payload.wgt_vec
      s0_payload.elem_mask := io.mac_in.payload.elem_mask
      s0_payload.start_pixel := io.ctrl.start_pixel
      s0_payload.accum_enable := io.ctrl.accum_enable
      s0_payload.pixel_done := io.ctrl.pixel_done
      s0_payload.mode_avgpool := io.ctrl.mode_avgpool
      s0_payload.mode_maxpool := io.ctrl.mode_maxpool
    }
  }

  // ============================================================
  // Stage 1: 解包 + 乘法 (带暂停保护)
  // ============================================================

  case class Stage1Payload() extends Bundle {
    val products = Vec(SInt(2 * cfg.dataWidth bits), cfg.simdNum)
    val start_pixel = Bool()
    val accum_enable = Bool()
    val pixel_done = Bool()
    val mode_avgpool = Bool()
    val mode_maxpool = Bool()
  }

  val s1_payload = Reg(Stage1Payload())

  val acts_byte = s0_payload.act_vec.subdivideIn(cfg.dataWidth bits)
  val wgts_byte = s0_payload.wgt_vec.subdivideIn(cfg.dataWidth bits)
  val mul_results = Vec(SInt(2 * cfg.dataWidth bits), cfg.simdNum)

  // 组合逻辑乘法
  for (i <- 0 until cfg.simdNum) {
    val act_op = acts_byte(i).asSInt
    val raw_mul = SInt(2 * cfg.dataWidth bits)

    when(s0_payload.mode_avgpool || s0_payload.mode_maxpool) {
      // AVGPOOL：权重恒为 1，直接做符号扩展即可，避免综合出乘法器/DSP
      raw_mul := act_op.resize(2 * cfg.dataWidth)
    } otherwise {
      raw_mul := (act_op * wgts_byte(i).asSInt).resized
    }
    when(s0_payload.elem_mask(i)) {
      mul_results(i) := raw_mul
    } otherwise {
      mul_results(i) := 0
    }
  }

  // 寄存器更新
  when(!is_stalled) {
    s1_valid := s0_valid

    // 只有上前一级是 valid 的时候才更新数据，省功耗（可选）
    when(s0_valid) {
      s1_payload.products := mul_results
      s1_payload.start_pixel := s0_payload.start_pixel
      s1_payload.accum_enable := s0_payload.accum_enable
      s1_payload.pixel_done := s0_payload.pixel_done
      s1_payload.mode_avgpool := s0_payload.mode_avgpool
      s1_payload.mode_maxpool := s0_payload.mode_maxpool
    }
  }

  // ============================================================
  // Stage 2: 加法树 (带暂停保护)
  // ============================================================

  case class Stage2Payload() extends Bundle {
    val sum_of_products = SInt(cfg.accWidth bits)
    val start_pixel = Bool()
    val accum_enable = Bool()
    val pixel_done = Bool()
    val mode_avgpool = Bool()
    val mode_maxpool = Bool()
  }

  val s2_payload = Reg(Stage2Payload())
  val sum_comb = s1_payload.products.reduceBalancedTree(_ + _)

  when(!is_stalled) {
    s2_valid := s1_valid

    when(s1_valid) {
      s2_payload.sum_of_products := sum_comb.resize(cfg.accWidth)
      s2_payload.start_pixel := s1_payload.start_pixel
      s2_payload.accum_enable := s1_payload.accum_enable
      s2_payload.pixel_done := s1_payload.pixel_done
      s2_payload.mode_avgpool := s1_payload.mode_avgpool
      s2_payload.mode_maxpool := s1_payload.mode_maxpool
    }
  }

  // ============================================================
  // Stage 3: 累加 & 输出 (关键: 累加器也要冻结)
  // ============================================================

  val acc_reg = Reg(SInt(cfg.accWidth bits)) init (0)
  val acc_next = SInt(cfg.accWidth bits)

  // 组合逻辑计算下一拍累加值
  // 注意：这里使用的是 acc_reg(当前值) 和 s2_payload(当前流水线值)
  when(s2_payload.mode_maxpool) {
    when(s2_payload.start_pixel) {
      acc_next := s2_payload.sum_of_products
    } elsewhen (s2_payload.accum_enable) {
      acc_next := (acc_reg > s2_payload.sum_of_products) ? acc_reg | s2_payload.sum_of_products
    } otherwise {
      acc_next := acc_reg
    }
  } otherwise {
    when(s2_payload.start_pixel) {
      when(s2_payload.accum_enable) {
        acc_next := s2_payload.sum_of_products
      } otherwise {
        acc_next := 0
      }
    } elsewhen (s2_payload.accum_enable) {
      acc_next := acc_reg + s2_payload.sum_of_products
    } otherwise {
      acc_next := acc_reg
    }
  }

  // 只有在未暂停且 S2 有效时，才更新累加器
  when(!is_stalled) {
    when(s2_valid) {
      acc_reg := acc_next
    }

    // 输出 Valid/Data 更新逻辑
    // 如果 S2 完成了一个像素，且当前未暂停，则输出数据
    when(s2_valid && s2_payload.pixel_done) {
      out_valid_reg := True
      // AVGPOOL 的 sum>>2（trunc）由 SFU 侧完成（trunc-shift bypass），Lane 仅输出累加和。
      out_data_reg := acc_next
    } elsewhen (io.acc_out.ready) {
      // 如果没有新结果产生，但下游握手了，则清除 valid
      out_valid_reg := False
    }
  }
}

// ==============================
// Verilog 生成入口
// ==============================
object PELane extends App {
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new PELane(VenusCoreConfig.default.clusterCfg))
}
