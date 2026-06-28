package venuscore.cluster

import spinal.core._
import venuscore.config._

/**
 * ClusterUopGeom（uOP 几何信息）
 * -------------
 * uOP 解码阶段可推导出的“通用几何信息”，用于减少 ClusterCtrl 内部的隐式推导与组合逻辑。
 *
 * 说明：
 * - ifm_h/ifm_w：该 uOP 对应的输入特征图全局平面尺寸（不包含 padding，单位：像素）
 * - 这些值由 CtrlUopFetch 在取回 32B uOP 后填充（当前版本 IFM_W/IFM_H 直接来自 uOP 的 W6），
 *   Cluster 侧直接使用即可。
 */
case class ClusterUopGeom(cfg: ClusterConfig) extends Bundle {
  val ifm_h = UInt(12 bits)
  val ifm_w = UInt(12 bits)
}

/**
 * ClusterUopPrecalc（uOP 预计算字段）
 * -----------------
 * 在 uOP 解码阶段（CtrlUopFetch）即可确定的“预计算常量”，用于：
 * - 避免每个 ClusterCtrl 都重复推导同一批几何/地址参数（多 cluster 例化时节省资源）；
 * - 尽量把大组合/乘法链路从 ClusterCtrl 的关键路径上移到 CtrlUopFetch，并寄存后下发。
 *
 * 说明：
 * - 这些字段不改变 ISA 语义，它们只是把 ClusterCtrl 原本要算的东西提前算好并打包。
 * - `*_bytes` 一律是 byte 粒度；`*_words` 一律是 32-bit word 粒度。
 */
case class ClusterUopPrecalc(cfg: ClusterConfig) extends Bundle {
  private val wbuf_ctrl_width = cfg.wbufAddrWidth + log2Up(cfg.laneNum)
  private val addr_width = cfg.dmaConfig.addrWidth

  // IBUF/ActDMA：输入一行需要的像素数/字节数（像素=word，按 NCHWc4：1 像素 = 1×32bit word）
  val in_row_pixels = UInt(12 bits)
  val in_row_bytes = UInt(addr_width bits)
  val ibuf_line_words = UInt(cfg.ibufAddrWidth bits) // in_row_pixels * c4_in

  // FI_STRIDE：每通道平面 word 数 → byte
  val fi_stride_bytes = UInt(addr_width bits)

  // SFU/OutDMA：一行输出 word 数、行 byte stride、整平面 byte stride
  val row_out_words = UInt(16 bits) // w_tile * c4_out（word）
  val row_stride_bytes = UInt(addr_width bits) // w_tile * 4
  val fo_plane_stride_bytes = UInt(addr_width bits) // fo_stride * 4

  // WgtDMA：本 tile 的 kernel 总数（= c4_out * laneNum）
  val total_kernel_cnt = UInt(wbuf_ctrl_width bits)

  // WgtDMA：加载权重阶段的 DMA length（word 数）
  // 用于避免在 ClusterCtrl 内综合乘法器计算 (wgt_len_words * total_kernel_cnt)。
  val wgt_dma_len_words = UInt(cfg.dmaConfig.maxWordWidth bits)

  // 预先把 Y_INDEX 的对齐做掉，ClusterCtrl 只需要叠加 row_offset / group_offset
  val fi_addr_tile = UInt(addr_width bits)
  val fo_addr_tile = UInt(addr_width bits)

  // IFM 最后一行的基地址（absolute byte addr）：
  // 用于在 pad_bottom / 越界场景下做地址饱和，避免在 ClusterCtrl 内综合乘法器计算 (ifm_h-1)*in_row_bytes。
  val fi_last_row_addr = UInt(addr_width bits)
}

/**
 * ClusterUop（Cluster 内部解码后的 uOP 表示）
 * ----------
 * 单条 uOP 在 Cluster 内部的解码后表示（Tile 级算子配置）。
 *
 * 包含三大类信息：
 * 1) 头部 / 模式：
 *      - opcode     : 算子类型（Conv / DW / PW / Matmul / Pool / CFG）
 *      - act_type   : 激活函数类型
 *      - first_flag : 是否本 layer 的第一条 uOP
 *      - last_flag  : 是否本 layer 的最后一条 uOP
 *
 * 2) tile / 循环信息：
 *      - h_tile / w_tile    : 输出特征图 tile 高 / 宽（单位：像素）
 *      - c4_in / c4_out     : 输入/输出通道按 c4 打包后的 group 数
 *      - y_index            : 在全局输出特征图中的起始行号
 *      - qmode              : 权重量化模式（INT8 / INT4 / INT2 等）
 *      - fi_stride / fo_stride : IFM / OFM 整图跨度
 *
 * 3) 地址信息（DMA 基地址）：
 *      - coe_addr : 当前 tile 量化参数块基地址
 *      - w_addr   : 当前 tile 权重块基地址
 *      - fi_addr  : 输入特征图基地址
 *      - fo_addr  : 输出特征图基地址
 *
 * 注意：
 *   - ISA 侧只有 PARAM_ADDR，ClusterCtrl 在 uOP 解码阶段根据
 *     PARAM_ADDR + C4_OUT + Q_BYTES + ALIGN_BYTES 推导出 coe_addr / w_addr，
 *     然后写入本结构。
 */
case class ClusterUop(cfg: ClusterConfig) extends Bundle {
  // ---- 头部 ----
  val opcode = ClusterUopOpcode()
  val act_type = ClusterActType()
  val first_flag = Bool()
  val last_flag = Bool()
  val sync = Bool()

  // stride + padding（步长与 padding）
  val stride = UInt(2 bits)
  val top_pad = Bool() // PAD_TOP（仅支持 0/1 行）
  val bot_pad = Bool() // PAD_BOTTOM
  val left_pad = Bool() // PAD_LEFT（仅支持 0/1 列）
  val right_pad = Bool() // PAD_RIGHT

  // ---- tile / 循环相关参数 ----
  val h_tile = UInt(8 bits) // 输出 tile 高度（像素）
  val w_tile = UInt(8 bits) // 输出 tile 宽度（像素）

  val c4_in = UInt(10 bits) // 输入通道以 c4 分组后的 group 数
  val c4_out = UInt(10 bits) // 输出通道以 c4 分组后的 group 数

  val y_index = UInt(10 bits) // 全局输出特征图中的起始行号（y 方向）
  val qmode = ClusterQMode() // 权重量化模式（0:INT8,1:INT4,2:INT2,3:保留）

  // 整图跨度（单位由系统统一约定：如“像素组”或“word”）
  val fi_stride = UInt(16 bits) // IFM 整图跨度
  val fo_stride = UInt(16 bits) // OFM 整图跨度

  // ---- 解码阶段推导出的通用几何信息（由 CtrlUopFetch 填充）----
  val geom = ClusterUopGeom(cfg)

  // ---- 解码阶段推导出的预计算常量（由 CtrlUopFetch 填充）----
  val precalc = ClusterUopPrecalc(cfg)

  // ---- 地址参数 ----
  val coe_addr = UInt(cfg.dmaConfig.addrWidth bits) // 量化参数基地址
  val w_addr = UInt(cfg.dmaConfig.addrWidth bits) // 权重基地址
  val fi_addr = UInt(cfg.dmaConfig.addrWidth bits) // IFM 基地址
  val fo_addr = UInt(cfg.dmaConfig.addrWidth bits) // OFM 基地址

  /** 打平成 Bits，便于在某些地方做宽位寄存或比较 */
  def toBits: Bits = {
    opcode.asBits ##
      act_type.asBits ##
      first_flag.asBits ##
      last_flag.asBits ##
      sync.asBits ##
      stride.asBits ##
      top_pad.asBits ##
      bot_pad.asBits ##
      left_pad.asBits ##
      right_pad.asBits ##
      h_tile.asBits ##
      w_tile.asBits ##
      c4_in.asBits ##
      c4_out.asBits ##
      y_index.asBits ##
      qmode.asBits ##
      fi_stride.asBits ##
      fo_stride.asBits ##
      geom.ifm_h.asBits ##
      geom.ifm_w.asBits ##
      precalc.in_row_pixels.asBits ##
      precalc.in_row_bytes.asBits ##
      precalc.ibuf_line_words.asBits ##
      precalc.fi_stride_bytes.asBits ##
      precalc.row_out_words.asBits ##
      precalc.row_stride_bytes.asBits ##
      precalc.fo_plane_stride_bytes.asBits ##
      precalc.total_kernel_cnt.asBits ##
      precalc.wgt_dma_len_words.asBits ##
      precalc.fi_addr_tile.asBits ##
      precalc.fo_addr_tile.asBits ##
      precalc.fi_last_row_addr.asBits ##
      coe_addr.asBits ##
      w_addr.asBits ##
      fi_addr.asBits ##
      fo_addr.asBits
  }
}

object ClusterUop {
  /** 复位时用的默认 uOP（类似 NOP） */
  def resetValue(cfg: ClusterConfig): ClusterUop = {
    val u = ClusterUop(cfg).getZero // 所有位先清 0
    u
  }

  /** uOP 总位宽（编译期常量） */
  def width(cfg: ClusterConfig): Int =
    ClusterUop(cfg).toBits.getWidth
}

/** Cluster 执行的算子类型（内部枚举，不强制等同 ISA OPCODE 编码） */
object ClusterUopOpcode extends SpinalEnum {
  val NOP, // 空操作
  CONV2D, // 标准卷积
  PWCONV, // point-wise 卷积
  DWCONV, // depth-wise 卷积
  MATMUL, // 矩阵乘 / FC
  AVGPOOL, // 平均池化
  MAXPOOL, // 最大池化
  RESERVED, // 预留
  CFG // 配置类 uOP
  = newElement()
}

/** SFU 激活函数类型 */
object ClusterActType extends SpinalEnum {
  val NONE, // 不做激活
  RELU,
  RELU6,
  SIGMOD, // 历史命名，实际为 sigmoid
  RESERVE // 预留
  = newElement()
}

/** 量化模式（后续扩展 INT4/INT2 用） */
object ClusterQMode extends SpinalEnum {
  val Q8, // INT8 路径
  Q4, // 预留给 INT4
  Q2, // 预留给 INT2
  RESERVED // 其他/非法编码
  = newElement()
}
