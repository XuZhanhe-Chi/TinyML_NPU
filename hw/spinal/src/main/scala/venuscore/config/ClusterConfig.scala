package venuscore.config

import spinal.core._

/**
 * ClusterConfig
 * ---------------------------------------------------------------------
 * 作用：Cluster 子系统配置与派生参数（IBUF/WBUF/OBUF/SFU/MacDp 等共用）。
 *
 * 约定：
 * - `*_Width`：bit 数。
 * - `*_SizeBytes`：Byte 数（容量单位统一为 Byte）。
 * - `*_DepthWords`：word 深度（word = `ibufWordWidth/8` Byte，等价于一个像素位置的 NCHWc4 word）。
 *
 * 注意：
 * - 该配置仅做“纯派生”，不应包含与具体状态机/实现强耦合的 magic number；
 * - 若需修改容量/并行度，推荐从 `VenusCoreConfig` 的 preset 统一入口修改，避免分叉。
 */
case class ClusterConfig(
  dataWidth: Int, // 数据位宽（激活/权重，单位：bit）
  accWidth: Int, // 累加位宽（覆盖 MAC 与 SFU 输入）
  laneNum: Int, // Cluster 内 lane 数
  simdNum: Int, // 每 lane SIMD 通道数
  signedMul: Boolean, // 乘法是否有符号
  biasWidth: Int, // SFU bias 位宽
  scaleWidth: Int, // SFU scale 位宽
  shiftWidth: Int, // SFU shift 位宽
  wbufSizeBytes: Int, // WBUF 容量（Byte）
  ibufSizeBytes: Int, // IBUF 总容量（Byte，注意：3-line 总容量）
  ibufLines: Int = 3, // IBUF 行数（默认三行）
  enableRollingPrefetch: Boolean = true, // 3-line rolling（Conv/DW stride=1）是否启用“下一行预取”
  ibufCoordWidth: Int = 10, // IBUF 坐标位宽（cGroup/x）
  obufDepthWords: Int = 16, // OBUF 深度（word）
  obufAlmostFullMargin: Int = 4, // OBUF almost_full 余量阈值（word）
  dmaConfig: DmaConfig,
  memType: MemoryType = MemoryType.REG_ARRAY  // 存储类型选择
) {

  // IBUF 的 3-line buffer：按 32bit word 组织（NCHWc4，一个像素位置一个 word）。
  // 这里按“总容量/3/4Byte”估算每行可容纳的 word 数。
  val lineBufWCapWords = ibufSizeBytes / 3 / 4

  // uOP/控制侧公共宽度
  val uopWidth = 32 // uOP 逻辑宽度（byte-stream view）
  val uopIdWidth = 8 // uOP ID / seq 编号位宽（用于 trace/对齐）
  val dmaCmdWidth = 64 // DMA 命令打包宽度（base/len/repeat/stride 等）
  val addrWidth = 32 // 地址宽度（byte address，历史上固定 32）

  // WBUF（每 lane 一路读）
  val wbufWordWidth: Int = simdNum * dataWidth
  val wbufDepth: Int = wbufSizeBytes / (wbufWordWidth / 8)
  val wbufAddrWidth: Int = log2Up(wbufDepth)

  // IBUF（读口按 SIMD 宽度；内部地址按 word）
  val ibufWordWidth: Int = simdNum * dataWidth
  val ibufDepth: Int = ibufSizeBytes / (ibufWordWidth / 8)
  val ibufAddrWidth: Int = log2Up(ibufDepth)
  val ibufLineCapWords: Int = ibufDepth / ibufLines
  val ibufRowIdWidth: Int = log2Up(ibufLines)

  // LineBuf（历史：窗口行缓存，当前主要用于 DW/窗口滑动）
  val lineBufWordWidth: Int = ibufWordWidth
  val lineBufAddrWidth: Int = log2Up(lineBufWCapWords)

  // OBUF（输出回写缓冲）
  val obufWordWidth: Int = simdNum * dataWidth
  val obufAddrWidth: Int = log2Up(obufDepthWords)
  val obufLevelWidth: Int = log2Up(obufDepthWords + 1)

}
