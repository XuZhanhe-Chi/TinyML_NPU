package venuscore.config

/**
 * MemoryType - Cluster 本地缓存实现选择。
 *
 * 首版公开仓库只保留 SpinalHDL Mem 路径，便于在 ZYBO7010/Vivado 流程中综合。
 */
sealed trait MemoryType

object MemoryType {
  case object REG_ARRAY extends MemoryType
}

/**
 * VenusCoreConfig
 * ---------------------------------------------------------------------
 * 作用：整机顶层配置（控制器/DMA/Cluster 的统一来源）。
 *
 * 约定：
 * - `*_Width`：bit 数。
 * - `*_SizeBytes`：Byte 数（容量单位统一为 Byte，便于与 compiler/firmware 对齐）。
 * - `*_DepthWords`：word 深度（word = `ahbDataWidth` bits）。
 */
case class VenusCoreConfig(
    // ---- 顶层总线/数据位宽 ----
    dataWidth: Int = 8,
    accWidth: Int = 32,
    ahbAddrWidth: Int = 32,
    apbAddrWidth: Int = 12,
    ahbDataWidth: Int = 32,
    apbDataWidth: Int = 32,
    sharedMemSizeBytes: Int = 128 * 1024,

    // ---- 并行度 ----
    clusterNum: Int = 1,
    lanePerCluster: Int = 4,
    simdPerLane: Int = 4,

    // ---- Cluster 子配置 ----
    signedMul: Boolean = true,
    biasWidth: Int = 24,
    scaleWidth: Int = 16,
    shiftWidth: Int = 6,
    wbufSizeBytes: Int = 2048,
    ibufSizeBytes: Int = 3072 * 4,
    ibufLines: Int = 3,
    enableRollingPrefetch: Boolean = true,
    ibufCoordWidth: Int = 12,
    obufDepthWords: Int = 256,
    obufAlmostFullMargin: Int = 4,
    memType: MemoryType = MemoryType.REG_ARRAY
) {

  def dmaCfg = DmaConfig(
    addrWidth = ahbAddrWidth,
    dataWidth = ahbDataWidth,
    // WgtDMA 一次加载长度需要覆盖：total_kernel_cnt * (c4_in * kh * kw)。
    maxWord = 8192,
    clusterNum = clusterNum
  )

  def clusterCfg = ClusterConfig(
    dataWidth = dataWidth,
    accWidth = accWidth,
    laneNum = lanePerCluster,
    simdNum = simdPerLane,
    signedMul = signedMul,
    biasWidth = biasWidth,
    scaleWidth = scaleWidth,
    shiftWidth = shiftWidth,
    wbufSizeBytes = wbufSizeBytes,
    ibufSizeBytes = ibufSizeBytes,
    ibufLines = ibufLines,
    enableRollingPrefetch = enableRollingPrefetch,
    ibufCoordWidth = ibufCoordWidth,
    obufDepthWords = obufDepthWords,
    obufAlmostFullMargin = obufAlmostFullMargin,
    dmaConfig = dmaCfg,
    memType = memType
  )

  def ctrlCfg = CtrlConfig(
    dmaConfig = dmaCfg,
    clusterConfig = clusterCfg,
    addrWidth = ahbAddrWidth,
    clusterNum = clusterNum,
    regAddrWidth = apbAddrWidth,
    apbDataWidth = apbDataWidth
  )
}

object VenusCoreConfig {
  /** ZYBO7010 首版闭环配置：1 cluster, 4 lanes, 128KB shared BRAM。 */
  val zybo7010 = VenusCoreConfig()

  val default = zybo7010
}
