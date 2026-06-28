package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.common._
import venuscore.config._

/**
 * Cluster 公共类型定义
 * --------------------
 * 这里集中放：
 *   - LineBuf / SFU / Lane / OBuf / ClusterCtrl 等端口 Bundle
 *   - 通用 RAM 读端口、MAC 输入等数据结构
 *
 * 说明：
 *   - 所有带 IMasterSlave 的 Bundle，只描述“信号方向语义”，
 *     真正 master/slave 角色由各模块在 new IO 时通过 master()/slave() 指定。
 */

// ============================================================================
// 1.  SFU 相关数据结构
// ============================================================================

/**
 * SFU 系数配置结构：
 *   - index : 对应哪个输出通道 / lane
 *   - bias  : 偏置
 *   - scale : 量化 scale（一般为定点）
 *   - shift : 右移位数
 *   - last  : 是否一层的最后一个配置（可选，用于结束标志）
 */
case class SfuCoe(cfg: ClusterConfig) extends Bundle {
  val bias = SInt(cfg.biasWidth bits)
  val scale = UInt(cfg.scaleWidth bits)
  val shift = UInt(cfg.shiftWidth bits)
}

/**
 * SFU 控制端口：
 *   - enable   : SFU 总使能
 *   - act_type : 激活函数类型
 *
 * 典型用法：
 *   - ClusterSFU 内部：slave(SfuCtrlPort)
 *   - ClusterCtrl 内部：master(SfuCtrlPort)
 */
case class SfuCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  val enable = Bool()
  val act_type = ClusterActType()
  val trunc_shift = Bool()
  val pool_ties_even = Bool()
  val valid = Bool()
  val busy = Bool()

  // 在 SFU 模块这一侧（被控制对象）
  override def asSlave(): Unit = {
    in(enable, act_type, trunc_shift, pool_ties_even)
    out(busy, valid)
  }

  // 在 ClusterCtrl / 上游控制器这一侧
  override def asMaster(): Unit = {
    out(enable, act_type, trunc_shift, pool_ties_even)
    in(busy, valid)
  }
}

/**
 * SFU 的累加结果输入向量：
 *   - lanes : 每个 lane 的 Int32 累加结果
 *   - mask  : 对应 lane 是否参与当前 SFU 运算
 */
case class SfuAccIn(cfg: ClusterConfig) extends Bundle {
  val lanes = Vec(SInt(cfg.accWidth bits), cfg.laneNum)
  val mask = Bits(cfg.laneNum bits) // 每 lane 一个 bit，1=参与，0=跳过
}

// ============================================================================
// 2. 通用 RAM 读端口（例如给 WBuf / 权重存储使用）
// ============================================================================

/**
 * 通用只读端口，用于描述简单的“地址 → 数据”型接口。
 *
 * 典型用法：
 *   - 在 RAM 模块内部：slave(RamRdPort)
 *   - 在地址产生器 / 控制器：master(RamRdPort)
 */
case class RamRdPort(aw: Int, dw: Int) extends Bundle with IMasterSlave {
  val rd_en = Bool()
  val rd_addr = UInt(aw bits)
  val rd_data = Bits(dw bits)

  // RAM 内部（被访问的一方）
  override def asSlave(): Unit = {
    out(rd_data)
    in(rd_addr, rd_en)
  }

  // 地址产生器 / 控制模块
  override def asMaster(): Unit = {
    in(rd_data)
    out(rd_addr, rd_en)
  }
}

// ============================================================================
// 3. WBuf / IBuf 控制与读端口
// ============================================================================

/**
 * WBuf 控制端口
 * ---------------
 * 语义：
 * - load_start      : 拉高一拍，开始一轮权重加载
 * - load_len_words  : 单个“卷积核”（kernel slot）需要的 32bit word 数（<= wbufDepth）
 * - load_kernel_cnt : 本轮要加载的卷积核个数（可以 > laneNum）
 * - load_busy       : WBuf 正在接收 wgt_stream
 * - load_done       : 本轮所有 kernel 加载完毕（1 拍脉冲）
 *
 * 写入规则（逻辑）：
 * 设 laneNum = cfg.laneNum。
 * 对于第 k 个 kernel（k 从 0 开始）：
 * 计算 bank：bank_idx  = k % laneNum
 * slot_idx  = k / laneNum      // 在该 bank 内是第几个 kernel
 * 计算基址：base_addr = slot_idx * load_len_words
 *
 * kernel 内第 j 个 word (0 <= j < load_len_words) 写入：
 * 写入：mem_banks(bank_idx)[ base_addr + j ]
 *
 * 注意：
 * - 上层软件必须保证：
 * 容量约束：ceil(load_kernel_cnt / laneNum) * load_len_words <= wbufDepth
 * 以避免某个 bank 溢出。
 */
case class WBufCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {

  // 为简化，len 与 kernel_cnt 都给足够宽度，足以覆盖
  //  wbufDepth * laneNum  级别的范围
  private val ctrlWidth = cfg.wbufAddrWidth + log2Up(cfg.laneNum)

  // 输入
  val load_start = Bool()
  val load_len_words = UInt(ctrlWidth bits) // 实际在 WBuf 内会被裁剪到 wbufAddrWidth
  val load_kernel_cnt = UInt(ctrlWidth bits) // 支持 > laneNum 的 kernel 数
  // 控制本轮数据从 WBuf 的哪一行开始存放 (用于区分 Coeffs 和 Weights 的存放区域)
  val load_base_addr = UInt(cfg.wbufAddrWidth bits)
  // 输出
  val load_busy = Bool()
  val load_done = Bool()

  // WBuf 内部：产生 busy/done，接受 start/len/kernel_cnt
  override def asSlave(): Unit = {
    out(load_busy, load_done)
    in(load_start, load_len_words, load_kernel_cnt, load_base_addr)
  }

  // ClusterCtrl：驱动 start/len/kernel_cnt，观察 busy/done
  override def asMaster(): Unit = {
    in(load_busy, load_done)
    out(load_start, load_len_words, load_kernel_cnt, load_base_addr)
  }
}


/**
 * IBuf 控制端口
 * ---------------
 * Tile 配置（读侧地址计算用）：
 *   - cfg_w_tile : 当前 tile 的有效宽度 (word 数)
 *   - cfg_c4     : 输入通道分组数（按 c4 打包）
 *   - cfg_valid  : 配置有效（打一拍即可）
 *
 * 写侧加载控制：
 *   - load_start     : 拉高一拍，开始向 load_row_id 对应物理行写数据
 *   - load_row_id    : 物理行索引 0/1/2
 *   - load_len_words : 本次写多少个 32bit word（<= lineCapWords）
 *   - load_busy      : 正在写入
 *   - load_done      : 一次加载完成（1 拍脉冲）
 */
case class IBufCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  // Tile 配置：用于读侧地址计算 (rd_cgroup * w_tile + rd_x)
  val cfg_w_tile = UInt(cfg.ibufCoordWidth bits)
  val cfg_c4 = UInt(cfg.ibufCoordWidth bits)
  val cfg_valid = Bool()

  // 写侧加载控制
  val load_start = Bool()
  val load_row_id = UInt(cfg.ibufRowIdWidth bits)
  val load_len_words = UInt(cfg.ibufAddrWidth bits)

  val load_busy = Bool()
  val load_done = Bool()

  // IBuf 内部：产生 busy/done，接受配置与 load_xxx
  override def asSlave(): Unit = {
    out(load_busy, load_done)
    in(cfg_w_tile, cfg_c4, cfg_valid, load_start, load_row_id, load_len_words)
  }

  // ClusterCtrl：驱动配置与 load_xxx，观察 busy/done
  override def asMaster(): Unit = {
    in(load_busy, load_done)
    out(cfg_w_tile, cfg_c4, cfg_valid, load_start, load_row_id, load_len_words)
  }
}

/**
 * IBuf 统一读端口（Conv / PW / MatMul / DW 共用）
 * ---------------
 * 模式选择：
 *   - rd_dw_mode = False：标量模式，一拍读一条物理行
 *   - rd_dw_mode = True ：DW 模式，一拍读三条物理行 (top/mid/bot)
 *
 * 公共地址：
 *   - rd_cgroup : C 方向 group（按 c4 打包）
 *   - rd_x      : 水平坐标 x
 *
 * 标量模式（rd_dw_mode = 0）：
 *   - 使用 rd_row_id 选择物理行
 *   - 使用 rd_data / rd_valid
 *
 * DW 模式（rd_dw_mode = 1）：
 *   - 使用 rd_row_top / rd_row_mid / rd_row_bot 选择三条物理行
 *   - 使用 rd_data_top / rd_data_mid / rd_data_bot / rd_valid
 *
 * 注意：
 *   - rd_data 是 rd_data_top 的别名，不会额外生成一份物理线网
 */
case class IBufRdPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  // 读请求
  val rd_en = Bool()
  val rd_dw_mode = Bool() // False: scalar, True: DW

  // 标量模式：选择单条物理行
  val rd_row_id = UInt(cfg.ibufRowIdWidth bits)

  // DW 模式：三条物理行映射
  val rd_row_top = UInt(cfg.ibufRowIdWidth bits)
  val rd_row_mid = UInt(cfg.ibufRowIdWidth bits)
  val rd_row_bot = UInt(cfg.ibufRowIdWidth bits)

  // 公共坐标
  val rd_cgroup = UInt(cfg.ibufCoordWidth bits)
  val rd_x = UInt(cfg.ibufCoordWidth bits)

  // 输出（内部真正存在的是 *_top/mid/bot）
  val rd_data_top = Bits(cfg.ibufWordWidth bits)
  val rd_data_mid = Bits(cfg.ibufWordWidth bits)
  val rd_data_bot = Bits(cfg.ibufWordWidth bits)
  val rd_valid = Bool()

  // 标量模式别名：复用 rd_data_top，不额外生成线网
  def rd_data: Bits = rd_data_top

  // IBuf 内部：根据模式输出对应数据
  override def asSlave(): Unit = {
    out(rd_data_top, rd_data_mid, rd_data_bot, rd_valid)
    in(rd_en, rd_dw_mode, rd_row_id, rd_row_top, rd_row_mid, rd_row_bot, rd_cgroup, rd_x)
  }

  // 上游读引擎：产生地址，消费数据
  override def asMaster(): Unit = {
    in(rd_data_top, rd_data_mid, rd_data_bot, rd_valid)
    out(rd_en, rd_dw_mode, rd_row_id, rd_row_top, rd_row_mid, rd_row_bot, rd_cgroup, rd_x)
  }
}

// ============================================================================
// 4. Lane / LineBuf / OBuf 控制端口
// ============================================================================

/**
 * Lane 控制端口
 * ---------------
 * - enable       : Lane 总使能
 * - start_pixel  : 每个输出像素的开始脉冲
 * - accum_enable : MAC 累加使能
 * - pixel_done   : 当前像素计算完成
 * - mode_avgpool : 是否按 AvgPool 模式工作
 * - mode_maxpool : 是否按 MaxPool 模式工作
 *
 * 典型用法：
 *   - PELane 内部：slave(LaneCtrlPort)
 *   - ClusterMacDp / ClusterCtrl：作为 master 端驱动 LaneCtrlPort
 */
case class LaneCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  val enable = Bool()
  val start_pixel = Bool()
  val accum_enable = Bool()
  val pixel_done = Bool()
  val mode_avgpool = Bool()
  val mode_maxpool = Bool()
  val busy = Bool()

  override def asSlave(): Unit = {
    in(enable, start_pixel, accum_enable, pixel_done, mode_avgpool, mode_maxpool)
    out(busy)
  }

  override def asMaster(): Unit = {
    out(enable, start_pixel, accum_enable, pixel_done, mode_avgpool, mode_maxpool)
    in(busy)
  }
}


/**
 * MacDpCtrlPort（MacDp 行级控制端口）
 * -------------
 * ClusterMacDp 的行级控制端口。
 *
 * 语义：
 *   - enable           : 整体使能门控（通常直接接 Cluster 级 cl_enable）
 *   - start            : 拉高一拍，表示开始处理一行新的输出像素
 *   - row_len          : 本行需要消费多少个 LineBuf 窗口（= 有效输出像素个数）
 *   - op_mode          : 算子模式，约定与 LineBufCtrlPort.cfg_op_mode 一致
 *   - c4_in            : 输入通道 NCHWc4 的 group 数（Cin 按 c4 打包后的组数）
 *   - row_top/mid/bot  : 当前输出行对应的 3 条物理 IBUF 行 ID
 *   - kernel_group_idx : 当前使用的 WBUF kernel 组索引（每组包含 laneNum 个 kernel，
 *     在 WBuf 中对应各 bank 的第几块 kernel slot）
 *   - ibuf_valid_w     : 当前 tile 内部有效 IFM 宽度（用于 X 方向边界判断）
 *   - pad_*            : Padding 方向与大小控制
 *   - wbuf_wgt_base    : 当前算子权重块的 WBuf 起始地址
 *   - busy             : MacDp 正在处理当前行
 *   - row_done         : 当前行全部像素处理完成（1 拍脉冲）
 */
case class MacDpCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  val enable = Bool()
  val start = Bool()
  val row_len = UInt(cfg.lineBufAddrWidth bits)
  val op_mode = Bits(3 bits) // 000:Conv3x3,001:PW1x1,010:DW3x3,011:Avg2x2,100:Avg3x3,101:Max2x2
  val stride = UInt(2 bits) // 空间 stride（当前支持 1/2；AvgPool 仍按内部固定逻辑处理）

  // 当前 tile 的 Cin 组数 + 本输出行的 3 行物理行 ID
  val c4_in = UInt(cfg.ibufCoordWidth bits) // 输入通道 NCHWc4 的 group 数
  val row_top = UInt(cfg.ibufRowIdWidth bits) // 逻辑 top 行对应的 IBUF rowId
  val row_mid = UInt(cfg.ibufRowIdWidth bits) // 逻辑 mid 行
  val row_bot = UInt(cfg.ibufRowIdWidth bits) // 逻辑 bot 行

  // Padding 信息
  val ibuf_valid_w = UInt(cfg.ibufCoordWidth bits)
  val pad_top_en = Bool() // 指示当前窗口的 Top 行是 Padding
  val pad_bot_en = Bool() // 指示当前窗口的 Bot 行是 Padding
  val pad_left = Bool() // 用于 X 方向地址修正
  val pad_right = Bool() // 用于 X 方向边界检查

  // 当前使用的 kernel 组索引 + 权重基地址
  val kernel_group_idx = UInt(cfg.wbufAddrWidth bits)
  val wbuf_wgt_base = UInt(cfg.wbufAddrWidth bits)

  val busy = Bool()
  val row_done = Bool()

  override def asSlave(): Unit = {
    in(enable, start, row_len, op_mode, stride, c4_in, row_top, row_mid, row_bot, kernel_group_idx, wbuf_wgt_base)
    in(ibuf_valid_w, pad_top_en, pad_bot_en, pad_left, pad_right)
    out(busy, row_done)
  }

  override def asMaster(): Unit = {
    out(enable, start, row_len, op_mode, stride, c4_in, row_top, row_mid, row_bot, kernel_group_idx, wbuf_wgt_base)
    out(ibuf_valid_w, pad_top_en, pad_bot_en, pad_left, pad_right)
    in(busy, row_done)
  }
}

/**
 * OBuf 控制端口
 * ---------------
 * - flush : 拉高一拍，清空 FIFO（wr_ptr / rd_ptr / count）
 * - empty : FIFO 为空
 * - full  : FIFO 已满
 *
 * 典型用法：
 *   - ClusterOBuf 内部：slave(OBufCtrlPort)
 *   - ClusterCtrl：作为 master 端驱动 OBufCtrlPort
 */
case class OBufCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  val flush = Bool()
  val empty = Bool()
  val full = Bool()

  override def asSlave(): Unit = {
    in(flush)
    out(empty, full)
  }

  override def asMaster(): Unit = {
    out(flush)
    in(empty, full)
  }
}

// ============================================================================
// 5. Cluster 顶层控制端口（Ctrl <-> 上级 VenusCoreCtrl）
// ============================================================================

/**
 * ClusterCtrlPort（Cluster 顶层控制/状态端口）
 * ---------------
 * 用于上级 SoC / VenusCoreCtrl 与单个 Cluster 的本地 Ctrl 交互。
 *
 * - cl_enable : 使能该 Cluster
 * - cl_flush  : 中止当前任务并清空本 Cluster（软复位）
 * - cl_id     : Cluster 编号（可用于调试/映射）
 *
 * - cl_busy   : Cluster 正在执行（非 IDLE）
 * - cl_done   : 当前 tile / uOP 完成（1 拍脉冲）
 * - cl_status : 8bit 状态/错误码
 */
case class ClusterCtrlPort(cfg: ClusterConfig) extends Bundle with IMasterSlave {
  // 控制输入（来自上级）
  val cl_enable = Bool()
  val cl_flush = Bool()
  val cl_id = Bits(4 bits)

  // 状态输出（给上级）
  val cl_busy = Bool()
  val cl_done = Bool()
  val cl_status = Bits(8 bits)

  // 在 ClusterCtrl 内部（被控制）：输出 busy/done/status，输入 enable/flush/id
  override def asSlave(): Unit = {
    out(cl_busy, cl_done, cl_status)
    in(cl_enable, cl_flush, cl_id)
  }

  // 在上级 VenusCoreCtrl / SoC 端：驱动控制，采集状态
  override def asMaster(): Unit = {
    in(cl_busy, cl_done, cl_status)
    out(cl_enable, cl_flush, cl_id)
  }
}

// ============================================================================
// 6. MAC 输入向量（Lane 用） + Cluster 顶层 IO
// ============================================================================

/**
 * Lane MAC 输入：
 *  - act_vec   : 激活向量（simdNum × dataWidth）
 *  - wgt_vec   : 权重向量（simdNum × dataWidth）
 *  - elem_mask : 每个 SIMD 通道是否参与运算
 *
 * 一般包装在 Stream[LaneMacIn] 里：
 *   - Stream.payload：LaneMacIn（每 lane 的 MAC 输入）
 *   - valid/ready 用于节拍控制
 */
case class LaneMacIn(cfg: ClusterConfig) extends Bundle {
  val act_vec = Bits(cfg.simdNum * cfg.dataWidth bits)
  val wgt_vec = Bits(cfg.simdNum * cfg.dataWidth bits)
  val elem_mask = Bits(cfg.simdNum bits)
}

/**
 * ClusterTop 聚合 IO：
 *   - uop_data      : uOP 流
 *   - sfu_cfg       : SFU 系数流
 *   - ctrl          ：ClusterCtrlPort（Cluster 控制/状态）
 *   - act_dma_port  : 激活 DMA 读端口
 *   - wgt_dma_port  : 权重 DMA 读端口
 *   - out_dma_port  : 输出 DMA 写端口
 */
case class ClusterTopIo(cfg: ClusterConfig) extends Bundle {
  // uOP 流接口
  val uop_data = slave Stream (ClusterUop(cfg))

  // Ctrl 状态 & 控制
  val ctrl = slave(ClusterCtrlPort(cfg))

  // DMA 端口（Act/Wgt/Out）
  val act_dma_port = master(DmaFetchPort(cfg.dmaConfig))
  val wgt_dma_port = master(DmaFetchPort(cfg.dmaConfig))
  val out_dma_port = master(DmaStorePort(cfg.dmaConfig))
}

trait HasClusterTopIo {
  val io: ClusterTopIo
}
