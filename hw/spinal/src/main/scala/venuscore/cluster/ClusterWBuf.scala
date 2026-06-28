package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._
import venuscore.common._

/**
 * ClusterWBuf：权重缓存（WBUF）
 * ----------------------------------
 * 功能:
 * 1. 本地权重 SRAM，接收 WgtDMA 写入
 * 2. 为 lane_num 个 Lane 提供独立的读带宽（每 Lane 一块 MEM）
 * 3. 不关心算子语义，仅做地址读写
 *
 * 公开版使用 SpinalHDL Mem 实现本地缓存，便于 Vivado 直接综合。
 *
 * 写入语义（新版，支持 load_kernel_cnt > lane_num）：
 * - ctrl.load_start      : 触发一次"权重加载轮次"
 * - ctrl.load_len_words  : 单个卷积核需要的 word 数（例如 C4*9 等），记为 len
 * - ctrl.load_kernel_cnt : 本轮要加载多少个卷积核（kernel slot），可以 >= lane_num
 *
 * 一轮加载的总 word 数为：
 * total_words = load_kernel_cnt * load_len_words（总写入 word 数）
 *
 * 写入规则（round-robin 分配 kernel 到各 bank）：
 * 设 lane_num = cfg.laneNum。
 * 对第 k 个 kernel（k 从 0 开始）：
 * bank_idx  = k % lane_num（选择写入的 bank）
 * slot_idx  = k / lane_num           // 在该 bank 内是第几个 kernel
 * base_addr = slot_idx * len         // 该 kernel 在 bank 内的起始地址
 *
 * 对该 kernel 内第 j 个 word（0 <= j < len）：
 * 写入 mem_banks(bank_idx)[ base_addr + j ]
 *
 * 上层必须保证：
 * 容量约束：ceil(load_kernel_cnt / lane_num) * load_len_words <= cfg.wbufDepth
 * 否则某个 bank 会写越界（不在本模块内做严格检查）。
 */
class ClusterWBuf(cfg: ClusterConfig) extends Component {

  // ==========================================================================
  // 基本参数检查
  // ==========================================================================
  assert(cfg.wbufWordWidth == 32, "ClusterWBuf requires 32-bit word width (SIMD4 * INT8)")
  assert(cfg.laneNum == 4, "ClusterWBuf currently designed for 4 Lanes")

  private val lane_num = cfg.laneNum
  private val wbuf_addr_width = cfg.wbufAddrWidth
  private val wbuf_word_width = cfg.wbufWordWidth

  // ==========================================================================
  // IO 定义
  // ==========================================================================
  val io = new Bundle {
    // 写侧数据端口：来自 WgtDMA 的权重数据流（数据平面）
    val wgt_stream = slave(Stream(Bits(wbuf_word_width bits)))

    // 读侧数据端口：给每个 Lane 的独立读口（数据平面）
    val rd_port = Vec(slave(RamRdPort(wbuf_addr_width, wbuf_word_width)), lane_num)

    // 控制端口：来自 ClusterCtrl（控制平面）
    val ctrl = slave(WBufCtrlPort(cfg))
  }
  noIoPrefix()

  // ==========================================================================
  // 存储阵列实现: 每个 Lane 一块 MEM
  // ==========================================================================
  val wbuf_mem_banks = Seq.fill(lane_num) {
    Mem(Bits(wbuf_word_width bits), cfg.wbufDepth)
  }
  noIoPrefix()

  // ==========================================================================
  // 写入控制寄存器
  // ==========================================================================
  // 当前是否在接收一轮权重
  val load_busy_reg = Reg(Bool()) init (False)
  // 一轮结束时拉高 1 拍
  val load_done_reg = Reg(Bool()) init (False)

  // 记录本轮配置
  // len: 每个 kernel 的 word 数（按单 bank 深度裁剪）
  val load_len_reg = Reg(UInt(wbuf_addr_width bits)) init (0)

  // kernel 总数（本轮），宽度给足
  private val kernel_cnt_width = cfg.wbufAddrWidth + log2Up(cfg.laneNum)
  val kernel_cnt_reg = Reg(UInt(kernel_cnt_width bits)) init (0)

  // 当前正在写的是第几个 kernel（0-based）
  val cur_kernel_idx_reg = Reg(UInt(kernel_cnt_width bits)) init (0)

  // 当前 kernel 内部的 word 地址（0..load_len_reg-1）
  val wr_word_idx_reg = Reg(UInt(wbuf_addr_width bits)) init (0)

  // 每个 bank 的“下一个 kernel 起始地址”
  val bank_base_addr_reg = Vec(
    Reg(UInt(wbuf_addr_width bits)) init (0),
    lane_num
  )

  // 对外 busy/done
  io.ctrl.load_busy := load_busy_reg
  io.ctrl.load_done := load_done_reg

  // done 默认一拍脉冲
  load_done_reg := False

  // 在一轮 load 期间才接受流
  io.wgt_stream.ready := load_busy_reg

  // 当前 kernel 是不是最后一个 kernel，以及是不是该 kernel 的最后一个 word
  val is_last_word_in_kernel = wr_word_idx_reg === (load_len_reg - 1)
  val is_last_kernel = cur_kernel_idx_reg === (kernel_cnt_reg - 1).resized

  // 计算当前 kernel 对应的 bank 索引：round-robin
  // 这里因为 lane_num = 4，kernel_idx 的低 2bit 就是 bank_idx
  val bank_idx_width = log2Up(lane_num)
  val cur_bank_idx = UInt(bank_idx_width bits)
  cur_bank_idx := cur_kernel_idx_reg(bank_idx_width - 1 downto 0)

  // ==========================================================================
  // 启动一轮权重加载
  // ==========================================================================
  when(io.ctrl.load_start && !load_busy_reg) {
    // 运行时约束（仿真时尽可能早暴雷）
    assert(io.ctrl.load_kernel_cnt =/= 0, "WBuf load_kernel_cnt must be > 0 when load_start=1")
    assert(io.ctrl.load_len_words =/= 0, "WBuf load_len_words must be > 0 when load_start=1")
    assert(io.ctrl.load_len_words.resized <= cfg.wbufDepth, "WBuf load_len_words must be <= wbufDepth")

    load_busy_reg := True

    // len 控制裁剪到 addrWidth
    load_len_reg := io.ctrl.load_len_words.resized
    kernel_cnt_reg := io.ctrl.load_kernel_cnt.resized

    cur_kernel_idx_reg := 0
    wr_word_idx_reg := 0

    // 每轮加载从 bank 内地址 0 开始
    for (i <- 0 until lane_num) {
      bank_base_addr_reg(i) := io.ctrl.load_base_addr
    }
  }

  // ==========================================================================
  // 写入数据路径
  // 每当 wgt_stream.fire（valid && ready）时写入一个 word
  // ==========================================================================
  when(io.wgt_stream.fire && load_busy_reg) {

    // 对每个 bank 生成写入逻辑，只有匹配当前 bank_idx 的那一个会真正写入
    for (i <- 0 until lane_num) {
      val sel_bank = cur_bank_idx === U(i, cur_bank_idx.getWidth bits)
      // 当前 bank 的写地址 = bank_base_addr + wr_word_idx
      val wr_addr = (bank_base_addr_reg(i) + wr_word_idx_reg).resized

      wbuf_mem_banks(i).write(
        enable = sel_bank,
        address = wr_addr,
        data = io.wgt_stream.payload
      )
    }

    // 更新 word 计数
    when(is_last_word_in_kernel) {
      // 一个 kernel 内最后一个 word 写完，准备切换到下一个 kernel
      wr_word_idx_reg := 0

      // 当前 kernel 对应 bank 的 base_addr += len
      for (i <- 0 until lane_num) {
        val sel_bank = cur_bank_idx === U(i, cur_bank_idx.getWidth bits)
        when(sel_bank) {
          bank_base_addr_reg(i) := (bank_base_addr_reg(i) + load_len_reg).resized
        }
      }

      // 如果还不是最后一个 kernel，则切到下一个 kernel
      when(!is_last_kernel) {
        cur_kernel_idx_reg := cur_kernel_idx_reg + 1

        // bank_idx 轮转：0,1,2,3,0,1,...
        when(cur_bank_idx === U(lane_num - 1, bank_idx_width bits)) {
          // 下一个 kernel 回到 bank0
          // cur_bank_idx 本身是组合信号，不需要寄存；下一个周期会通过 cur_kernel_idx_reg 重新计算
        } otherwise {
          // 同样，通过 cur_kernel_idx_reg 自然更新，无须专门寄存 bank_idx
        }
      } otherwise {
        // 整个加载轮次结束
        load_busy_reg := False
        load_done_reg := True
      }

    } otherwise {
      // 仍在当前 kernel 内，继续写下一个 word
      wr_word_idx_reg := wr_word_idx_reg + 1
    }
  }

  // ==========================================================================
  // 读出逻辑：每个 Lane 独立读口
  // ==========================================================================
  for (i <- 0 until lane_num) {
    val mem = wbuf_mem_banks(i)
    io.rd_port(i).rd_data := mem.readSync(
      enable = io.rd_port(i).rd_en,
      address = io.rd_port(i).rd_addr
    )
  }
}

// ==============================
// Verilog 生成入口
// ==============================
object ClusterWBuf extends App {
  val fpgaCfg = VenusCoreConfig.default.clusterCfg

  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new ClusterWBuf(fpgaCfg)).printPruned()
}
