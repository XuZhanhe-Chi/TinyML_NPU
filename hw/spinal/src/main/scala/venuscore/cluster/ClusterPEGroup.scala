package venuscore.cluster

import spinal.core._
import spinal.lib._
import venuscore.config._

/**
 * ClusterPEGroup（PELane 组装与结果聚合）
 * --------------
 * 仅包含多个 PELane + 聚合逻辑:
 *   - 输入:
 *     * lane_ctrl(i): 每个 Lane 的 LaneCtrlPort（一般来自 ClusterMacDp）
 *     * lane_mac_in(i): 每个 Lane 的 MAC 输入流（来自 ClusterMacDp）
 *   - 输出:
 *     * sfu_acc_out: 聚合后送入 SFU 的 SfuAccIn 向量
 *
 * 说明:
 *   - 哪些 Lane 有效由 lane_ctrl(i).enable 决定；
 *   - sfu_acc_out.valid 只有在所有 enable 的 Lane 都有 acc_out.valid 时才拉高；
 *   - 对于 disable 的 Lane，其结果 lanes[i] 置 0，对应 mask 位为 0。
 */
case class ClusterPEGroup(cfg: ClusterConfig) extends Component {

  val io = new Bundle {
    val lane_ctrl = Vec(slave(LaneCtrlPort(cfg)), cfg.laneNum)
    val lane_mac_in = Vec(slave(Stream(LaneMacIn(cfg))), cfg.laneNum)
    val sfu_acc_out = master(Stream(SfuAccIn(cfg)))
  }
  noIoPrefix()

  // 实例化所有 PELane
  val lanes = Array.fill(cfg.laneNum)(PELane(cfg))

  for (i <- 0 until cfg.laneNum) {
    lanes(i).io.ctrl <> io.lane_ctrl(i)
    lanes(i).io.mac_in <> io.lane_mac_in(i)
  }

  val acc_streams = lanes.map(_.io.acc_out)

  // Lane 使能掩码
  val lane_enable_mask = Bits(cfg.laneNum bits)
  for (i <- 0 until cfg.laneNum) {
    lane_enable_mask(i) := io.lane_ctrl(i).enable
  }

  // 只对 enable 的 lane 检查 valid，disable 的认为“天然有效”
  val lane_valid_vec = Vec(Bool(), cfg.laneNum)
  for (i <- 0 until cfg.laneNum) {
    lane_valid_vec(i) := (!lane_enable_mask(i)) || acc_streams(i).valid
  }
  val all_valid = lane_valid_vec.asBits.andR

  val any_lane_enabled = lane_enable_mask.orR

  // 默认输出
  io.sfu_acc_out.valid := all_valid && any_lane_enabled
  io.sfu_acc_out.payload.mask := lane_enable_mask
  for (i <- 0 until cfg.laneNum) {
    io.sfu_acc_out.payload.lanes(i) := S(0, cfg.accWidth bits)
  }

  // 当所有有效 Lane 都有数据时，读取并下传
  when(all_valid) {
    for (i <- 0 until cfg.laneNum) {
      when(lane_enable_mask(i) && acc_streams(i).valid) {
        io.sfu_acc_out.payload.lanes(i) := acc_streams(i).payload
      }
    }
  }

  // 将 SFU 侧 backpressure 下传给 Lane
  val fire_all = io.sfu_acc_out.valid && io.sfu_acc_out.ready

  for (i <- 0 until cfg.laneNum) {
    acc_streams(i).ready := fire_all && lane_enable_mask(i)
  }
}

// ===========================================================================
// Verilog 生成器
// ===========================================================================
object ClusterPEGroup extends App {
  val cl_cfg = VenusCoreConfig.default.clusterCfg
  SpinalConfig(
    targetDirectory = "rtl",
    oneFilePerComponent = false,
    enumPrefixEnable = false,
    headerWithDate = false,
    anonymSignalPrefix = "tmp"
  ).generateVerilog(new ClusterPEGroup(cl_cfg)).printPruned()
}
