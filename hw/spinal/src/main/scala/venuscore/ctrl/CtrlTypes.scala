package venuscore.ctrl

import spinal.core._
import spinal.lib._
import venuscore.cluster._
import venuscore.common._
import venuscore.config._

/**
 * CtrlFetchReq
 * ------------
 * 封装 uOP 抓取请求的参数
 * (start 信号对应 Stream 的 valid)
 */
case class CtrlFetchReq(cfg: CtrlConfig) extends Bundle {
  val uop_base = UInt(cfg.addrWidth bits) // uOP 序列基地址
  val uop_count = UInt(16 bits) // uOP 条数
}

/**
 * CtrlIsaOpCode
 * -------------
 * ISA 层 OPCODE 编码（与 VenusCore ISA 规格书保持一致）。
 *
 * OPCODE 映射：
 * 0x0 : NOP
 * 0x1 : CONV3x3
 * 0x2 : PW1x1
 * 0x3 : DW3x3
 * 0x4 : AVGPOOL
 * 0x5 : MAXPOOL
 * 0x6 : MATMUL / FC
 * 0x7~F : 保留
 */
object CtrlIsaOpCode {
  def CMD_NOP = U(0x0, 4 bits)

  def CMD_CONV3x3 = U(0x1, 4 bits)

  def CMD_PW1x1 = U(0x2, 4 bits)

  def CMD_DW3x3 = U(0x3, 4 bits)

  def CMD_AVGPOOL = U(0x4, 4 bits)

  def CMD_MAXPOOL = U(0x5, 4 bits)

  def CMD_MATMUL = U(0x6, 4 bits)
}