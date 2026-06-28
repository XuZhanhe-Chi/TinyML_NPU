package venuscore.common

import spinal.core._
import venuscore.config._
import spinal.lib._

/**
 * BusIf（DMA 侧基础接口定义）
 * ---------------------------------------------------------------------
 * 本文件集中定义 DMA 在模块间传递的通用 Bundle：
 * - `DmaCmd`：AGU/Engine 共享的“二维地址生成”描述（base/length/repeat/stride）。
 * - `DmaFetchPort`：读通道（cmd 下发 + data 返回）。
 * - `DmaStorePort`：写通道（cmd 下发 + data 发送）。
 *
 * 约定（与软件/测试保持一致）：
 * - `base/stride` 都是 byte address（单位：Byte）。
 * - `length` 是 word 数（word = `cfg.dataWidth` bits）。
 * - `repeat` 是重复次数（典型用于按行/按 tile 重复 burst）。
 */

case class DmaFetchPort(cfg: DmaConfig) extends Bundle with IMasterSlave {
  val cmd = Stream(DmaCmd(cfg))
  val data = Stream(Bits(cfg.dataWidth bits))

  override def asMaster(): Unit = {
    master(cmd)
    slave(data)
  }

  override def asSlave(): Unit = {
    slave(cmd)
    master(data)
  }

}

case class DmaStorePort(cfg: DmaConfig) extends Bundle with IMasterSlave {
  val cmd = Stream(DmaCmd(cfg))
  val data = Stream(Bits(cfg.dataWidth bits))

  override def asMaster(): Unit = {
    master(cmd)
    master(data)
  }

  override def asSlave(): Unit = {
    slave(cmd)
    slave(data)
  }
}

case class DmaCmd(cfg: DmaConfig) extends Bundle {
  // base: 起始地址（单位：byte）
  val base = UInt(cfg.addrWidth bits)
  // length: 每次 burst 内连续传输的 word 数
  val length = UInt(cfg.maxWordWidth bits)
  // repeat: 重复多少次同样模式（比如多少行）
  val repeat = UInt(cfg.maxRepeat bits)
  // stride: 相邻两次 burst 的起始地址间隔（单位：byte）
  val stride = UInt(cfg.addrWidth bits)
}
