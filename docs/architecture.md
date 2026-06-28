# Architecture Notes

## Public RTL Interface

`VenusCoreTop` exposes three stable interfaces:

- APB3 slave for control registers.
- AHB-Lite master for DMA reads/writes.
- `venus_irq` for done/error notification.

The generated Verilog port names intentionally keep SpinalHDL defaults:

```text
apb_s_PADDR, apb_s_PSEL, apb_s_PENABLE, apb_s_PREADY,
apb_s_PWRITE, apb_s_PWDATA, apb_s_PRDATA
ahb_m_HADDR, ahb_m_HWRITE, ahb_m_HSIZE, ahb_m_HBURST,
ahb_m_HPROT, ahb_m_HTRANS, ahb_m_HMASTLOCK,
ahb_m_HWDATA, ahb_m_HRDATA, ahb_m_HREADY, ahb_m_HRESP
venus_irq, clk, resetn
```

## ZYBO7010 Wrapper

`venuscore_zybo_wrapper` converts the PS-facing control interface to APB3 and routes the NPU AHB-Lite DMA master to shared BRAM.

Address map:

- Control: `0x43C0_0000`
- Shared BRAM: `0x4000_0000..0x4001_FFFF`
- IRQ: `IRQ_F2P[0]`

## Firmware Dataflow

The testvector app uses one 128KB shared BRAM window:

```text
0x40000000 + UOPS_OFFSET   -> writable uOP staging
0x40000000 + PARAMS_OFFSET -> parameter block staging
0x40000000 + ACT_OFFSET    -> activation arena
```

The compiler bundle is emitted in offset-address mode. The PS driver copies uOPs/params into shared BRAM, patches uOP W3/W4/W5 with runtime bases, starts the NPU, then invalidates and checks the output bytes.
