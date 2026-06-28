# Changelog

All notable changes to TinyML_NPU are documented here. The project follows semantic versioning for tagged public releases.

## [Unreleased]

No unreleased user-facing changes.

## [0.1.0] - 2026-06-28

### Added

- Public SpinalHDL source for the VenusCore INT8 NPU.
- Restricted ONNX QDQ compiler path, hand-written operator examples, bundle exporter, relocation checks, and behavioral simulation.
- Original Digilent Zybo XC7Z010 batch flow for Vivado/Vitis 2021.1.
- Bare-metal KWS testvector application and versioned JTAG result ABI.
- Protocol regressions for the AXI-Lite/APB3 and AHB-Lite/BRAM bridges.
- Reproducible root Makefile, environment checks, CI, release packaging, and project documentation.

### Verified

- 44-uOP KWS bundle passes 120 balanced KWS samples on physical hardware with 120/120 reference top1 matches, 117/120 label accuracy, and maximum INT8 error 0 under tolerance 5.
- Post-route timing meets 50 MHz with +3.025 ns WNS on the reference build.

[Unreleased]: https://github.com/XuZhanhe-Chi/TinyML_NPU/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/XuZhanhe-Chi/TinyML_NPU/releases/tag/v0.1.0
