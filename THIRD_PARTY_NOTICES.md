# Third-Party and Generated-Artifact Notices

TinyML_NPU source files authored for this repository are distributed under the Apache License 2.0 unless a file states otherwise.

## Digilent Vivado Board Files

The repository does not copy or redistribute Digilent board files. The build helper obtains the external `Digilent/vivado-boards` repository at commit:

```text
36f34ab687b7fa9c778b779d027f3bce63b3ace9
```

Users are responsible for reviewing and complying with the license and notices in that external repository. TinyML_NPU uses the original Zybo board definition `digilentinc.com:zybo:part0:2.0`.

## AMD Xilinx Tools and Generated Files

Vivado and Vitis are proprietary AMD Xilinx tools and are not distributed by this repository. GitHub Release assets with `.bit`, `.xsa`, or `.elf` extensions are generated outputs of Vivado/Vitis 2021.1 and may contain vendor-generated platform data, initialization code, libraries, or metadata.

Those generated assets are provided for reproducibility and board evaluation. The repository's Apache-2.0 license does not replace or override applicable AMD Xilinx license terms. Users must have the necessary tool and device rights and must review the vendor terms before redistributing or using generated artifacts.

## Trademarks

AMD, Xilinx, Vivado, and Vitis are trademarks of their respective owners. Digilent and Zybo are trademarks of their respective owners. Their names are used only to identify compatibility; no endorsement is implied.
