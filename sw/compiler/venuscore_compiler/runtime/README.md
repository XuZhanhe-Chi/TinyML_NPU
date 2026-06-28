# Runtime Export

This package contains compiler-side exporters for firmware integration.

- `binary_format.py`: writes `uops.bin`, `params.bin`, `metadata.json`, optional `plan.bin`, and `bundle.h`.
- `bundle_h_parser.py`: parses generated `bundle.h` for host-side checks.
- `soc_exporter.py`: emits C arrays and metadata macros for the ZYBO7010 demo app.
- `post_checks.py`: validates bundle consistency and relocation rules.
