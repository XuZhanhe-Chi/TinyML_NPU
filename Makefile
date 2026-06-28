SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard .venv/bin/python),$(abspath .venv/bin/python),python3)
VERSION ?= v0.1.0
BOARD ?= 0

.PHONY: help env setup check test-python test-hw test-rtl test-firmware \
	hygiene rtl zybo-deps zybo-probe zybo-bitstream zybo-app zybo-run \
	release-package clean

help:
	@printf '%s\n' \
	  'TinyML_NPU targets:' \
	  '  make env [BOARD=1]     Report tool availability without changing the system' \
	  '  make setup             Create .venv and install compiler development dependencies' \
	  '  make check             Run all open-source checks; does not install dependencies' \
	  '  make rtl               Generate VenusCoreTop and VenusCoreTopBB under build/rtl' \
	  '  make zybo-bitstream    Build the Zybo bitstream and XSA with Vivado 2021.1' \
	  '  make zybo-app          Build the standalone ELF with Vitis 2021.1' \
	  '  make zybo-run          Program and validate a connected original Digilent Zybo' \
	  '  make release-package   Stage release assets under build/release/$(VERSION)'

env:
	@args=(); \
	if [[ "$(BOARD)" == "1" ]]; then args+=(--board); fi; \
	bash scripts/check_env.sh "$${args[@]}"

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e './sw/compiler[dev]'

check: test-python test-hw test-rtl test-firmware hygiene
	@echo '[CHECK] ALL PASS'

test-python:
	cd sw/compiler && "$(PYTHON)" -m pytest -q
	cd sw/compiler && "$(PYTHON)" -m examples.hand_written.conv3x3_single_tile \
	  --output-dir out/examples/conv3x3_single_tile
	cd sw/compiler && "$(PYTHON)" -m scripts.check_bundle_relocation \
	  --out-dir out/examples/conv3x3_single_tile \
	  --act-base 0x40000000 --param-base 0x40000000

test-hw: rtl
	@echo '[CHECK] SpinalHDL compile and RTL generation PASS'

test-rtl: rtl
	bash scripts/test_glue_rtl.sh
	@if [[ ! -f build/rtl/VenusCoreTop.v || ! -f build/rtl/VenusCoreTopBB.v ]]; then \
	  echo 'Run make rtl before make test-rtl.' >&2; exit 2; \
	fi
	iverilog -g2005-sv -tnull -o /tmp/tinyml_npu_zybo_check.vvp \
	  fpga/zybo7010/rtl/*.v build/rtl/VenusCoreTop.v build/rtl/VenusCoreTopBB.v

test-firmware:
	bash scripts/check_firmware.sh

hygiene:
	bash scripts/check_public_tree.sh
	bash scripts/check_markdown_links.sh

rtl:
	bash scripts/gen_rtl.sh --top VenusCoreTop
	bash scripts/gen_rtl.sh --top VenusCoreTopBB

zybo-deps:
	bash scripts/fetch_digilent_board_files.sh

zybo-probe:
	bash scripts/probe_xilinx_hw.sh

zybo-bitstream:
	bash scripts/build_zybo7010.sh

zybo-app:
	bash scripts/build_vitis_zybo7010.sh

zybo-run:
	bash scripts/run_zybo7010.sh

release-package:
	bash scripts/package_release.sh "$(VERSION)"

clean:
	rm -rf build sw/compiler/out
	find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
