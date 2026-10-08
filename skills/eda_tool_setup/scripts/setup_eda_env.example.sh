#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example, site-neutral EDA setup for ACE-RTL/CVDP native runs.
# Source this file after setting any tool-root variables that apply locally:
#   source skills/eda_tool_setup/scripts/setup_eda_env.example.sh

_ace_prepend_path() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  case ":${PATH:-}:" in
    *":$dir:"*) ;;
    *) export PATH="$dir${PATH:+:$PATH}" ;;
  esac
}

_ace_prepend_ld_library_path() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
}

_ace_add_tool_root() {
  local root="$1"
  [ -n "$root" ] || return 0
  _ace_prepend_path "$root/bin"
  _ace_prepend_path "$root/tools.lnx86/bin"
  _ace_prepend_path "$root/tools.lnx86/bin/64bit"
  _ace_prepend_path "$root/tools.lnx86/inca/bin"
}

_ace_add_tool_root "${XCELIUM_HOME:-}"
_ace_add_tool_root "${VMANAGER_HOME:-}"
_ace_add_tool_root "${INCISIVE_HOME:-}"
_ace_add_tool_root "${CADENCE_HOME:-}"
_ace_add_tool_root "${VERILATOR_HOME:-}"
_ace_add_tool_root "${YOSYS_HOME:-}"
_ace_add_tool_root "${IVERILOG_HOME:-}"

export CVDP_NATIVE_HARNESS="${CVDP_NATIVE_HARNESS:-1}"
export CVDP_NATIVE_XRUN_CMD="${CVDP_NATIVE_XRUN_CMD:-xrun}"
export CVDP_NATIVE_IMC_CMD="${CVDP_NATIVE_IMC_CMD:-imc}"

if [ -n "${XCELIUM_HOME:-}" ]; then
  export MDV_XLM_HOME="${MDV_XLM_HOME:-$XCELIUM_HOME}"
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  _ace_prepend_ld_library_path "$VIRTUAL_ENV/lib"
  export PYGPI_PYTHON_BIN="${PYGPI_PYTHON_BIN:-$VIRTUAL_ENV/bin/python3}"
elif [ -d .cvdp-native/lib ]; then
  _ace_prepend_ld_library_path "$PWD/.cvdp-native/lib"
  export PYGPI_PYTHON_BIN="${PYGPI_PYTHON_BIN:-$PWD/.cvdp-native/bin/python3}"
fi

unset -f _ace_prepend_path
unset -f _ace_prepend_ld_library_path
unset -f _ace_add_tool_root
