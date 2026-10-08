---
name: eda_tool_setup
description: Use when CVDP or ACE-RTL native runs need simulator, synthesis, coverage, or license setup before classifying EDA-tool failures as INFRA_SETUP.
---

# EDA Tool Setup

Use this skill before declaring an EDA binary, simulator, synthesis tool,
coverage tool, or license unavailable for a CVDP native run.

## Rules

- First inspect the selected CVDP row and harness services. Only configure
  commercial tools when the selected services require them.
- Check the current shell first with `command -v <tool>` and the vendor's
  non-mutating version command, for example `xrun -version`, `imc -version`, or
  `verilator --version`.
- For open-source tools, discover before installing: check `PATH`, then any
  active module or Conda/Mamba environment, then common roots such as
  `IVERILOG_HOME`, `VERILATOR_HOME`, and `YOSYS_HOME` with their `bin`
  directories prepended to `PATH`.
- Prefer documented site setup methods such as environment modules, vendor
  setup scripts, or administrator-provided shell fragments. Do not assume any
  organization-specific filesystem layout.
- Keep CVDP data, Python environments, datasets, prompts, scripts, and outputs
  repository-local. Only prepend tool binary directories and set vendor license
  variables in the evaluation environment.
- Record the exact tool commands and license-related environment variables used
  in the run report or setup notes. Do not record secret values or license file
  contents.
- Do not treat simulator or coverage failures as candidate RTL failures when
  logs show missing binaries, license checkout errors, mismatched
  simulator/coverage database versions, missing shared libraries, or malformed
  native path mapping.

## Common Native Tools

No-commercial CVDP rows commonly use:

```bash
command -v python3 git make iverilog vvp verilator yosys
```

If any are missing after environment discovery, install them through the system
package manager, Conda/Mamba, or another documented local package source.

Commercial or coverage-oriented rows may require Cadence tools:

```bash
command -v xrun imc
command -v irun || true
```

Set these when the tool names differ from the harness defaults:

```bash
export CVDP_NATIVE_XRUN_CMD="${CVDP_NATIVE_XRUN_CMD:-xrun}"
export CVDP_NATIVE_IMC_CMD="${CVDP_NATIVE_IMC_CMD:-imc}"
```

If your site exposes Cadence installation roots rather than modules, set one or
more of these and source the example setup script:

```bash
export XCELIUM_HOME=/path/to/xcelium
export VMANAGER_HOME=/path/to/vmanager
export INCISIVE_HOME=/path/to/incisive
source skills/eda_tool_setup/scripts/setup_eda_env.example.sh
```

## License Variables

Use your vendor or site documentation for license values. Common variables are:

```bash
export CDS_LIC_FILE="port@cadence-license-server"
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-port@license-server}"
export SNPSLMD_LICENSE_FILE="port@synopsys-license-server"
export MGLS_LICENSE_FILE="port@siemens-license-server"
```

For Cadence coverage flows, `MDV_XLM_HOME` should point to the Xcelium
installation compatible with the `imc` version that reads the coverage database.

For cocotb/Xcelium VPI, make the active Python shared library visible:

```bash
export PYGPI_PYTHON_BIN="$PWD/.cvdp-native/bin/python3"
export LD_LIBRARY_PATH="$PWD/.cvdp-native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
