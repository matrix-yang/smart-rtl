# Native Setup Without Docker

Use this when Docker is unavailable or when a native evaluator path is preferred.

## Official CVDP Checkout

Clone the official benchmark repository into the current workspace and use that
checkout as `CVDP_REPO_ROOT`. Do not use a CVDP checkout or environment from
outside the workspace unless the user explicitly authorizes it.

```bash
mkdir -p external
if [ ! -d external/cvdp_benchmark/.git ]; then
  git clone https://github.com/NVlabs/cvdp_benchmark.git external/cvdp_benchmark
fi
export CVDP_WORKSPACE_ROOT="$PWD"
export CVDP_REPO_ROOT="$CVDP_WORKSPACE_ROOT/external/cvdp_benchmark"
cd "$CVDP_REPO_ROOT"
git rev-parse HEAD
```

Read the official setup instructions from this cloned repository, then follow
its no-Docker/native path. Keep the Python environment local to the workspace,
for example `.cvdp-native`.

## Environment

```bash
export CVDP_NATIVE_HARNESS=1
export CVDP_NATIVE_XRUN_CMD="${CVDP_NATIVE_XRUN_CMD:-xrun}"
export CVDP_NATIVE_IMC_CMD="${CVDP_NATIVE_IMC_CMD:-imc}"
```

For Cadence coverage flows, set `MDV_XLM_HOME` to the Xcelium installation that
matches the `imc` version reading coverage data.

## Install Simulation Backends

Before installing anything, check whether the required open-source tools are
already available:

```bash
command -v iverilog vvp verilator yosys make
```

If they are not on `PATH`, first load any documented module or Conda/Mamba
environment your site provides. If tools are installed under explicit roots,
prepend their `bin` directories and check again:

```bash
export IVERILOG_HOME=/path/to/iverilog
export VERILATOR_HOME=/path/to/verilator
export YOSYS_HOME=/path/to/yosys
export PATH="$IVERILOG_HOME/bin:$VERILATOR_HOME/bin:$YOSYS_HOME/bin:$PATH"
command -v iverilog vvp verilator yosys
```

Use one of these install paths only when Docker is unavailable and the tools are
still absent. `vvp` is installed by the `iverilog` package.

Ubuntu/Debian with sudo:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git make iverilog verilator yosys
```

Rocky/RHEL/Fedora with sudo:

```bash
sudo dnf install -y epel-release || true
sudo dnf install -y python3 python3-pip git make iverilog verilator yosys
```

No sudo, with Conda/Mamba:

```bash
mamba create -y -p "$CVDP_WORKSPACE_ROOT/.cvdp-native" -c conda-forge python=3.11 pip git make iverilog verilator yosys
mamba activate "$CVDP_WORKSPACE_ROOT/.cvdp-native"
```

Commercial or coverage-oriented CVDP flows may also require Cadence tools. Load
the site EDA module or source a site-provided setup script before running the
evaluator:

```bash
module avail xcelium cadence incisive
module load <site-provided-cadence-or-xcelium-module>
command -v xrun
command -v imc
```

If your site uses installation roots instead of modules, see
`skills/eda_tool_setup/SKILL.md`.

## Verify Backends

Run this before launching CVDP:

```bash
missing=0
for tool in python3 git make iverilog vvp verilator yosys; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required native backend tool: $tool"
    missing=1
  fi
done

if [ "${CVDP_NEEDS_CADENCE:-0}" = "1" ]; then
  for tool in "$CVDP_NATIVE_XRUN_CMD" "$CVDP_NATIVE_IMC_CMD"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "missing Cadence tool: $tool"
      missing=1
    fi
  done
fi

test "$missing" -eq 0
```

Do not continue to the real evaluator until this check passes. If it fails with
missing open-source tools, install the native backend packages above. If it
fails with missing commercial tools or license checkout errors, configure your
site EDA environment and rerun the preflight.

## Native CVDP Runner

When Docker is unavailable or `docker ps` fails with daemon permission errors,
run CVDP through the workspace-local ACE native runner:

```bash
cd "$CVDP_WORKSPACE_ROOT"
export PATH="$CVDP_WORKSPACE_ROOT/.cvdp-native/bin:$PATH"
export PYTHONNOUSERSITE=1
export CVDP_NATIVE_HARNESS=1
python3 skills/ace-rtl/scripts/ace_cvdp_native_runner.py \
  --workspace "$CVDP_WORKSPACE_ROOT" \
  --cvdp-repo "$CVDP_REPO_ROOT" \
  --dataset "$CVDP_DATASET_ROOT/<dataset>.jsonl" \
  --cid <cid> \
  --generator-model <model> \
  --reflector-model <model> \
  --coordinator-model <model>
```

The native runner uses the official CVDP repository preparation and real
evaluator, but launches pytest simulation services directly with
workspace-local tools. It maps container paths such as `/code/rundir`,
`/code/rtl`, `/code/src`, `/src`, and `/rundir` in a single pass.

For commercial or coverage-oriented rows, do not start a broad native run until
the row/service preflight passes for the configured simulator command and
`imc`. Missing Cadence binaries/license setup, simulator/coverage version
mismatch, missing `MDV_XLM_HOME`, missing repo-local Python shared libraries for
cocotb VPI, missing `PYGPI_PYTHON_BIN`, or unreplaced `/code`/`/src` paths are
setup blockers, not candidate RTL failures.

Before a full sweep, run a one-row shakedown with
`--attempts 1 --max-iterations 1 --limit 1`.
