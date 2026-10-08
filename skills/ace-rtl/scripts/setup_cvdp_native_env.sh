#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
CVDP_REPO="$REPO_ROOT/external/cvdp_benchmark"
PYTHON_BIN="$REPO_ROOT/.cvdp-native/bin/python3"

mkdir -p "$REPO_ROOT/external" "$REPO_ROOT/datasets"

if [ ! -d "$CVDP_REPO/.git" ]; then
  git clone https://github.com/NVlabs/cvdp_benchmark.git "$CVDP_REPO"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$REPO_ROOT/.cvdp-native"
fi

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements.txt"
"$PYTHON_BIN" "$REPO_ROOT/skills/ace-rtl/scripts/download_cvdp_datasets.py"

echo "CVDP native setup is ready."
echo "Python: $PYTHON_BIN"
echo "CVDP repo: $CVDP_REPO"
