#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download CVDP v1.1.0 dataset JSONL files from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXPECTED_FILES = [
    "cvdp_v1.1.0_agentic_code_generation_commercial.jsonl",
    "cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl",
    "cvdp_v1.1.0_agentic_heavy_code_generation.jsonl",
    "cvdp_v1.1.0_nonagentic_code_comprehension.jsonl",
    "cvdp_v1.1.0_nonagentic_code_generation_commercial.jsonl",
    "cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CVDP v1.1.0 JSONL datasets")
    parser.add_argument(
        "--target-dir",
        default="datasets/cvdp-benchmark-dataset",
        help="Local dataset directory, relative to the current working directory by default",
    )
    parser.add_argument("--repo-id", default="nvidia/cvdp-benchmark-dataset")
    parser.add_argument("--pattern", default="cvdp_v1.1.0*.jsonl")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when all expected v1.1.0 JSONL files already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()
    missing = [name for name in EXPECTED_FILES if not (target_dir / name).is_file()]
    if not missing and not args.force:
        print(target_dir)
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install the Hugging Face client first: python3 -m pip install -U huggingface_hub", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=[args.pattern],
    )

    missing = [name for name in EXPECTED_FILES if not (target_dir / name).is_file()]
    if missing:
        print("Missing expected files after download:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
