# Dataset Setup

Download the CVDP v1.1.0 JSONL files from Hugging Face into a repo-local dataset folder. Do this automatically when `CVDP_DATASET_ROOT` is unset.

Run from the repository root that contains `skills/ace-rtl`:

```bash
python3 -m pip install -U huggingface_hub
python3 skills/ace-rtl/scripts/download_cvdp_datasets.py
export CVDP_DATASET_ROOT="$PWD/datasets/cvdp-benchmark-dataset"
```

The local `datasets/` folder is ignored by git. Do not commit downloaded dataset files.

The downloader fetches all files matching `cvdp_v1.1.0*.jsonl` from `nvidia/cvdp-benchmark-dataset` and verifies that all expected v1.1.0 JSONL files exist.
