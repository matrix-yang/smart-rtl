# Docker Setup

Use this when Docker is available and the standard CVDP flow is desired.

```bash
export CVDP_REPO_ROOT="${CVDP_REPO_ROOT:-$PWD}"
cd "$CVDP_REPO_ROOT"
python3 -m pip install -r requirements.txt
python3 -m pip install -U huggingface_hub
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

For datasets, run `setup-datasets.md` from the repository containing `skills/ace-rtl`.

Set image variables if needed:

```bash
export OSS_SIM_IMAGE=nvidia/cvdp-sim:v1.0.0
export OSS_PNR_IMAGE=nvidia/cvdp-sim:v1.0.0
```

Commercial EDA flows need Cadence-capable images and license network configuration:

- `VERIF_EDA_IMAGE`
- `LICENSE_NETWORK`
- site license variables

Do not use Docker setup on nodes where Docker is unavailable. Use native setup instead.
