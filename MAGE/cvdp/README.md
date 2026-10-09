# MAGE CVDP adapter for ChipMATE-V-9B

This adapter keeps MAGE's three CVDP roles—interface/test planning, RTL
generation, and simulation review/editing—but calls an OpenAI-compatible model
endpoint directly. The direct file-block protocol is necessary because the
deployed model emits textual tool tags rather than structured OpenAI tool calls.

Build:

```bash
docker build -t cvdp-mage-chipmate-v9b:latest .
```

Smoke one Table 1 task:

```bash
bash run_single.sh cvdp_agentic_binary_to_gray_0003
```

Run the Table 1 CVDP split (exactly 34 `cid003` tasks):

```bash
bash run_full.sh
```

Important environment variables are `MAGE_API_BASE_URL`, `MAGE_MODEL`,
`MAGE_MAX_OUTPUT_TOKENS`, `MAGE_CONTEXT_CHAR_BUDGET`, `CVDP_WORK_DIR`, and
`CVDP_THREADS`. The full script filters the canonical 92-task dataset to the
paper's 34-task `cid003` split and refuses to run if the count differs.
