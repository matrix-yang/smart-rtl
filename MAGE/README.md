# External MAGE benchmark adapters

This directory vendors the two MAGE benchmark adapters used for the comparison
reported alongside SMART. The source copies are kept separate by benchmark:

- `cvdp/`: copied from
  `/home/yangyifan/ai4chipdesign/cvdp_benchmark/examples/mage-codex-agent`.
- `chipverilog/`: copied from
  `/nfs/FM/yangyifan/rtl_bench/MAGE-ChipVerilog`.

Only code, configuration, documentation, and tests needed by the adapters are
included. External Git metadata, virtual environments, benchmark datasets, and
historical experiment outputs are intentionally excluded.

The copied launchers retain their original absolute-path and model defaults.
Review those settings before launching a new experiment from this repository.
