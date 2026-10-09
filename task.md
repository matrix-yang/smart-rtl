我已在192.168.16.17:30000部署专有的模型，可以使用专有模型做一下两个测试吗？
   SMART / CVDP cid003，34 题    scripts/run_bw_cid003_full_campaign.py:99。它筛选 cid003、强制检查任务数为 34，并逐题调用 adaptive_live。

   SMART / ChipVerilog，64 题    单题入口是 scripts/smart_rtl.py:39 的 chipverilog 子命令，底层为 skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/adaptive_chipverilog_live.py:161，会检查完整任务集是否为 64 题。

最终记录结果需要对齐table1

我已在192.168.16.17:30000部署专有的模型，可以使用专有模型做一下两个测试吗？
ACE-RTL / CVDP	skills/ace-rtl/scripts/ace_cvdp_native_runner.py → skills/ace-rtl/scripts/ace_cvdp_native/cli.py:801。默认每题 5 条独立轨迹，所以 34×5=170，正好对应 Table 1。
ACE-RTL / ChipVerilog	skills/ace-rtl/scripts/ace_chipverilog_runner.py:47。全量时不传 --modules，并使用 --parallel-experiments 5；64×5=320，对应 Table 1。仓库文档也明确给出了 skills/ace-rtl/references/benchmarks/chipverilog/run-chipverilog.md:54。
最终记录结果需要对齐table1


我已在192.168.16.17:30000部署专有的模型，可以使用专有模型做一下mage两个测试吗,
ACE-RTL / CVDP	skills/ace-rtl/scripts/ace_cvdp_native_runner.py → skills/ace-rtl/scripts/ace_cvdp_native/cli.py:801。默认每题 5 条独立轨迹，所以 34×5=170，正好对应 Table 1。
ACE-RTL / ChipVerilog	skills/ace-rtl/scripts/ace_chipverilog_runner.py:47。全量时不传 --modules，并使用 --parallel-experiments 5；64×5=320，对应 Table 1。仓库文档也明确给出了 skills/ace-rtl/references/benchmarks/chipverilog/run-chipverilog.md:54。
最终记录结果需要对齐table1