# ChipVerilog `mips_mips_16_core_top`：Adaptive BW-RTL vs ACE-RTL

> 文档状态：paper-supporting, artifact-grounded case study，2026-08-30  
> 对比范围：最新完整 Adaptive MIPS16 campaign 与 ACE-RTL 完整 MIPS16 comparator  
> 证据类型：同任务、同模型名、同 official evaluator 协议的 observational paired comparison

## 1. 核心结论

在 ChipVerilog `mips_mips_16_core_top` 上，Adaptive BW-RTL 的第一个、也是唯一一个
候选通过 fail-closed formal equivalence；ACE-RTL 连续提交 6 个候选后仍停在
`elaboration_fail`。按“一个候选调用一次 official evaluator subprocess”的统一口径：

| Metric | ACE-RTL | Adaptive BW-RTL | Case-level difference |
|---|---:|---:|---:|
| Correctness | FAIL | **PASS** | BW obtains a verified solution |
| Candidate / EDA calls | 6 | 1 | **83.3% fewer** |
| Sequential repair rounds | 5 | 0 | no repair needed |
| Decision rounds | fixed loop | 2 | width-1 dispatch, then stop |
| Observed wall time | 1,955.331 s | 90.761 s | 95.36% lower; 21.54x observed speedup |
| Candidate artifact class | 6/6 testbench | 1/1 target RTL | BW preserves task identity |
| Token telemetry | unavailable | 54,916 complete | no exact token ratio allowed |

Adaptive 的证明结果为 `formal_status=equivalent`、`proof_type=induction`、`k=16`。
ACE 的 6 个候选均未进入功能等价判断；它们定义的始终是
`mips_16_core_top_tb`，而不是要求的 `mips_16_core_top`。

这里的优势不是“开了更大的宽度”。恰恰相反，Director 判断该任务的计划集中度高，
只分配一个候选；收到完整 PASS 证据后立即把深度置为 stop。该 case 展示的是动态调度
能够同时避免不必要的 parallel sampling 和不必要的 sequential revision。

## 2. Case 选择：先列全集，再选样例

选择前固定以下 hard gates：双方必须使用同一 task ID、同一模型标签、同一 official
evaluator 入口与 depth/timeout，reference/testbench 对角色不可见，并且具有 terminal
artifact。通过 hard gates 后按以下顺序排序：

1. 是否存在正确性分离（BW PASS、ACE FAIL）；
2. EDA 调用是否更少；
3. 轨迹是否能解释调度机制，而不只是最终分数；
4. lineage、evaluator report 与成本记录是否完整。

完整 11-task 筛选表见 `chipverilog_mips16_case_selection.csv`。其中：

- 双方共同通过 6 个任务；
- ACE-only PASS 有 2 个：`mips_ID_stage`、`mips_hazard_detection_unit`；
- BW-only PASS 只有 1 个：`mips_mips_16_core_top`；
- 双方共同失败 2 个。

因此 `mips_mips_16_core_top` 是唯一满足最高优先级“BW PASS / ACE FAIL”的任务，
不是在多个优势样例中事后挑选数字最大的一个。负面事实也必须保留：整个 MIPS16 子集
上 Adaptive BW 为 7/11，ACE 为 8/11；本 case 不能证明总体正确率领先。

本文的 Adaptive 数据固定来自状态为 `completed` 的完整 campaign
`chipverilog_mips16_adaptive_20260830T151424Z`；其中 10 个任务为本批 live run，
`mips_alu` 从同协议 final smoke 复用一次。随后创建的
`chipverilog_mips16_failure_repair_20260830T161524Z` 在本次取数快照仍未完成，故不与
完整批次混合。ACE 固定使用覆盖全部 11 个 MIPS16 任务的
`20260820T155133Z`；`20260823T132133Z` 只重跑了部分 MIPS 任务，故不替换完整 comparator。

## 3. 对比合同与可比性

### 3.1 对齐项

- task ID：`mips_mips_16_core_top`，target top：`mips_16_core_top`；
- model label：双方均为 `deepseek-v4-pro`；
- public task：相同 target specification、宏文件和 10 个 fixed dependency RTL；
- evaluator：`tools/formal_equivalence.py verify`；
- formal depth：16；timeout：120 s；工具：Icarus/VVP/Yosys；
- target reference 与 benchmark testbench：均不向角色模型暴露；
- correctness：只有 official evaluator 的 `status=pass` 且 `verdict=pass` 才记为成功。

当前 ChipVerilog evaluator 文件 SHA-256 为
`d6f1dc2178ff95d391de9840c66be7aad0a344d0096053fbd20cebb517c330d6`，文件时间早于
两次运行。Adaptive campaign 记录了该 fingerprint；ACE 历史 invocation 记录了相同入口
和参数，但没有单独保存当时的 script hash，因此论文宜写“same evaluator entrypoint and
configuration”，不要写“historical container byte-identical”。

### 3.2 不完全对齐项

- Adaptive prompt 比 ACE prompt 多了明确的 artifact prohibition：该任务绝不是 testbench、
  stimulus、checker、coverage 或 assertion generation；
- Adaptive 角色记录了 `reasoning_effort=low`，ACE 历史记录只有 `temperature=0.2`；
- formal evaluator 对该任务没有随机 seed；LLM generation 也没有共享、已记录的 seed；
- ACE 历史 run 没有 token telemetry；
- wall time 来自不同日期的实际运行，未做机器负载归一化。

因此该结果是可复核的 paired observational case，不是只改变 scheduler 的受控消融。
Prompt contract、Planner 的任务建模和 Director 的计算分配共同造成了结果差异。

## 4. 为什么这个任务有代表性

目标模块本身是结构性顶层，但上下文并不小：public prompt 为 32,846 characters，包含
10 个 fixed dependencies。正确解要求把五级流水线、register file 和 hazard detection
unit 的接口一次性接对，包括 57/38/37-bit pipeline buses、branch/stall control 和三个
destination-register hazard signals。

这个任务的风险不在新算法，而在“任务类型识别”：模型应生成一个只做结构连接的
synthesizable target module。如果初始候选误入 testbench 生成，局部语法反馈会把搜索
吸引到错误对象内部，继续修语法而不是恢复目标 artifact。

## 5. Adaptive BW-RTL 轨迹

### 5.1 Planner：把问题压缩成单一结构计划

Planner 产生唯一计划 `direct_struct_top`：实例化 `IF_stage`、`ID_stage`、`EX_stage`、
`MEM_stage`、`WB_stage`、`register_file` 和 `hazard_detection_unit`，按公开接口连接。
它明确判断 fixed dependencies 已给出完整接口，不存在需要并行探索的状态、时序或算术
语义分岔。

### 5.2 Director：宽度设为 1

第一次 Director assessment 为：

- `spec_certainty=high`；
- `plan_concentration=high`；
- `implementation_risk=medium`，主要风险是 wiring error；
- `verification_observability=medium`。

因此 Director 只派发一个 `seed_plan`。这里 `physical_workers=1` 是执行并发限制；逻辑
宽度由 dispatch 数量决定，本轮确实是 1。

### 5.3 Builder 与 formal evidence

Builder 产生 124-line `mips_16_core_top`，候选 hash 为 `0ea1afe454aabe75`。它定义目标
模块一次，实例化七个 fixed modules，没有 candidate-contract rejection。唯一一次 EDA
调用给出：

- interface compatible；
- `formal_status=equivalent`；
- `proof_type=induction`；
- temporal induction depth `k=16`；
- evidence ID `e0001`，完整 fail-closed bundle。

### 5.4 Director：深度设为 stop

第二次 Director assessment 将 observability 提升为 `strong`、risk 降为 `low`，引用
`c0001/e0001` 发出 `stop_1`。搜索共使用 4 次 LLM 调用：Planner 1、Director 2、Builder
1；49,760 prompt + 5,156 completion = 54,916 recorded tokens。

## 6. ACE-RTL 轨迹：错误 artifact 上的局部修复闭环

| Candidate | Evaluator result | 主要现象 | 下一步 |
|---|---|---|---|
| 1 | `compile_fail` | 生成 `mips_16_core_top_tb`；line 143 `covergroup` syntax error | focused repair |
| 2 | `elaboration_fail` | 仍是 testbench；helper function 的 `return {}` 报错 | focused repair |
| 3 | `elaboration_fail` | 仍是 testbench；line 216 unexpected `@` | focused repair |
| 4 | `elaboration_fail` | 同一 line-216 `@` failure signature | focused repair |
| 5 | `elaboration_fail` | line 35 的 `return {}` 错误重新出现 | focused repair |
| 6 | `elaboration_fail` | line 216 unexpected `@` 再次出现 | max-depth stop |

第一次失败后的 debug guidance 已明确要求：用完整、可综合的
`mips_16_core_top` 替换返回代码，不要返回 testbench 或 covergroup。然而 candidate 2–6
仍全部定义 `mips_16_core_top_tb`。相邻候选确实发生修改，但搜索从未跳出错误 artifact
lineage：6/6 candidates 均为 testbench，0/6 定义目标 top。

失败还发生振荡：`unexpected {` 与 `unexpected @` 交替出现；candidate 4 修过 wait-loop
语法后，candidate 5 又恢复 `return {}` 风格。最终并非功能逻辑无法修复，而是固定深度
耗尽时仍没有产生可被验证的目标 RTL。

ACE generator prompt 从 initial 的 32,611 bytes 增长到最后一轮 56,406 bytes，增加
73.0%。这证明了 context accumulation，但不能单凭该相关性断言 context length 是未恢复
任务身份的唯一原因。

## 7. EDA 调用与成本分析

### 7.1 Case-level

- ACE：6 candidate snapshots，6 official evaluator subprocess calls；
- Adaptive BW：1 candidate，1 official evaluator subprocess call；
- 节省：5/6，即 83.3%。

这里的 EDA call 指一次 `formal_equivalence.py verify` 子进程；内部 compile、simulation、
Yosys/formal stage 不另行重复计数。Adaptive 的 1 次调用完成 k=16 induction proof；ACE
6 次调用均在 compile/elaboration gate 退出。

Observed wall time 从 1,955.331 s 降至 90.761 s，表面上减少 95.36%（21.54x）。由于两次
运行不是同机同负载的同步实验，该数字只能作为运行观察，不能作为规范化 latency claim。

### 7.2 Subset-level efficiency context

在完整 11-task MIPS16 slice 上，ACE 使用 39 次 evaluator calls，Adaptive BW 使用 17
次，减少 56.4%。但其中包含双方失败任务的 early termination，不能把总调用减少单独
解释成效率胜利。

更干净的口径是双方共同通过的 6 个任务：ACE 共使用 17 次 evaluator calls，Adaptive
BW 每题一次，共 6 次，在相同最终正确性集合上减少 64.7%。该 batch-level 结果支持
“容易/高确定性任务应少分配深度”的效率优势；`core_top` 则额外提供单任务正确性分离。

## 8. 与论文 motivation 的对应关系

### 8.1 Static allocation：一致，但需收窄

该轨迹支持“固定最大深度不能根据 failure topology 重新分配计算”。ACE 在错误 artifact
上连续走满 6 次 evaluation；Adaptive 在 high-concentration task 上选择 width 1，并在
首个正式 PASS 后停止。注意 ACE 本身也会在 PASS 时 early stop，因此不能写“ACE 对所有
简单任务强制跑满固定轮数”。准确说法应是：ACE 的候选拓扑保持单 lineage，失败时缺少
按全局状态改变宽度或重置 artifact hypothesis 的动作。

### 8.2 Irreversible repair trajectory：本 case 一致

观测到的 ACE 轨迹没有 sibling branch、rollback 或 task-hypothesis reset。每轮都从当前
testbench candidate 做 focused correction，故搜索操作上是不可逆的单 lineage。该结论
针对本次实际 trace，不应扩大为“ACE 所有版本完全没有 fresh-start 机制”。

### 8.3 Functional regression：只有弱支持

candidate 4 到 5 出现旧语法错误回归，说明局部修复非单调；但不存在“已经功能正确的
FSM 被后续 datapath 修复破坏”的证据，因为 ACE 从未生成有效目标 RTL。因此本 case 最好
称为 **failure-signature oscillation / artifact fixation**，不应作为强 functional
regression 例证。

### 8.4 Context degradation：相关证据，不是因果证明

ACE prompt 增长 73.0%，且正确的“替换为目标 RTL”指导没有在后续五个候选中落实，这与
context fixation/degradation 相符。但没有 context-length intervention，不能证明长上下文
是唯一原因。Adaptive 两个 decision rounds 后停止，没有形成 stale error history。

## 9. 我们在这个 case 中真正证明了什么

### 可以写

- 在同一 ChipVerilog task、同一模型标签和同一 formal evaluator 配置下，Adaptive BW
  首个候选通过 k=16 induction，而 ACE 6 个候选后仍 elaboration fail；
- Adaptive BW 将 evaluator calls 从 6 降为 1，减少 83.3%；
- Planner 正确识别 pure structural integration，Director 因 high plan concentration 选择
  logical width 1，并在 fail-closed PASS 后停止；
- ACE 的 6 个候选全部停留在错误 testbench artifact，局部修复出现 failure-signature
  oscillation；
- 在双方共同通过的 6 个 MIPS16 任务上，Adaptive evaluator calls 为 6，ACE 为 17。

### 不能写

- Adaptive BW 在完整 MIPS16 subset 的总体正确率高于 ACE；实际是 7/11 vs 8/11；
- 该结果单独证明 dynamic scheduler 是唯一原因；prompt contract 和 Planner 建模也不同；
- 双方是相同 LLM seed 的受控实验；没有共享、已记录的 LLM seed；
- ACE 的精确 token cost；历史 run 无 token telemetry；
- 21.54x 是硬件归一化的速度提升；
- 本 case 证明了强 functional regression 或 context degradation 的因果关系。

## 10. 论文可直接使用的文本

### Main-text paragraph

> We study `mips_mips_16_core_top`, the only task in the eleven-design
> ChipVerilog MIPS16 slice for which Adaptive BW-RTL passed while the paired
> ACE-RTL run failed. The Adaptive Planner recognized the task as a
> high-certainty structural integration problem, and the Director allocated a
> single candidate rather than speculative parallel branches. This candidate
> instantiated the five pipeline stages, register file, and hazard unit and
> passed fail-closed equivalence by temporal induction at depth 16. The
> Director then terminated the search, yielding one evaluator invocation. In
> contrast, all six ACE-RTL candidates remained testbench artifacts rather
> than the requested target module, despite feedback explicitly requesting a
> synthesizable replacement. Its linear repair trajectory alternated between
> parser failures and exhausted the five-revision budget. Thus, on this case,
> Adaptive BW-RTL changed the outcome from failure to a formally verified
> solution while reducing evaluator invocations from six to one (83.3%). This
> is an observational system-level comparison: the Adaptive run also uses a
> hardened artifact contract and structured task planning, so the result does
> not isolate scheduling as the sole causal factor.

### Figure caption

> **Case study on the ChipVerilog MIPS16 core top.** ACE-RTL followed a
> six-candidate linear repair trajectory, but every candidate remained a
> testbench and failed before functional equivalence. Adaptive BW-RTL inferred
> a single concentrated structural-wiring hypothesis, evaluated one target
> RTL candidate, proved it equivalent by induction at depth 16, and stopped in
> the next Director round. Evaluator calls decreased from six to one. The
> comparison uses the same task, model label, and evaluator configuration but
> is not a scheduler-only ablation.

## 11. 可复核材料

- 全子集筛选表：`docs/case_studies/chipverilog_mips16_case_selection.csv`
- Case 汇总指标：`docs/case_studies/chipverilog_mips16_core_top_metrics.csv`
- 逐步轨迹：`docs/case_studies/chipverilog_mips16_core_top_trace.csv`
- Mermaid 图：`docs/figures/chipverilog_mips16_core_top_case.mmd`
- Comparator contract：
  `artifacts/baseline/ace_chipverilog_mips16_20260820/json/metric_contract.json`
- Adaptive result：
  `artifacts/experiment/chipverilog_mips16_adaptive_20260830T151424Z/tasks/mips_mips_16_core_top/result.json`
- Adaptive selected RTL：同目录 `selected_candidate/rtl.sv`
- ACE result：
  `<ACE_RTL_BASELINE_ROOT>/outputs/chipverilog/20260820T155133Z/deepseek-v4-pro/mips_mips_16_core_top/summary.json`
- ACE candidate/evaluator history：同目录 `candidate_history/` 与 `evaluator_reports/`。
