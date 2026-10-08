# BW-RTL Agent Loop 与 Skill 规范审查

日期：2026-08-29  
范围：当前 `RTLSearchEngine`、`TreePolicy`、Planner/Builder/Verifier/Analyst 适配器、ACE-RTL evaluator boundary，以及描述角色和行动的 `GeneralACEInstructions`/`skills/ace-rtl/SKILL.md`。

## 1. 审查结论

当前版本已经是一个**可执行的树状 loop 原型**，但还不是一个语义闭合的“广度—深度—回退”控制器，也不宜直接开始付费的 CVDP `dynamic_equalizer` 正式实验。

| 维度 | 结论 | 说明 |
|---|---|---|
| 端到端可执行性 | 通过 | Planner → Builder → Verifier → Analyst → policy 能跑通，四种 action 均有 smoke 覆盖 |
| 候选/目标文件隔离 | 基本通过 | 候选不可变、目标路径和完整性检查、context/target 分离已实现 |
| evaluator 隐私边界 | **不通过** | 原始 evaluator service log 被拼入 `public_feedback` 并送给 Analyst，可能暴露私有测试结构 |
| 广度/深度调度 | **不通过** | `appendleft` 形成严格 DFS；`branch_width=2` 也可能始终只搜索第一个 plan |
| 回退/目标路由 | **不通过** | policy 的 `EXPAND.destination_id` 未被 engine 执行，记录目标与实际目标可能不一致 |
| spec/plan 反思更新 | **不通过** | `plan_conflict` 只回到未修改的 plan，然后继续 Builder；Planner 看不到失败证据，也没有 plan revision |
| 全局候选池 | 部分 | artifacts 被保存，但 policy 仅比较当前祖先链，没有全局去重、跨 plan 比较或全局 frontier 选择 |
| 停止正确性 | 部分 | 单次、单 seed 的 PASS 会立即 STOP，没有独立确认，不能作为正式隐藏正确率结论 |
| telemetry | 部分 | 逻辑 role call 可见，但 provider retry、失败请求与实际计费请求未被完整区分 |
| Skill 与实现一致性 | **不通过** | 当前 ACE skill 强制原始 reflector/coordinator；新 loop 却使用 Planner/Analyst，并且所谓 Director 没有实际 role |

总判定：**机制 smoke PASS；实验有效性与 no-cheating gate FAIL。应先修 P0，再做付费 live run。**

## 2. 当前 loop 实际在做什么

当前控制流是：

1. Planner 一次生成 `branch_width` 个根 plan，并依次放入 `pending`。
2. engine 从队首取一个 work item，Builder 生成完整候选。
3. contract 通过后运行真实 Verifier；普通编译/功能失败再调用 Analyst。
4. `TreePolicy` 根据硬 EDA stage、当前祖先链和少量 Analyst flags 决定 `EXPAND/REFINE/BACKTRACK/STOP`。
5. `REFINE` 和 `BACKTRACK` 产生的工作通过 `appendleft` 插到队首，因此优先无限加深当前分支。
6. PASS 或 infrastructure failure 立即 STOP；步数耗尽则选 EDA rank 最高的候选。

这不是一个“Director 观察全局候选池并选择 frontier”的 loop。更准确的命名应是：**带 plan 前缀的确定性 DFS repair loop**。

## 3. P0：正式实验前必须修复

### P0-1：原始 evaluator log 进入 Analyst，存在 reward-hacking/测试泄漏风险

证据链：

- `llm.py:279-327` 的 `build_evaluator_error_report()` 会读取 service `log` 文件，并把文本原样附加到报告。
- `cvdp_adapter.py:217-224` 将该报告直接写入 `EDAEvidence.public_feedback`。
- `instructions.py:111-124` 把 `public_feedback` 完整放进 Analyst prompt。
- 现有 ACE 规范 `references/role-guidance.md:3-7` 明确要求只给角色 sanitized summary，禁止 raw log 泄露私有测试结构；`agent-workflow.md:29-50` 也禁止真实测试文件、私有 harness 和揭示测试结构的完整日志。

实际抽查中，`dynamic_equalizer` 报告出现了 `test_dynamic_equalizer_harness.py`、具体测试名、`Expected`/`Got` 等私有结构。它们不是 public JSONL context 的内容。

修复要求：

- raw report 只用于离线审计并留在 run directory，不得进入任何 LLM prompt。
- 新增 typed `SanitizedEDAEvidence`：stage、标准化编译错误、公开信号名、聚合 mismatch 数、首个 mismatch cycle、公开 sub-test ID（仅在 benchmark 明确声明可见时）。
- 建立 agent-visible context allowlist：public JSONL context 是否允许必须由 task contract 显式标记；路径名本身不能作为“公开”的充分条件。
- 添加 prompt-capture 测试，确保私有测试文件名、pytest case 名、expected-output 片段不出现在 Planner/Builder/Analyst 输入中。

### P0-2：所谓广度会被严格 DFS 饿死

`engine.py:196-211` 对 BACKTRACK/REFINE 使用 `pending.appendleft()`。针对性探针设置 `branch_width=2, max_steps=4`，两个 plan 均有效，第一条 plan 每次都得到 localized failure。结果：

```text
builder_plan_sequence = ['p1', 'p1', 'p1', 'p1']
step_plan_sequence    = ['p1', 'p1', 'p1', 'p1']
status                = max_steps
```

`p2` 一次也没有被执行。因此当前 `branch_width` 只表示“生成了几个 plan 节点”，不表示“探索了几个 plan”。

修复要求：

- 使用两阶段 scheduler：先让每个初始 plan 至少产生并验证一个 seed candidate，再允许深挖。
- 之后按 bounded depth quantum（例如连续最多 2 次 refine）调度，并返回全局 frontier 重新选择。
- scheduler/pending 必须成为单一真相源；测试中直接断言 `width=2` 时第二次深挖前两个 plan 均已首评。

### P0-3：`EXPAND.destination_id` 是伪语义

policy 在 `policy.py:95-109` 返回 EXPAND 时携带 `node.parent_id`，但 engine 的 EXPAND 分支 `engine.py:213-222` 不读取 `decision.destination_id`；它只在没有待执行 plan 时追加一个新根 plan。探针中 policy 指定 `plan_streaming`，下一次实际执行的 parent 却是 `plan_staged`。

这会让 trace 中“决策目标”和实际搜索路径不一致，无法用于机制分析或论文中的 action attribution。

修复要求二选一：

- 推荐：拆成 `BRANCH_PLAN` 与 `BRANCH_CANDIDATE`，两者都有明确 parent/destination contract；或
- 保留 EXPAND，但增加 `scope: PLAN | CANDIDATE`，且 engine 必须消费并验证 destination。

### P0-4：没有真正的 plan/spec reflection

与研究目标最直接冲突的问题：

- Planner 的后续输入只列 `EXISTING PLAN IDS`（`instructions.py:54-73`），没有已有 plan 内容、失败证据、已证伪假设或搜索摘要。
- `_add_plans()`（`engine.py:273-316`）没有 diagnosis/evidence 参数。
- `PlanHypothesis.invariants` 虽被解析和保存（`contracts.py:55-66`、`ace_adapter.py:256-268`），Builder instruction 却未包含 invariants（`instructions.py:84-101`）。
- Analyst 标记 `plan_conflict` 后，policy 回退到 plan 节点（`policy.py:90-93`），engine 随后仍在**同一未修改 plan**下调用 Builder。

所以当前系统只能“修代码”或“换一个事先生成的 plan”，不能根据失败反思 spec interpretation、修订 plan 或使假设失效。

修复要求：

- 增加 `PlanRevision`（也可建成新的 plan node），字段至少包括 `supersedes`、`invalidated_assumptions`、`preserved_invariants`、`new_falsifier`、`evidence_refs`。
- `plan_conflict` 必须路由给 Planner，不得直接在旧 plan 下再次 Builder。
- 新 plan/修订 plan 必须看到 sanitized global search summary，而不是 raw logs。
- Builder 每次必须收到完整 invariants 和确切 revision version。

### P0-5：STOP 只由单 seed PASS 触发

`policy.py:42-46` 对一次 PASS 立即 STOP。固定 seed 对父子比较是合理的，但单 seed PASS 不等于 hidden correctness；之前复查的历史 `dynamic_equalizer` PASS 候选在其他固定 seed 上可以失败。

修复要求：

- 搜索 seed 上 PASS 只标记 `PROVISIONAL_PASS`。
- STOP 前用至少一个 disjoint seed 或更强 oracle 进行确认。
- 最终状态拆为 `PASS_CONFIRMED`、`SEARCH_EXHAUSTED`、`INFRA_BLOCKED`，禁止混在普通失败率里。

## 4. P1：机制可信度和可复现性问题

### P1-1：contract/preflight 失败的具体义务丢失

`engine.py:170-176` 对 INCOMPLETE 不调用 Analyst；下一轮 WorkItem 没有 diagnosis，Builder 只收到通用的 `implement all public obligations`。探针中第一次候选缺少 target，第二次 Builder 没看到“missing declared target file”这一确切问题。

应从 contract/preflight evidence 确定性地产生 typed `RepairObligation`，无需额外 LLM，也要把安全的精确问题送给下一次 Builder。

### P1-2：重复与振荡检测只看局部、且漏掉真实 A/B/A/B 状态循环

- `_duplicate_destination()`（`policy.py:111-120`）只识别“与直接父节点相同”的特殊情况，不检查同 plan 的兄弟节点或全局候选池。
- `is_abab_oscillation()`（`scoring.py:73-78`）只看 failure signature。真实 trace 中更重要的是候选 bundle A/B/A/B 且失败类别不变；此时当前 policy 仍会 REFINE。

应以 `(candidate_bundle_hash, normalized_failure_signature, plan_revision)` 作为状态 key，同时做 lineage oscillation 和 plan/global duplicate 检测。

### P1-3：Director 不是 agent，却被伪装成了 agent

`instructions.director()` 生成的文本在 `engine.py:96` 只被计算 hash；`RoleSet` 根本没有 DirectorRole。实际决策是 `TreePolicy.decide()`，但每一步仍执行 `role_calls["director"] += 1`（`engine.py:188-190`）。

建议 v0 明确叫 `DeterministicController`：删除无效 Director prompt，并把 telemetry 改为 `controller_decisions`。在没有证据证明 LLM Director 带来稳定收益前，不需要增加一个昂贵角色。

### P1-4：tree 状态与真实 frontier 不一致

`SearchTree` 提供 `mark_action_tried()`、`PRUNED` 和 `active_frontier()`（`tree.py:173-207`），但 engine 不调用 action tracking，也不关闭被遗弃节点。smoke 最终 PASS 后，多个早期 plan/candidate 仍标为 ACTIVE。

这会让“可执行合法 action 是否耗尽”“frontier 大小”“搜索覆盖率”等统计失真。每个决策必须原子地更新 node status、tried actions 和 pending frontier。

### P1-5：API/解析错误不可恢复

transport retry 之外，Planner/Analyst JSON schema 错误、Builder 提取失败会终止整个 loop；API 额度耗尽后也没有逐步 checkpoint/resume。这正是此前 CVDP 重跑中已经遇到的现实故障模式。

应每步保存 tree、pending、plans、artifact refs、ledger 和 RNG/seed；schema 错误先做一次受限 repair，仍失败则作为 branch-local failure，而不是丢掉整棵树。

### P1-6：token/call telemetry 不是实际计费口径

当前 `llm_api_calls` 更接近成功的逻辑角色调用，provider 内部 retry 和失败请求可能未计入。正式比较 MAGE/ACE/BW 时必须同时报告：

- `logical_role_calls`
- `provider_requests`（含 retry）
- `successful_responses`
- `failed_or_cancelled_requests`
- prompt/completion/cached tokens，以及 telemetry 缺失标志
- verifier calls、EDA wall time、LLM wall time

## 5. Skill 规范审查

### 5.1 当前存在两套互相冲突的规范

`skills/ace-rtl/SKILL.md:48-51` 和 `references/agent-workflow.md:60-64` 要求 native runner 使用 `FocusedDebugger` 与 `FreshStartCoordinator`，并明确不要再加第二套 reflector/coordinator。BW loop 则引入 `AceLLMPlanner` 与 `AceLLMAnalyst`，没有复用这两个原始组件。

与此同时，`GeneralACEInstructions` 只是 Python prompt provider，不是一个可审计、可触发、包含输入/输出 schema 和禁止事项的正式 skill。

结论：**不要覆盖原始 ACE skill。** 应新增 `skills/bw-rtl/SKILL.md`，显式声明这是实验性机制层，在继承 ACE evaluator/setup/no-cheating 规则的同时，替换原始 loop 控制规范。

### 5.2 建议的角色契约

#### Planner

输入：TaskContract、sanitized global search summary、已有 plan/revision、invalidated assumptions、剩余未试 hypothesis classes。  
输出：`PlanProposal | PlanRevision`，包含 assumption、invariants、falsifier、supersedes、evidence refs。  
权限：不写 RTL，不看 raw/private evaluator artifacts，不宣称 PASS。

#### Builder

输入：精确 plan revision、全部 invariants、parent artifact、一个 typed repair obligation、允许的证据引用。  
输出：声明 targets 的完整 replacement files。  
权限：不生成/修改 testbench、checker、harness、build/evaluator；不自评正确性。

#### Verifier

输入：immutable candidate + benchmark adapter。  
输出：原始审计记录和独立的 sanitized structured evidence。  
权限：不调用 LLM，不修改候选，不把 private log 暴露给其他角色。

#### Analyst

输入：candidate、sanitized evidence、parent-child delta、近期状态摘要。  
输出：failure scope（local/plan/spec/infra）、confidence、repair obligation、invalidated assumptions、evidence refs。  
权限：不写 RTL，不选择最终 action，不读取 raw/private log。

#### Controller（v0 为确定性，不是 LLM agent）

输入：typed tree state、global frontier、legal actions、Analyst diagnosis。  
输出：唯一 action、精确 source/destination、reason code。  
优先级：no-cheat/contract → EDA hard stage → duplicate/oscillation/regression → breadth/depth policy → confirmation/stop。

### 5.3 建议的 action 契约

| Action | 前置条件 | 结果 |
|---|---|---|
| `SEED_PLAN` | plan 尚未首评 | 在该 plan 下生成首个候选 |
| `BRANCH_PLAN` | 现有假设不足或 plan/spec conflict | 新建或修订 plan，记录 supersedes/evidence |
| `BRANCH_CANDIDATE` | 有未试的实现路线 | 从明确 ancestor 生成 sibling，不覆盖旧节点 |
| `REFINE` | 有 localized、bounded repair obligation | 从当前候选生成 child，并设 depth quantum |
| `BACKTRACK` | 回归、重复、振荡或局部路线耗尽 | 关闭当前 branch，回到明确 ancestor/frontier |
| `CONFIRM_PASS` | search seed 上 provisional pass | 用独立 seed/强 oracle 验证 |
| `STOP` | PASS 已确认、合法工作耗尽或 infra 阻塞 | 输出明确终止类别和选中候选 |

如果暂时坚持四个公开 action，至少应让 EXPAND 带 `scope=PLAN|CANDIDATE`，并把 `CONFIRM_PASS` 作为 STOP 的强制内部 guard。

## 6. 必须新增的验收测试

1. `branch_width=2`：任何 plan 进入第二次 REFINE 前，两个 plan 都已首评。
2. EXPAND 决策的 destination 与下一候选实际 parent 完全一致。
3. 缺 target/本地 preflight 错误出现在下一次 Builder 的 typed obligation 中。
4. bundle hash A/B/A/B 且 failure signature 不变时触发 BACKTRACK。
5. 与旧 sibling/global candidate 重复的候选被拒绝，不浪费 verifier call。
6. 每次 Builder prompt 都包含当前 plan invariants；plan conflict 必须调用 Planner revision。
7. Director prompt 要么被真实消费，要么删除；确定性决策不得记为 LLM role call。
8. Analyst prompt 中不出现私有 test 文件名、pytest case、expected-output 片段；raw report 仍可离线归档。
9. provisional PASS 必须通过另一 seed 后才能 STOP/PASS_CONFIRMED。
10. runtime infrastructure failure 使用独立状态/退出码，不计入 candidate failure。
11. provider retry 和失败调用反映在实际请求/token telemetry 中。
12. 强制模拟 API 额度耗尽后可从 checkpoint 恢复，frontier 和已试 action 不改变。

## 7. 推荐实施顺序

1. **边界先行**：raw/sanitized evidence 分流，建立 prompt leakage tests。
2. **修 action 语义**：拆分 EXPAND，执行 destination，维护状态和 tried actions。
3. **修 scheduler**：initial breadth seeding + bounded depth quantum + global frontier。
4. **加入 plan revision**：让 plan/spec conflict 真正回到 Planner，贯通 invariants。
5. **加停止确认与恢复**：multi-seed confirmation、checkpoint/resume、infra 分类。
6. **修 telemetry**：按真实 provider 请求和实际 token 计量。
7. 全部离线/脚本测试通过后，再用 CVDP `cid003/dynamic_equalizer` 做第一条付费 live validation；该样例只验证 loop 可执行性，不能用于调 prompt 或硬编码策略。

## 8. 已有优点，应保留

- `TaskContract` 明确分离 generated targets 与 immutable context。
- `CandidateArtifact` 不可变，并对完整 target bundle、额外文件和空文件做检查。
- EDA stage 硬排序优先于 LLM confidence，infrastructure 与候选进度概念上分离。
- Verifier 不使用 LLM，候选注入和路径检查 fail closed。
- 固定搜索 seed 便于 parent/child 公平比较。
- 候选 hash 覆盖完整多文件 bundle。
- Planner/Builder/Analyst 的写权限边界方向正确，general prompt 没有样例专属修复提示。

这些优点说明原型适合继续演进；但现有测试主要证明“能运行”，尚不能证明“搜索语义、隐私边界和实验结论有效”。
