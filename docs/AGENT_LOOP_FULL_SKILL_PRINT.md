# BW-RTL 当前 Agent Loop 与完整 Skill/Prompt 打印

日期：2026-08-29  
对象：当前代码实际执行的 `RTLSearchEngine`，不是未来设计草图。  
说明：`{...}` 表示运行时注入的动态值；除此之外，下列文本保持当前实现的原始指令内容。

## 1. 节点真实性

| 节点 | 当前实现 | 是否调用 LLM | 是否有实际 prompt |
|---|---|---:|---:|
| Task Root | `TaskContract` | 否 | 否 |
| Planner | `AceLLMPlanner` | 是 | 是 |
| Plan | `PlanHypothesis` 数据节点 | 否 | 否 |
| Builder | `AceLLMBuilder` | 是 | 是 |
| Candidate | `CandidateArtifact` 数据节点 | 否 | 否 |
| Contract Gate | `CandidateArtifact.contract_issues()` | 否 | 否 |
| Verifier | `CVDPNativeVerifier` | 否 | 否，执行真实 EDA |
| Analyst | `AceLLMAnalyst` | 是，仅部分失败调用 | 是 |
| Director | `TreePolicy` | 否 | **Director 文本只计算 hash，未被消费** |
| Judge | 未实现 | 否 | 否 |
| Debate panel | 未实现 | 否 | 否 |

因此当前每一轮最多有两个 LLM call：Builder + Analyst。Planner 通常只在初始化或追加根 plan 时调用。

## 2. 当前完整 loop

```text
Public CVDP row
  |
  | sanitize reference patch/response
  v
TaskContract
  - task_id
  - public specification
  - exact target files
  - immutable public context
  |
  v
Planner (LLM, strict JSON)
  |
  +--> Plan p1 --+
  +--> Plan p2 --+--> pending deque
                       |
                       | popleft
                       v
                  Builder (LLM)
                       |
                       v
                CandidateArtifact
                       |
                       v
                 Contract Gate
                  /          \
      missing/extra/empty     complete
              |                   |
              |                   v
              |             Verifier (no LLM)
              |               preflight
              |               native CVDP EDA
              |                   |
              +-------> EDAEvidence
                              |
              +---------------+----------------+
              |               |                |
        INCOMPLETE       COMPILE/ELAB/SIM    PASS/INFRA
        skip Analyst       Analyst (LLM)     skip Analyst
              |               |                |
              +---------------+----------------+
                              |
                              v
                  TreePolicy / Controller
                       (no LLM)
                   /       |       |       \
              REFINE  BACKTRACK  EXPAND   STOP
                |         |         |        |
       appendleft child   |    add root     return
                          |    plan only if
                appendleft ancestor  no pending plan
```

实际调度细节：

1. 初始化时 Planner 生成 `branch_width` 个 plan，按顺序 `append` 到 `pending`。
2. 每一步 `popleft` 一个 work item。
3. `REFINE` 和 `BACKTRACK` 的下一工作使用 `appendleft`，所以会抢占尚未首评的其他 plan。
4. `EXPAND` 不消费 policy 给出的 `destination_id`；只有不存在 pending plan 时才让 Planner 新增一个根 plan。
5. 单次 PASS 或 infrastructure failure 都直接 STOP。

## 3. Planner 节点：完整 skill

### 3.1 调用条件

- loop 初始化：请求 `branch_width` 个 plan；
- policy 选择 EXPAND，且 pending 中不存在 plan-root work item，并且未达到 `max_plans`：再请求一个 plan。

### 3.2 System prompt

```text
You are the ACE-RTL planning role. Return strict JSON only and never write RTL.
```

### 3.3 Role instruction

```text
ROLE: RTL Planner
GOAL: derive {requested_count} materially distinct implementation plan hypothesis/hypotheses from the public RTL task.
PUBLIC SPECIFICATION:
{task.specification}
GENERATED TARGETS: {comma_separated_target_files}
EXISTING PLAN IDS: {existing_plan_ids_or_none}
RULES:
- Preserve every public interface, parameter, reset, clocking, latency, and protocol requirement.
- Treat context, harnesses, tests, and verification assets as immutable evidence, never as generated targets.
- State the architectural assumption that distinguishes each plan and list invariants the implementation must preserve.
- Do not emit RTL, do not claim verification success, and do not use private tests or golden implementations.
- Return structured plan hypotheses only; avoid cosmetic variants of an existing plan.
```

### 3.4 Adapter 追加的完整 user-prompt 尾部

Role instruction 后继续拼接：

````text
IMMUTABLE PUBLIC CONTEXT:
{each context file rendered as:
  # Context file: <path>
  ```
  <content>
  ```
or: No immutable public context files were supplied.}

Return exactly {requested_count} plan objects as JSON matching this shape:
{
  "plans": [
    {
      "plan_id": "short_unique_id",
      "summary": "implementation plan",
      "distinguishing_assumption": "why this differs",
      "invariants": [
        "public property to preserve"
      ]
    }
  ]
}
Return JSON only.  Do not include Markdown or RTL.
````

### 3.5 输出契约

```json
{
  "plans": [
    {
      "plan_id": "short_unique_id",
      "summary": "implementation plan",
      "distinguishing_assumption": "why this differs",
      "invariants": ["public property to preserve"]
    }
  ]
}
```

硬校验：plan 数量必须精确等于请求数量；ID 必须合法且唯一；summary/assumption 不得为空；summary 不能与已有 plan 完全相同。

当前缺口：Planner 只看到已有 plan ID，看不到已有 plan 内容、失败证据、invalidated assumptions 或全局搜索摘要。

## 4. Builder 节点：完整 skill

### 4.1 调用条件

每个 work item 都先调用 Builder。只有 Builder 可以生成 RTL。

### 4.2 System prompt

根据 `classify_task(primary_target, specification)` 选择：

RTL/optimization/generic：

```text
You are the ACE-RTL generator. Produce raw Icarus-compatible synthesizable SystemVerilog/Verilog source only.{MULTI_FILE_SUFFIX} Do not wrap the answer in Markdown fences or add commentary.
```

coverage/checker/verification：

```text
You are the ACE-RTL generator. Produce raw SystemVerilog verification/testbench source only.{MULTI_FILE_SUFFIX} Do not wrap the answer in Markdown fences or add commentary.
```

assertion：

```text
You are the ACE-RTL generator. Produce raw SystemVerilog source with the requested assertions only.{MULTI_FILE_SUFFIX} Do not wrap the answer in Markdown fences or add commentary.
```

多文件时 `{MULTI_FILE_SUFFIX}` 精确替换为：

```text
 If multiple target files are requested, separate them with exact `// TARGET_FILE: <path>` header comments.
```

单文件时该后缀为空。

### 4.3 Role instruction

```text
ROLE: RTL Builder
GOAL: execute action {EXPAND_OR_REFINE} for plan {plan.plan_id} and produce a complete synthesizable RTL candidate.
PLAN: {plan.summary}
DISTINGUISHING ASSUMPTION: {plan.distinguishing_assumption}
PARENT CANDIDATE: {none_or_parent_candidate_hash}
CURRENT OBLIGATION: {diagnosis.repair_obligation_or_implement_all_public_obligations}
EXACT GENERATED TARGETS: {comma_separated_target_files}
RULES:
- Return the complete content of every declared target file on every call; a patch or omitted target is invalid.
- Do not create a testbench, checker, assertion harness, golden model, or undeclared file.
- Do not edit immutable context, benchmark harnesses, tests, build scripts, or evaluator behavior.
- Preserve the public module names and interfaces and keep the result synthesizable.
- For REFINE, make the smallest evidence-grounded change that addresses the obligation without regressing satisfied behavior.
- For EXPAND, implement a genuinely different hypothesis or repair route rather than paraphrasing the parent.
- Emit RTL artifacts only; do not self-report PASS.
```

### 4.4 Adapter 追加的完整 user-prompt 尾部

```text
PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{all immutable public context files, with full content}

PARENT CANDIDATE:
{all parent target files with complete RTL, or: No parent candidate exists.}

DECLARED TARGETS:
- {target_file_1}
- {target_file_2_if_any}

OUTPUT CONTRACT:
{single-file:
  Return the complete raw source for <target_file> only.
or multi-file:
  Return one complete section per target headed exactly by `// TARGET_FILE: <path>`; include every target and no other file.}
Do not include analysis, JSON, a patch, tests, or Markdown fences.
```

### 4.5 输出契约与后处理

- 单文件：全部响应被当作该 target 的完整源码。
- 多文件：按 `// TARGET_FILE: <path>` 分段；分段失败时只会落到 primary target，随后 contract gate 拒绝缺失文件。
- 运行 `extract_code()` 和 `normalize_generated_sv()`。
- 拒绝绝对路径与含 `..` 的路径。
- 生成不可变、全文件 hash 的 `CandidateArtifact`。

当前缺口：`PlanHypothesis.invariants` 没有出现在 Builder prompt 中；INCOMPLETE/contract failure 又跳过 Analyst，所以下一次 Builder 可能只收到通用的 `implement all public obligations`。

## 5. Contract Gate 节点：完整行动规范

该节点不是 agent，不调用 LLM。

检查：

```text
expected targets - actual targets  -> missing declared target file
actual targets - expected targets  -> undeclared generated file
declared target with blank content  -> empty generated target file
```

若失败，构造：

```text
stage              = INCOMPLETE
original_stage     = LOCAL_CONTRACT
failure_class      = candidate_contract
failure_signature  = first contract issue
complete_bundle    = false
total_subtests     = 0
```

随后跳过 Verifier 和 Analyst，直接进入 TreePolicy。

## 6. Verifier 节点：完整 skill/行动规范

Verifier 不是 LLM agent，没有 system/user prompt。它的可执行规范如下：

1. 冻结并落盘完整候选及 manifest。
2. 对每个 target 运行 `high_confidence_candidate_issues()` 本地预检。
3. 若预检失败：生成 `LOCAL_PREFLIGHT` evaluator result，不运行真实 CVDP。
4. 若预检通过：把 frozen candidate 通过 `PrebuiltCandidateModel` 注入 CVDP native processor；该 model 明确不进行 LLM 调用。
5. 验证 CVDP 请求的 target 文件集合与 immutable candidate contract 完全一致。
6. 使用固定的 `COCOTB_RANDOM_SEED`（当前默认 1）和 evaluator timeout。
7. 原样保存 `real_evaluator_result.json`、summary 和 evaluator report。
8. 映射为 `EDAEvidence`：

```text
LOCAL_PREFLIGHT      -> INCOMPLETE / candidate_preflight
CANDIDATE_COMPILE    -> COMPILE    / candidate_compile
CANDIDATE_FUNCTIONAL -> SIMULATION / candidate_functional
INFRA_SETUP          -> INFRASTRUCTURE / infrastructure
real result pass     -> PASS / none
```

输出字段：

```text
stage
original_stage
failure_class
failure_signature (normalized, <= 240 chars)
complete_bundle
passed_subtests
total_subtests
mismatch_count (当前通常为空)
first_mismatch_cycle (当前通常为空)
public_feedback
```

权限边界：Verifier 不写 RTL、不修改候选、不调用 LLM，且其 EDA 结论优先于所有 LLM 判断。

当前严重缺口：`public_feedback` 使用的 evaluator report 会包含 raw service log，并被后续 Analyst 直接接收；这与“sanitized public evidence”目标不一致。

## 7. Analyst 节点：完整 skill

### 7.1 调用条件

仅在 stage 为 COMPILE、ELABORATE 或 SIMULATION 时调用。INCOMPLETE、INFRASTRUCTURE、PASS 均跳过。

### 7.2 System prompt

```text
You are the ACE-RTL failure-analysis role. Diagnose public EDA evidence and return strict JSON only.
```

### 7.3 Role instruction

```text
ROLE: RTL Failure Analyst
GOAL: convert public EDA evidence into a falsifiable diagnosis for the Director and Builder.
PLAN: {plan.plan_id} — {plan.summary}
CANDIDATE HASH: {candidate.candidate_hash}
EDA STAGE: {evidence.original_stage}
FAILURE SIGNATURE: {evidence.failure_signature}
PUBLIC EVALUATOR FEEDBACK:
{evidence.public_feedback_or_default_message}
RULES:
- Use only the public specification, immutable public context, candidate RTL, and recorded EDA evidence.
- Distinguish infrastructure, incomplete bundle, compile/elaboration, localized functional, and plan/spec failures.
- State one bounded repair obligation and whether the failure is localized.
- Mark plan conflict or alternative hypotheses only when the evidence supports it.
- Do not write RTL, modify evidence, inspect private/golden artifacts, or declare correctness.
```

默认反馈文本是：

```text
No additional public evaluator feedback was recorded.
```

### 7.4 Adapter 追加的完整 user-prompt 尾部

```text
PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{all immutable public context files, with full content}

CANDIDATE RTL:
{all current target files with complete RTL}

Return one JSON diagnosis matching:
{
  "failure_class": "concise category",
  "rationale": "evidence-grounded explanation",
  "repair_obligation": "one bounded next obligation",
  "localized_failure": null,
  "plan_conflict": false,
  "distinct_hypotheses_available": false
}
`localized_failure` must be true, false, or null. Return JSON only and never emit RTL.
```

### 7.5 输出契约

```json
{
  "failure_class": "concise category",
  "rationale": "evidence-grounded explanation",
  "repair_obligation": "one bounded next obligation",
  "localized_failure": true,
  "plan_conflict": false,
  "distinct_hypotheses_available": false
}
```

`localized_failure` 只能为 `true/false/null`；两个 flag 必须为 bool；三个文本字段不得为空。Analyst 只提供 diagnosis，不拥有最终 action 权限。

## 8. Director/Controller 节点：描述性 skill 与真实行动

### 8.1 当前 Director 描述文本

```text
ROLE: RTL Director
GOAL: manage the global executable search tree for task {task.task_id}.
RULES:
- Choose only EXPAND, REFINE, BACKTRACK, or STOP from immutable tree state, structured diagnosis, and EDA evidence.
- Hard EDA stage ordering overrides language-model confidence.
- Preserve promising ancestors; never overwrite a candidate in place.
- Backtrack on structural regression, exact repetition, or oscillation; stop only on fail-closed pass, infrastructure failure, or exhausted legal work.
- Do not write RTL, reinterpret private tests, or modify the verifier.
```

**该文本当前不会发给 LLM，也不会传给 `TreePolicy.decide()`；engine 只保存其 hash。** 因此它不是有效 skill，只是描述性元数据。

### 8.2 真实 Controller 输入

```text
SearchTree current candidate node
  - candidate hash
  - parent/ancestor lineage
  - immutable EDAEvidence

DecisionContext from Analyst
  - localized_failure: true | false | null
  - plan_conflict: bool
  - distinct_hypotheses_available: bool
  - search_exhausted: bool (当前 engine 不主动设置)
```

### 8.3 真实决策优先级

按代码执行顺序：

```text
1. evidence.stage == PASS
     -> STOP(reason=fail_closed_pass)

2. evidence.stage == INFRASTRUCTURE
     -> STOP(reason=infrastructure_failure)

3. context.search_exhausted
     -> STOP(reason=search_exhausted)

4. duplicate candidate state detected in supported ancestor pattern
     -> BACKTRACK(reason=duplicate_candidate_state,
                  destination=selected ancestor/owning plan)

5. parent reached SIMULATION but child regressed below SIMULATION
     -> BACKTRACK(reason=structural_regression,
                  destination=parent candidate)

6. last four failure signatures are A/B/A/B
     -> BACKTRACK(reason=failure_signature_oscillation,
                  destination=owning plan or root)

7. two consecutive subtest regressions
     -> BACKTRACK(reason=two_step_subtest_regression,
                  destination=grandparent candidate)

8. Analyst says plan_conflict
     -> BACKTRACK(reason=plan_or_spec_conflict,
                  destination=owning plan or root)

9. Analyst says distinct hypotheses exist OR localized_failure == false
     -> EXPAND(reason=untried_distinct_hypothesis,
               destination=current node parent)

10. stage == INCOMPLETE
      -> REFINE(reason=repair_incomplete_bundle)

11. stage in COMPILE/ELABORATE
      -> REFINE(reason=repair_structural_or_compile_failure)

12. stage == SIMULATION and localized_failure == true
      -> REFINE(reason=localized_functional_repair)

13. stage == SIMULATION otherwise
      -> REFINE(reason=functional_failure_default_refine)

14. fallback
      -> EXPAND(reason=no_local_repair_signal,
                destination=current node parent)
```

### 8.4 Action 的 engine 实际效果

```text
REFINE:
  pending.appendleft(
    parent=current candidate,
    plan=current plan,
    action=REFINE,
    diagnosis=current diagnosis)

BACKTRACK:
  mark current candidate BACKTRACKED
  mark destination ACTIVE
  pending.appendleft(
    parent=decision.destination,
    plan=current plan,
    action=EXPAND,
    diagnosis=current diagnosis)

EXPAND:
  ignore decision.destination
  if pending has no plan-root work item and max_plans permits:
      call Planner for one new root plan

STOP:
  PASS -> result.status=passed
  otherwise -> result.status=decision.reason
  selected_node=current candidate
```

这也是当前设计文档与真实执行的最大差异：Controller 并没有读取“全局 frontier 后选择一个叶子”，而是对刚产生的 candidate 做局部决策。

## 9. Judge 与 Debate

当前代码中没有 JudgeRole、DebateRole、judge prompt、debate panel 或对应 call ledger 记录。设计文档把它们列为 staged extensions，但它们不属于当前 loop，不能在实验中声称已经启用。

## 10. 如何打印某次真实运行的完整 prompt

Planner/Builder/Analyst 的最终 user prompt 会保存为：

```text
<run_dir>/roles/<role>/call_<sequence>.prompt.txt
```

对应 provider 返回：

```text
<run_dir>/roles/<role>/call_<sequence>.response.json
```

打印一个运行的全部真实 prompt：

```bash
find <run_dir>/roles -type f -name '*.prompt.txt' -print \
  -exec sed -n '1,99999p' {} \;
```

注意：这些文件可能含完整 public specification、public context、候选 RTL，以及当前实现未充分脱敏的 evaluator feedback，不应直接作为公开论文附件。

## 11. 源码对应关系

```text
loop/scheduling:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/engine.py

static role instructions:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/instructions.py

LLM prompt assembly/output schemas:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/ace_adapter.py

Builder system prompt selection:
  skills/ace-rtl/scripts/ace_cvdp_native/config.py

Verifier/evidence mapping:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/cvdp_adapter.py

Controller decision rules:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/policy.py

tree/status/backtracking:
  skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/tree.py
```
