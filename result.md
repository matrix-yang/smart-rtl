# SMART 专有模型完整基准结果

- 运行 ID：`table1_chipmate_v9b_20261009T032720Z`
- 模型：`ChipMATE-V-9B`
- 服务端点：`http://192.168.16.17:30000/v1`
- 结果状态：**完整，且通过一致性审计**

## Table 1 对齐结果

| Benchmark | Method | Passed tasks | Pass rate (%) | Initiated trajectories | Candidate eval. attempts | Completed EDA evaluations | LLM call attempts | Total LLM tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CVDP cid003 | SMART / ChipMATE-V-9B | 14/34 | 41.18 | 37 | 160 | 67 | 675 | 6,571,199 |
| ChipVerilog | SMART / ChipMATE-V-9B | 10/64 | 15.62 | 80 | 215 | 116 | 885 | 14,647,149 |

## 与论文 Table 1 SMART 参考结果对照

| Benchmark | Result source | Passed tasks | Pass rate (%) | Initiated trajectories | Candidate eval. attempts | Completed EDA evaluations | LLM call attempts | Total LLM tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CVDP cid003 | 本次 ChipMATE-V-9B | 14/34 | 41.18 | 37 | 160 | 67 | 675 | 6,571,199 |
| CVDP cid003 | 论文 SMART 参考 | 31/34 | 91.18 | 71 | 156 | 154 | 322 | 5,496,661 |
| ChipVerilog | 本次 ChipMATE-V-9B | 10/64 | 15.62 | 80 | 215 | 116 | 885 | 14,647,149 |
| ChipVerilog | 论文 SMART 参考 | 22/64 | 34.38 | 78 | 191 | 166 | 741 | 15,314,876 |

| Benchmark | Passed tasks delta | Pass-rate delta (pp) | Initiated delta | Candidate delta | Completed EDA delta | LLM-call delta | Token delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVDP cid003 | -17 | -50.00 | -34 | +4 | -87 | +353 | +1,074,538 |
| ChipVerilog | -12 | -18.75 | +2 | +24 | -50 | +144 | -667,727 |

以上差异仅作描述性对照，不构成受控的模型优劣检验：论文参考结果使用了不同的模型路由；本次运行使用上下文窗口为 32,768 tokens 的专有模型端点。

## 完整性与计数审计

- 任务集完整：CVDP `cid003` 为 34/34，ChipVerilog 为 64/64，共 98 题。
- 缺失任务 0，意外任务 0，重复结果 0，审计错误 0。
- 所有非验证器角色均固定使用 `ChipMATE-V-9B`。
- 两项合计：24 题通过、117 条 initiated trajectories、375 次 candidate evaluation attempts、183 次 completed EDA evaluations、1,560 次 LLM call attempts、21,218,348 tokens。
- 终态分布：CVDP 为 passed 14、director_error 6、safety_limit 9、stopped 5；ChipVerilog 为 passed 10、director_error 8、safety_limit 17、stopped 29。
- LLM 失败或无效响应仍计入调用与 token 统计：CVDP 67 次，ChipVerilog 84 次。
- ChipVerilog 有 362 次请求因 32,768-token 上下文限制进行了 tokenizer 计量截断；CVDP 为 0 次。截断后的请求均未超出配置上下文预算。

## 结果文件

- [最终完整报告](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/FINAL_REPORT.md)
- [Table 1 Markdown 结果](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/TABLE1_RESULTS.md)
- [Table 1 JSON 结果](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/TABLE1_RESULTS.json)
- [98 题逐题结果 CSV](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/TASK_RESULTS.csv)
- [完整性审计 JSON](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/INTEGRITY_AUDIT.json)
- [运行契约、配置与指标定义](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/RUN_CONTRACT.md)

结果目录：[`artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/`](artifacts/experiment/table1_chipmate_v9b_20261009T032720Z/)
