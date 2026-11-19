# QuantFlow —— 企业级 LLM 量化流水线

面向 `Qwen2.5-0.5B-Instruct` 的端到端量化项目，覆盖 **PTQ 并行量化对比 → 量化感知微调 → vLLM 服务基准**，全部在 K8s 上以可复现的 Job 编排执行。

## 三轴实验矩阵

| 维度 | 取值 |
|---|---|
| 量化算法 | smoothquant、awq |
| 量化方案 | W8A8、W4A16、W4A16_ASYM |
| 量化位置 | all、attention、mlp |

共 9 个量化组合 + 1 个 baseline（`baseline__fp16`）。命名规范：`<algorithm>__<scheme 小写>__<position>`。

## 目录结构

```
test_compress_project/
├── configs/                 # 集中式配置：project / experiment_matrix / benchmark
├── src/quantflow/           # 核心包：配置、数据、recipe、指标、QAT 桥接
├── pipelines/
│   ├── stage_1_1/           # PTQ 量化 + lm-eval 评估 + 困惑度（9 Job 并行）
│   ├── stage_1_2/           # 最优方案单节点 QAT 微调
│   └── stage_1_3/           # vLLM 三模型服务基准与准确率
├── scripts/                 # 选优等汇总脚本
└── docs/                    # 工作清单与各阶段校验清单
```

## 选优口径

主指标为相对 baseline 的综合保留率 `suite_retention_ratio`（门槛 **0.98**）；并列时依次比较困惑度（越低越好）、模型体积（越小越好）。

## 执行阶段

- **Stage 1（本地）**：编写全部代码、镜像构建文件与 K8s 编排
- **Stage 2（云上，人工暂停点）**：构建并推送三个阶段镜像，做数据盘快照
- **Stage 3（云上，校验暂停点）**：依次运行 1.1 → 选优 → 1.2 → 1.3，每步带校验门禁
