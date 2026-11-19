"""quantflow —— 企业级 LLM 量化流水线工具包。

职责：
- 加载集中式配置（project / experiment_matrix / benchmark）
- 构建可复现的量化 recipe（算法 × 方案 × 位置）
- 加载并预处理 ultrachat_200k
- 汇总指标并选择最优方案
- 桥接 TRL SFTTrainer 与 llmcompressor 的量化感知训练（QAT）生命周期
"""

from quantflow.config import (
    BenchmarkConfig,
    ExperimentMatrix,
    ProjectConfig,
    load_benchmark_config,
    load_experiment_matrix,
    load_project_config,
)

__all__ = [
    "ProjectConfig",
    "ExperimentMatrix",
    "BenchmarkConfig",
    "load_project_config",
    "load_experiment_matrix",
    "load_benchmark_config",
]

__version__ = "1.0.0"
