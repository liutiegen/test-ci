"""集中式配置加载：把 configs/*.yaml 解析为类型化对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 默认配置目录：<项目根>/configs。可通过环境变量 QUANTFLOW_CONFIG_DIR 覆盖。
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


@dataclass(frozen=True)
class RunSpec:
    """单个量化组合。"""

    run_id: str
    algorithm: str
    scheme: str
    position: str


@dataclass(frozen=True)
class ProjectConfig:
    """project.yaml 的类型化视图。"""

    raw: dict[str, Any]

    # ---- 便捷访问器 ----
    @property
    def model_name(self) -> str:
        return self.raw["project"]["model_name"]

    @property
    def local_model(self) -> str:
        return self.raw["local_paths"]["model"]

    @property
    def local_dataset(self) -> str:
        return self.raw["local_paths"]["dataset"]

    @property
    def in_pod_model(self) -> str:
        return self.raw["oss"]["in_pod"]["model"]

    @property
    def in_pod_dataset(self) -> str:
        return self.raw["oss"]["in_pod"]["dataset"]

    @property
    def in_pod_output(self) -> str:
        return self.raw["oss"]["in_pod"]["output"]

    @property
    def image_targets(self) -> dict[str, str]:
        return dict(self.raw["images"]["targets"])

    @property
    def quant_defaults(self) -> dict[str, Any]:
        return dict(self.raw["quantization_defaults"])

    @property
    def dataset_usage(self) -> dict[str, Any]:
        return dict(self.raw["datasets"])


@dataclass(frozen=True)
class ExperimentMatrix:
    """experiment_matrix.yaml 的类型化视图。"""

    raw: dict[str, Any]
    runs: tuple[RunSpec, ...] = field(default_factory=tuple)

    def get_run(self, run_id: str) -> RunSpec:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise KeyError(f"实验矩阵中不存在 run_id={run_id}")

    def run_ids(self) -> list[str]:
        return [run.run_id for run in self.runs]

    @property
    def baseline_run_id(self) -> str:
        return self.raw["baseline"]["run_id"]


@dataclass(frozen=True)
class BenchmarkConfig:
    """benchmark.yaml 的类型化视图。"""

    raw: dict[str, Any]

    def tasks_for(self, suite: str) -> list[str]:
        return list(self.raw["suites"][suite])

    @property
    def default_suite(self) -> str:
        return self.raw["default_suite"]

    @property
    def selection(self) -> dict[str, Any]:
        return dict(self.raw["selection"])

    @property
    def perplexity(self) -> dict[str, Any]:
        return dict(self.raw["perplexity"])


def _config_dir() -> Path:
    import os

    return Path(os.environ.get("QUANTFLOW_CONFIG_DIR", DEFAULT_CONFIG_DIR))


def load_project_config(config_dir: str | Path | None = None) -> ProjectConfig:
    base = Path(config_dir) if config_dir else _config_dir()
    return ProjectConfig(raw=_read_yaml(base / "project.yaml"))


def load_experiment_matrix(config_dir: str | Path | None = None) -> ExperimentMatrix:
    base = Path(config_dir) if config_dir else _config_dir()
    raw = _read_yaml(base / "experiment_matrix.yaml")
    runs = tuple(
        RunSpec(
            run_id=item["run_id"],
            algorithm=item["algorithm"],
            scheme=item["scheme"],
            position=item["position"],
        )
        for item in raw["runs"]
    )
    return ExperimentMatrix(raw=raw, runs=runs)


def load_benchmark_config(config_dir: str | Path | None = None) -> BenchmarkConfig:
    base = Path(config_dir) if config_dir else _config_dir()
    return BenchmarkConfig(raw=_read_yaml(base / "benchmark.yaml"))
