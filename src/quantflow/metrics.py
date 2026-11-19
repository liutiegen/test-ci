"""指标汇总、综合保留率计算与最优方案选择。

lm-eval 的结果形如 {"gsm8k": {"exact_match,none": 0.4, ...}, ...}，
本模块将其拍平、按任务主指标对齐，计算相对 baseline 的综合保留率并选优。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantflow.logging_utils import get_logger

logger = get_logger(__name__)

# 每个任务用于横向对比的主指标（取 acc_norm 优先，其次 acc / exact_match）
TASK_PRIMARY_METRIC = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "word_perplexity,none",
)


def flatten_lm_eval(results: dict[str, Any]) -> dict[str, float]:
    """从 lm-eval results 中提取每个任务的主指标值。"""
    flat: dict[str, float] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for candidate in TASK_PRIMARY_METRIC:
            if candidate in metrics:
                flat[task_name] = float(metrics[candidate])
                break
    return flat


@dataclass
class RunMetrics:
    run_id: str
    task_scores: dict[str, float]
    perplexity: float | None
    model_size_bytes: int | None
    raw: dict[str, Any]


def load_run_metrics(path: str | Path) -> RunMetrics:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return RunMetrics(
        run_id=raw["run_id"],
        task_scores=flatten_lm_eval(raw.get("results", {})),
        perplexity=raw.get("perplexity"),
        model_size_bytes=raw.get("model_size_bytes"),
        raw=raw,
    )


def _common_tasks(
    baseline: RunMetrics, candidate: RunMetrics
) -> list[str]:
    return [t for t in baseline.task_scores if t in candidate.task_scores]


def suite_retention_ratio(
    baseline: RunMetrics, candidate: RunMetrics
) -> float:
    """综合保留率：候选各任务得分相对 baseline 的平均比值。

    对 baseline 得分为 0 的任务跳过，避免除零。
    """
    ratios: list[float] = []
    for task in _common_tasks(baseline, candidate):
        base = baseline.task_scores[task]
        if base == 0:
            continue
        ratios.append(candidate.task_scores[task] / base)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


@dataclass
class Ranking:
    run_id: str
    retention: float
    perplexity: float | None
    model_size_bytes: int | None
    eligible: bool


def rank_runs(
    baseline: RunMetrics,
    candidates: list[RunMetrics],
    retention_floor: float = 0.98,
) -> list[Ranking]:
    rankings: list[Ranking] = []
    for candidate in candidates:
        retention = suite_retention_ratio(baseline, candidate)
        rankings.append(
            Ranking(
                run_id=candidate.run_id,
                retention=retention,
                perplexity=candidate.perplexity,
                model_size_bytes=candidate.model_size_bytes,
                eligible=retention >= retention_floor,
            )
        )

    # 排序：合格优先，保留率降序；再按困惑度升序、体积升序
    rankings.sort(
        key=lambda r: (
            not r.eligible,
            -r.retention,
            r.perplexity if r.perplexity is not None else float("inf"),
            r.model_size_bytes if r.model_size_bytes is not None else float("inf"),
        )
    )
    return rankings


def select_best(rankings: list[Ranking]) -> Ranking | None:
    """选出最优方案；若无合格方案返回 None（按 Rule 应暂停要求修正）。"""
    eligible = [r for r in rankings if r.eligible]
    if not eligible:
        logger.error("没有方案达到保留率门槛，需暂停并人工介入")
        return None
    return rankings[0]


def write_markdown_report(
    rankings: list[Ranking], output_path: str | Path, best_id: str | None
) -> None:
    lines = [
        "# 量化方案对比报告",
        "",
        "| 排名 | run_id | 综合保留率 | 困惑度 | 模型体积(B) | 是否达标 |",
        "|---|---|---|---|---|---|",
    ]
    for idx, ranking in enumerate(rankings, start=1):
        marker = " **BEST**" if ranking.run_id == best_id else ""
        lines.append(
            f"| {idx} | {ranking.run_id}{marker} | {ranking.retention:.4f} | "
            f"{ranking.perplexity if ranking.perplexity is not None else '-'} | "
            f"{ranking.model_size_bytes if ranking.model_size_bytes is not None else '-'} | "
            f"{'是' if ranking.eligible else '否'} |"
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("对比报告已写入 %s", output_path)
