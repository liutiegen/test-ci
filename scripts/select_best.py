"""选优入口：汇总 baseline 与 9 个量化方案的指标，产出 best_selection.json。

合并每个 run 的 lm-eval 主指标与独立困惑度文件，按综合保留率门槛筛选，
并写出 Markdown 对比报告。若无方案达标，按 Rule 抛错（流水线应暂停人工介入）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantflow.config import load_experiment_matrix, load_project_config
from quantflow.logging_utils import get_logger
from quantflow.metrics import (
    RunMetrics,
    flatten_lm_eval,
    rank_runs,
    select_best,
    write_markdown_report,
)

logger = get_logger("scripts.select_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量化方案选优")
    parser.add_argument("--output", type=str, help="输出根目录（默认读 project 配置）")
    parser.add_argument("--retention-floor", type=float, default=0.98)
    return parser.parse_args()


def load_merged_metrics(metrics_dir: Path, run_id: str) -> RunMetrics:
    """加载主指标文件，并把独立的困惑度文件合并进来。"""
    main_file = metrics_dir / f"{run_id}.json"
    with main_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    pp_file = metrics_dir / f"{run_id}_perplexity.json"
    if pp_file.exists():
        with pp_file.open("r", encoding="utf-8") as handle:
            raw["perplexity"] = json.load(handle)["perplexity"]

    return RunMetrics(
        run_id=raw["run_id"],
        task_scores=flatten_lm_eval(raw.get("results", {})),
        perplexity=raw.get("perplexity"),
        model_size_bytes=raw.get("model_size_bytes"),
        raw=raw,
    )


def main() -> None:
    args = parse_args()
    project = load_project_config()
    matrix = load_experiment_matrix()

    output_root = Path(args.output or project.in_pod_output)
    metrics_dir = output_root / "metrics"

    baseline = load_merged_metrics(metrics_dir, matrix.baseline_run_id)
    candidates = [load_merged_metrics(metrics_dir, run_id) for run_id in matrix.run_ids()]

    rankings = rank_runs(baseline, candidates, retention_floor=args.retention_floor)
    best = select_best(rankings)
    if best is None:
        raise RuntimeError(
            f"没有方案达到保留率门槛 {args.retention_floor}，按 Rule 暂停，需人工修正后重跑"
        )

    report_path = output_root / "quantization_report.md"
    write_markdown_report(rankings, report_path, best.run_id)

    selection = {
        "best_run_id": best.run_id,
        "retention": best.retention,
        "perplexity": best.perplexity,
        "model_size_bytes": best.model_size_bytes,
        "retention_floor": args.retention_floor,
        "finetuned_run_id": f"{best.run_id}-finetuned",
    }
    selection_file = output_root / "best_selection.json"
    selection_file.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("最优方案=%s，选择文件=%s", best.run_id, selection_file)


if __name__ == "__main__":
    main()
