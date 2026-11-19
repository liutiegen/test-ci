"""Stage 1.1 —— 评估入口：用 lm-evaluation-harness 衡量量化模型在 benchmark 上的表现。

通过 lm_eval 的 Python API simple_evaluate 执行，结果写为 JSON，
并附加 run_id 与模型体积，供后续选优汇总。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate

from quantflow.config import load_benchmark_config, load_project_config
from quantflow.logging_utils import get_logger

logger = get_logger("stage_1_1.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lm-eval 评估单个模型")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model", type=str, help="待评估模型目录")
    parser.add_argument("--suite", type=str, help="任务套件名 core/openllm")
    parser.add_argument("--tasks", type=str, help="逗号分隔的任务，优先级高于 --suite")
    parser.add_argument("--output", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=str, default="auto")
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def main() -> None:
    args = parse_args()
    project = load_project_config()
    benchmark = load_benchmark_config()

    # baseline 直接读原始模型；量化方案读 output/models/<run_id>
    if args.model:
        model_path = Path(args.model)
    elif args.run_id.startswith("baseline"):
        model_path = Path(project.in_pod_model)
    else:
        model_path = Path(project.in_pod_output) / "models" / args.run_id

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        suite = args.suite or benchmark.default_suite
        tasks = benchmark.tasks_for(suite)

    output_root = Path(args.output or project.in_pod_output)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("评估 run_id=%s model=%s tasks=%s", args.run_id, model_path, tasks)
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )

    record = {
        "run_id": args.run_id,
        "tasks": tasks,
        "results": results["results"],
        "model_size_bytes": directory_size_bytes(model_path),
        "group_subtasks": results.get("group_subtasks"),
    }
    out_file = metrics_dir / f"{args.run_id}.json"
    out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("指标已写入 %s", out_file)


if __name__ == "__main__":
    main()
