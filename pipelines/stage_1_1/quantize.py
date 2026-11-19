"""Stage 1.1 —— 量化入口：对指定 (algorithm, scheme, position) 执行一次 PTQ。

可通过 --run-id 直接引用实验矩阵，或显式传入三元组。
在 Pod 内默认从 OSS 挂载路径读取模型/数据集，产物写回 OSS 输出目录。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from quantflow.config import (
    load_experiment_matrix,
    load_project_config,
)
from quantflow.data import load_ultrachat
from quantflow.logging_utils import get_logger
from quantflow.recipes import build_recipe

logger = get_logger("stage_1_1.quantize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTQ 量化单个方案")
    parser.add_argument("--run-id", type=str, help="实验矩阵中的 run_id")
    parser.add_argument("--algorithm", choices=["smoothquant", "awq"])
    parser.add_argument("--scheme", type=str)
    parser.add_argument("--position", choices=["all", "attention", "mlp"])
    parser.add_argument("--model", type=str, help="模型目录，默认取 Pod 内 OSS 路径")
    parser.add_argument("--dataset", type=str, help="数据集目录")
    parser.add_argument("--output", type=str, help="输出根目录")
    parser.add_argument("--num-calibration", type=int)
    parser.add_argument("--max-seq-length", type=int)
    return parser.parse_args()


def resolve_spec(args: argparse.Namespace):
    matrix = load_experiment_matrix()
    if args.run_id:
        run = matrix.get_run(args.run_id)
        return run.algorithm, run.scheme, run.position, args.run_id
    if not (args.algorithm and args.scheme and args.position):
        raise SystemExit("必须提供 --run-id 或完整的 --algorithm/--scheme/--position")
    run_id = f"{args.algorithm}__{args.scheme.lower()}__{args.position}"
    return args.algorithm, args.scheme, args.position, run_id


def main() -> None:
    args = parse_args()
    project = load_project_config()
    defaults = project.quant_defaults
    algorithm, scheme, position, run_id = resolve_spec(args)

    model_path = args.model or project.in_pod_model
    dataset_path = args.dataset or project.in_pod_dataset
    output_root = Path(args.output or project.in_pod_output)
    num_calib = args.num_calibration or defaults["num_calibration_samples"]
    max_seq_len = args.max_seq_length or defaults["max_seq_length"]

    save_dir = output_root / "models" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("run_id=%s model=%s", run_id, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    recipe = build_recipe(
        algorithm=algorithm,
        scheme=scheme,
        position=position,
        smoothquant_strength=defaults["smoothquant_strength"],
        awq_duo_scaling=defaults["awq_duo_scaling"],
    )

    calib_split = project.dataset_usage["calibration"]["split"]
    calib_dataset = load_ultrachat(
        dataset_path=dataset_path,
        split=calib_split,
        tokenizer=tokenizer,
        num_samples=num_calib,
        max_seq_length=max_seq_len,
    )

    start = time.time()
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_calib,
    )
    elapsed = time.time() - start

    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    manifest = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scheme": scheme,
        "position": position,
        "model_path": str(save_dir),
        "quantization_seconds": round(elapsed, 2),
        "num_calibration_samples": num_calib,
    }
    (save_dir / "quant_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("量化完成，产物目录=%s 耗时=%.1fs", save_dir, elapsed)


if __name__ == "__main__":
    main()
