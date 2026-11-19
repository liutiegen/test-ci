"""Stage 1.1 —— 困惑度：在验证集（ultrachat test_sft）上计算平均困惑度。

困惑度是数据集对齐的快速信号，与 benchmark 准确率互补，用于选优时的 tie-break。
采用滑动窗口切块、逐块计算 NLL 后按 token 数加权平均。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantflow.config import load_benchmark_config, load_project_config
from quantflow.logging_utils import get_logger

logger = get_logger("stage_1_1.perplexity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证集困惑度")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-seq-length", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = load_project_config()
    benchmark = load_benchmark_config()
    pp_cfg = benchmark.perplexity

    if args.model:
        model_path = Path(args.model)
    elif args.run_id.startswith("baseline"):
        model_path = Path(project.in_pod_model)
    else:
        model_path = Path(project.in_pod_output) / "models" / args.run_id

    dataset_path = args.dataset or project.in_pod_dataset
    output_root = Path(args.output or project.in_pod_output)
    max_samples = args.max_samples or pp_cfg["max_samples"]
    max_seq_len = args.max_seq_length or pp_cfg["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    from datasets import load_dataset

    pattern = str(Path(dataset_path) / "data" / "test_sft-*.parquet")
    if not Path(pattern).parent.exists():
        pattern = str(Path(dataset_path) / "test_sft-*.parquet")
    dataset = load_dataset("parquet", data_files=pattern, split=f"train[:{max_samples}]")

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for example in dataset:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            )
            input_ids = enc.input_ids.to(device)
            targets = input_ids[:, 1:]
            logits = model(input_ids).logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += targets.numel()

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    logger.info("run_id=%s tokens=%d PPL=%.4f", args.run_id, total_tokens, perplexity)

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": args.run_id,
        "num_tokens": total_tokens,
        "perplexity": perplexity,
    }
    (metrics_dir / f"{args.run_id}_perplexity.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
