"""Stage 1.2 —— 对 stage1.1 选出的最优量化模型做量化感知微调（单节点）。

流程：读取最优 run_id -> 加载 compressed-tensors 量化模型 -> 构建与方案一致的
QAT 修饰器 -> TRL SFTTrainer 加载 ultrachat train_sft -> 桥接 llmcompressor
会话执行训练 -> 保存最终微调模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantflow.config import load_experiment_matrix, load_project_config
from quantflow.data import load_raw_ultrachat
from quantflow.logging_utils import get_logger
from quantflow.qat_bridge import QATCallback, build_qat_modifier
from trl import SFTConfig, SFTTrainer

logger = get_logger("stage_1_2.finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量化模型 QAT 微调")
    parser.add_argument("--best-run-id", type=str, help="stage1.1 选出的最优 run_id")
    parser.add_argument("--selection-file", type=str, help="select_best 产出的 JSON")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=20000)
    return parser.parse_args()


def resolve_best_run_id(args: argparse.Namespace) -> str:
    if args.best_run_id:
        return args.best_run_id
    project = load_project_config()
    selection_file = args.selection_file or str(
        Path(project.in_pod_output) / "best_selection.json"
    )
    with open(selection_file, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    return selection["best_run_id"]


def main() -> None:
    args = parse_args()
    project = load_project_config()
    matrix = load_experiment_matrix()
    best_run_id = resolve_best_run_id(args)
    run = matrix.get_run(best_run_id)
    logger.info("微调最优方案 run_id=%s scheme=%s position=%s",
                best_run_id, run.scheme, run.position)

    quant_model_dir = Path(project.in_pod_output) / "models" / best_run_id
    output_root = Path(args.output or project.in_pod_output)
    final_dir = output_root / "models" / f"{best_run_id}-finetuned"
    final_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(quant_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        quant_model_dir, torch_dtype=torch.bfloat16
    )

    # QAT 修饰器：与已选方案保持一致的伪量化
    qat_modifier = build_qat_modifier(scheme=run.scheme, position=run.position)

    dataset_path = args.dataset or project.in_pod_dataset
    train_split = project.dataset_usage["training"]["split"]
    train_dataset = load_raw_ultrachat(
        dataset_path=dataset_path,
        split=train_split,
        num_samples=args.train_samples,
    )

    sft_config = SFTConfig(
        output_dir=str(final_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_length=args.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 先有 trainer，再挂 QAT 回调（回调内部需要 trainer.optimizer）
    qat_callback = QATCallback(trainer=trainer, recipe=[qat_modifier])
    trainer.add_callback(qat_callback)

    trainer.train()

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    manifest = {
        "source_run_id": best_run_id,
        "final_run_id": f"{best_run_id}-finetuned",
        "scheme": run.scheme,
        "position": run.position,
        "max_steps": args.max_steps,
        "model_path": str(final_dir),
    }
    (final_dir / "finetune_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("微调完成，最终模型目录=%s", final_dir)


if __name__ == "__main__":
    main()
