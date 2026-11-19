"""ultrachat_200k 加载、对话模板预处理与分词。

对齐 llm-compressor 官方 examples/custom_dataset_example.py：
本地 parquet -> HF Dataset -> apply_chat_template -> tokenizer -> 可直接传入 oneshot。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from quantflow.logging_utils import get_logger

logger = get_logger(__name__)


def _parquet_glob(dataset_path: str, split: str) -> str:
    """ultrachat 按 split 命名 parquet，例如 data/train_sft-*.parquet。"""
    base = Path(dataset_path)
    # 兼容两种布局：parquet 直接在目录下，或位于 data/ 子目录
    candidates = [base / "data", base]
    for candidate in candidates:
        files = sorted(candidate.glob(f"{split}-*.parquet"))
        if files:
            return str(candidate / f"{split}-*.parquet")
    raise FileNotFoundError(f"未找到 split={split} 的 parquet：{dataset_path}")


def load_ultrachat(
    dataset_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    num_samples: int | None = None,
    max_seq_length: int = 2048,
    shuffle_seed: int = 42,
) -> Dataset:
    """加载并预处理 ultrachat，返回可用于 oneshot / SFT 的分词数据集。

    :param dataset_path: 本地或 Pod 内 ultrachat_200k 目录
    :param split: train_sft / test_sft / train_gen / test_gen
    :param tokenizer: 目标模型的 tokenizer（用于对话模板与分词）
    :param num_samples: 截取前 N 条（None 表示全量）
    :param max_seq_length: 分词最大长度
    """
    pattern = _parquet_glob(dataset_path, split)
    selector = f"[:{num_samples}]" if num_samples else ""
    logger.info("加载数据集 pattern=%s split=%s%s", pattern, split, selector)
    dataset = load_dataset("parquet", data_files=pattern, split=f"train{selector}")
    dataset = dataset.shuffle(seed=shuffle_seed)

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    dataset = dataset.map(to_text)

    def tokenize(sample: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    logger.info("数据集就绪，样本数=%d", len(dataset))
    return dataset


def load_raw_ultrachat(
    dataset_path: str,
    split: str,
    num_samples: int | None = None,
    shuffle_seed: int = 42,
) -> Dataset:
    """加载保留 messages 列的原始数据集，供 TRL SFTTrainer 自行处理对话格式。"""
    pattern = _parquet_glob(dataset_path, split)
    selector = f"[:{num_samples}]" if num_samples else ""
    dataset = load_dataset("parquet", data_files=pattern, split=f"train{selector}")
    return dataset.shuffle(seed=shuffle_seed)
