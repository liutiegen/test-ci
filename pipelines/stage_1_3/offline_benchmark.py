"""Stage 1.3 —— vLLM 离线服务基准：在受控随机提示上对比 baseline / 压缩 / 微调模型。

固定输入长度与输出长度，测量端到端吞吐、请求延迟与首 token 延迟（TTFT），
结果写 JSON，保证三个模型在同一负载口径下可比较。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from quantflow.logging_utils import get_logger
from vllm import LLM, SamplingParams

logger = get_logger("stage_1_3.benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM 离线基准")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), q))


def build_prompts(num_prompts: int, input_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    # 避开前若干特殊 token，取普通词表区间
    lower, upper = 100, vocab_size - 1
    return [[rng.randint(lower, upper) for _ in range(input_len)] for _ in range(num_prompts)]


def main() -> None:
    args = parse_args()
    logger.info("vLLM 基准 label=%s model=%s", args.label, args.model)

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    warmup = build_prompts(8, args.input_len, vocab_size, args.seed)
    llm.generate(prompt_token_ids=warmup, sampling_params=sampling_params)

    prompts = build_prompts(args.num_prompts, args.input_len, vocab_size, args.seed + 1)
    start = time.perf_counter()
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    wall_time = time.perf_counter() - start

    total_input = sum(len(p) for p in prompts)
    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)

    ttft, decode_time, e2e = [], [], []
    for o in outputs:
        stats = getattr(o, "metrics", None)
        if stats is None:
            continue
        if stats.first_token_latency:
            ttft.append(float(stats.first_token_latency))
        if stats.first_token_ts and stats.last_token_ts:
            decode_time.append(float(stats.last_token_ts - stats.first_token_ts))
            e2e.append(float(stats.last_token_ts - stats.arrival_time))

    record = {
        "label": args.label,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "wall_time_s": wall_time,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput_rps": args.num_prompts / wall_time,
        "output_token_throughput_tps": total_output / wall_time,
        "total_token_throughput_tps": (total_input + total_output) / wall_time,
        "ttft_ms": {
            "p50": percentile(ttft, 50) * 1000,
            "p90": percentile(ttft, 90) * 1000,
            "p99": percentile(ttft, 99) * 1000,
        },
        "decode_time_s": {
            "p50": percentile(decode_time, 50),
            "p90": percentile(decode_time, 90),
            "p99": percentile(decode_time, 99),
        },
        "e2e_latency_s": {
            "p50": percentile(e2e, 50),
            "p90": percentile(e2e, 90),
            "p99": percentile(e2e, 99),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("基准结果写入 %s", out_path)


if __name__ == "__main__":
    main()
