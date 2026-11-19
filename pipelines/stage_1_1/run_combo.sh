#!/usr/bin/env bash
# Stage 1.1 Pod 入口：对单个 run_id 执行 量化 -> benchmark 评估 -> 困惑度。
# 由 K8s Job 通过环境变量 RUN_ID 注入；LIMIT 可在冒烟时限制每个任务样本数。
set -euo pipefail

: "${RUN_ID:?必须通过环境变量提供 RUN_ID}"
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

echo "===== [1/3] 量化 ${RUN_ID} ====="
python pipelines/stage_1_1/quantize.py --run-id "${RUN_ID}"

echo "===== [2/3] Benchmark 评估 ${RUN_ID} ====="
python pipelines/stage_1_1/evaluate.py --run-id "${RUN_ID}" ${LIMIT_ARG}

echo "===== [3/3] 困惑度 ${RUN_ID} ====="
python pipelines/stage_1_1/perplexity.py --run-id "${RUN_ID}"

echo "===== 完成 ${RUN_ID} ====="
