#!/usr/bin/env bash
# Stage 1.2 Pod 入口：单节点微调最优量化模型。
# BEST_RUN_ID 通过环境变量注入；未提供时 finetune.py 读取 best_selection.json。
set -euo pipefail

ARGS=()
if [ -n "${BEST_RUN_ID:-}" ]; then
  ARGS+=(--best-run-id "${BEST_RUN_ID}")
fi
if [ -n "${MAX_STEPS:-}" ]; then
  ARGS+=(--max-steps "${MAX_STEPS}")
fi

echo "===== 开始微调 ${BEST_RUN_ID:-（读取选择文件）} ====="
python pipelines/stage_1_2/finetune.py "${ARGS[@]}"
echo "===== 微调结束 ====="
