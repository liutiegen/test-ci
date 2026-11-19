#!/usr/bin/env bash
# Stage 1.3 Pod 入口：按 MODEL_ROLE（baseline/compressed/finetuned）解析模型路径，
# 运行 vLLM 服务基准，并用 vllm 后端跑 lm-eval 准确率（core 套件）。
set -euo pipefail

: "${MODEL_ROLE:?必须通过环境变量提供 MODEL_ROLE}"

BASE_MODEL="/data/Qwen2.5-0.5B-Instruct"
OUT_DIR="/data/output"
BEST_FILE="${OUT_DIR}/best_selection.json"

BEST_RUN_ID="$(python -c "import json,sys;print(json.load(open('${BEST_FILE}'))['best_run_id'])")"

case "${MODEL_ROLE}" in
  baseline)
    MODEL_PATH="${BASE_MODEL}"
    LABEL="baseline"
    ;;
  compressed)
    MODEL_PATH="${OUT_DIR}/models/${BEST_RUN_ID}"
    LABEL="compressed"
    ;;
  finetuned)
    MODEL_PATH="${OUT_DIR}/models/${BEST_RUN_ID}-finetuned"
    LABEL="finetuned"
    ;;
  *)
    echo "未知 MODEL_ROLE=${MODEL_ROLE}" >&2
    exit 1
    ;;
esac

echo "===== [1/2] vLLM 服务基准 ${LABEL} (${MODEL_PATH}) ====="
python pipelines/stage_1_3/offline_benchmark.py \
  --model "${MODEL_PATH}" \
  --label "${LABEL}" \
  --output "${OUT_DIR}/serving/${LABEL}.json" \
  --num-prompts "${NUM_PROMPTS:-200}" \
  --input-len "${INPUT_LEN:-512}" \
  --output-len "${OUTPUT_LEN:-128}"

echo "===== [2/2] 准确率（lm-eval vllm 后端，core 套件）${LABEL} ====="
python pipelines/stage_1_1/evaluate.py \
  --run-id "stage13-${LABEL}" \
  --model "${MODEL_PATH}" \
  --suite core

echo "===== 完成 ${LABEL} ====="
