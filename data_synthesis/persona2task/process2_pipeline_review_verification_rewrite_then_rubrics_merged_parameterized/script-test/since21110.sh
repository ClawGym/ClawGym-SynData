#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_VALUE="${MODEL_VALUE:-gpt-5}"
DISTILL_API_KEY_VALUE="${DISTILL_API_KEY_VALUE:-}"
DISTILL_API_BASE_VALUE="${DISTILL_API_BASE_VALUE:-}"
PERSONA_FILE_VALUE="${PERSONA_FILE_VALUE:-${ROOT}/seeds/persona.jsonl}"
CATEGORY_FILE_VALUE="${CATEGORY_FILE_VALUE:-${ROOT}/seeds/category2.json}"
ACTION_FILE_VALUE="${ACTION_FILE_VALUE:-${ROOT}/seeds/action2-remove.json}"
WORK_DIR_VALUE="${WORK_DIR_VALUE:-${ROOT}/runs/skip-3-origin-persona-21110-23000-action3}"
DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

python3 "${ROOT}/run_merged_pipeline.py" \
  "${DRY_RUN_ARGS[@]}" \
  --work-dir "${WORK_DIR_VALUE}" \
  --num-prompts 1890 \
  --basic-operation-count 3 \
  --persona-start-index 21110 \
  --question-language English \
  --from-stage 1 \
  --to-stage 12 \
  --skip-stages 3 \
  --global-pool-size 16 \
  --persona-file "${PERSONA_FILE_VALUE}" \
  --category-file "${CATEGORY_FILE_VALUE}" \
  --action-file "${ACTION_FILE_VALUE}" \
  --task-model-mode distill_openai \
  --task-model "${MODEL_VALUE}" \
  --task-distill-api-key "${DISTILL_API_KEY_VALUE}" \
  --task-distill-api-base "${DISTILL_API_BASE_VALUE}" \
  --judge-model-mode distill_openai \
  --judge-model "${MODEL_VALUE}" \
  --judge-distill-api-key "${DISTILL_API_KEY_VALUE}" \
  --judge-distill-api-base "${DISTILL_API_BASE_VALUE}" \
  --iter-model-mode distill_openai \
  --iter-model "${MODEL_VALUE}" \
  --iter-distill-api-key "${DISTILL_API_KEY_VALUE}" \
  --iter-distill-api-base "${DISTILL_API_BASE_VALUE}" \
  --rubric-model-mode distill_openai \
  --rubric-model "${MODEL_VALUE}" \
  --rubric-distill-api-key "${DISTILL_API_KEY_VALUE}" \
  --rubric-distill-api-base "${DISTILL_API_BASE_VALUE}"

# In the old 0413 pipeline, --skip-stages 3,4,5,6 skipped middle rubrics
# plus dedup. In this merged pipeline, middle rubrics no longer exist, so
# --skip-stages 3 is the equivalent "skip dedup and continue from tasks" mode.
