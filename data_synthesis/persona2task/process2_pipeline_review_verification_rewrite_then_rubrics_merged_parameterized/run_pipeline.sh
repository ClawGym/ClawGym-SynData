#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PERSONA_FILE="${SCRIPT_DIR}/seeds/persona.jsonl"
CATEGORY_FILE="${SCRIPT_DIR}/seeds/category2.json"
ACTION_FILE="${SCRIPT_DIR}/seeds/action2-remove.json"
WORK_DIR=""

NUM_PROMPTS=10
BASIC_OPERATION_COUNT=3
PERSONA_START_INDEX=21100
QUESTION_LANGUAGE="English"
FROM_STAGE=1
TO_STAGE=12
SKIP_STAGES="3"
GLOBAL_POOL_SIZE=16
SEED=42

MODEL_MODE="distill_openai"
MODEL=""
MODEL_ID=""
API_KEY=""
API_BASE=""
DISTILL_API_KEY=""
DISTILL_API_BASE=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash run_pipeline.sh --model gpt-5 --distill-api-key KEY --distill-api-base URL [options]

Required for --model-mode distill_openai:
  --model NAME
  --distill-api-key KEY
  --distill-api-base URL

Required for --model-mode openai_compatible:
  --api-base URL
  Optional: --api-key KEY --model-id MODEL_ID

Seed file options, defaulting to local files under ./seeds:
  --persona-file PATH
  --category-file PATH
  --action-file PATH

Pipeline options:
  --work-dir PATH
  --num-prompts N
  --basic-operation-count N
  --persona-start-index N
  --question-language TEXT
  --from-stage STAGE
  --to-stage STAGE
  --skip-stages CSV
  --global-pool-size N
  --seed N
  --dry-run

Pass additional run_merged_pipeline.py args after --.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persona-file) PERSONA_FILE="$2"; shift 2 ;;
    --category-file) CATEGORY_FILE="$2"; shift 2 ;;
    --action-file) ACTION_FILE="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --basic-operation-count) BASIC_OPERATION_COUNT="$2"; shift 2 ;;
    --persona-start-index) PERSONA_START_INDEX="$2"; shift 2 ;;
    --question-language) QUESTION_LANGUAGE="$2"; shift 2 ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --to-stage) TO_STAGE="$2"; shift 2 ;;
    --skip-stages) SKIP_STAGES="$2"; shift 2 ;;
    --global-pool-size) GLOBAL_POOL_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model-mode) MODEL_MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --distill-api-key) DISTILL_API_KEY="$2"; shift 2 ;;
    --distill-api-base) DISTILL_API_BASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required_file in "${PERSONA_FILE}" "${CATEGORY_FILE}" "${ACTION_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing seed file: ${required_file}" >&2
    exit 1
  fi
done

if [[ "${MODEL_MODE}" == "distill_openai" ]]; then
  if [[ -z "${MODEL}" || -z "${DISTILL_API_KEY}" || -z "${DISTILL_API_BASE}" ]]; then
    echo "distill_openai requires --model, --distill-api-key, and --distill-api-base." >&2
    usage >&2
    exit 2
  fi
elif [[ "${MODEL_MODE}" == "openai_compatible" ]]; then
  if [[ -z "${API_BASE}" ]]; then
    echo "openai_compatible requires --api-base." >&2
    usage >&2
    exit 2
  fi
else
  echo "Unsupported --model-mode: ${MODEL_MODE}" >&2
  exit 2
fi

if [[ -z "${WORK_DIR}" ]]; then
  PERSONA_END_INDEX=$((PERSONA_START_INDEX + NUM_PROMPTS))
  WORK_DIR="${SCRIPT_DIR}/runs/skip-${SKIP_STAGES//,/+}-origin-persona-${PERSONA_START_INDEX}-${PERSONA_END_INDEX}-action${BASIC_OPERATION_COUNT}"
fi

PIPELINE_ARGS=(
  --work-dir "${WORK_DIR}"
  --num-prompts "${NUM_PROMPTS}"
  --basic-operation-count "${BASIC_OPERATION_COUNT}"
  --persona-start-index "${PERSONA_START_INDEX}"
  --question-language "${QUESTION_LANGUAGE}"
  --from-stage "${FROM_STAGE}"
  --to-stage "${TO_STAGE}"
  --skip-stages "${SKIP_STAGES}"
  --global-pool-size "${GLOBAL_POOL_SIZE}"
  --seed "${SEED}"
  --persona-file "${PERSONA_FILE}"
  --category-file "${CATEGORY_FILE}"
  --action-file "${ACTION_FILE}"
)

add_stage_model_args() {
  local prefix="$1"
  PIPELINE_ARGS+=("--${prefix}-model-mode" "${MODEL_MODE}")
  if [[ "${MODEL_MODE}" == "distill_openai" ]]; then
    PIPELINE_ARGS+=(
      "--${prefix}-model" "${MODEL}"
      "--${prefix}-distill-api-key" "${DISTILL_API_KEY}"
      "--${prefix}-distill-api-base" "${DISTILL_API_BASE}"
    )
  else
    if [[ -n "${API_KEY}" ]]; then
      PIPELINE_ARGS+=("--${prefix}-api-key" "${API_KEY}")
    fi
    PIPELINE_ARGS+=("--${prefix}-api-base" "${API_BASE}")
    if [[ -n "${MODEL_ID}" ]]; then
      PIPELINE_ARGS+=("--${prefix}-model-id" "${MODEL_ID}")
    fi
  fi
}

add_stage_model_args task
add_stage_model_args judge
add_stage_model_args iter
add_stage_model_args rubric

if [[ "${DRY_RUN}" == "1" ]]; then
  PIPELINE_ARGS+=(--dry-run)
fi

python3 "${SCRIPT_DIR}/run_merged_pipeline.py" "${PIPELINE_ARGS[@]}" "${EXTRA_ARGS[@]}"
