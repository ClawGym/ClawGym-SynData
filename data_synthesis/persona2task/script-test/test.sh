#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DISTILL_API_KEY_VALUE="${DISTILL_API_KEY_VALUE:-YOUR_DISTILL_API_KEY}" \
DISTILL_API_BASE_VALUE="${DISTILL_API_BASE_VALUE:-https://example.invalid/v1}" \
bash script-test/since21100.sh
