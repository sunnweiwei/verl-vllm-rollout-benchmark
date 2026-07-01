#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT_ID="${1:-${AGENT_ID:-4}}"
MODEL="${MODEL:-gemini-3.5-flash}"
GEMINI_THINKING_LEVEL="${GEMINI_THINKING_LEVEL:-HIGH}"
KEY_INDEX="${KEY_INDEX:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
GENERATION_RETRIES="${GENERATION_RETRIES:-1}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data/projects/verl-vllm-testgen-runs}"
RUN_NAME="${RUN_NAME:-evalgen-agent-${AGENT_ID}-${MODEL}}"

PROJECT_SOURCE="${PROJECT_SOURCE:-/mnt/data/projects/verl-vllm-capability-benchmark/task/verl}"
AGENT_PROMPT="${AGENT_PROMPT:-/mnt/data/projects/verl-vllm-capability-benchmark/task/PROMPT.md}"
REFERENCE_PROJECT="${REFERENCE_PROJECT:-/mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0}"

GENERATOR_ARGS=()
case "${AGENT_ID}" in
  1|v1|agent-1)
    GENERATION_MODE="single"
    GENERATOR_ARGS=(--generator-prompt "${ROOT_DIR}/docs/prompts/unit-test-generator-prompt-v1.md")
    ;;
  2|v2|agent-2)
    GENERATION_MODE="single"
    GENERATOR_ARGS=(--generator-prompt "${ROOT_DIR}/docs/prompts/unit-test-generator-prompt-v2.md")
    ;;
  3|v3|agent-3)
    GENERATION_MODE="single"
    GENERATOR_ARGS=(--generator-prompt "${ROOT_DIR}/docs/prompts/unit-test-generator-prompt-v3.md")
    ;;
  4|v4|agent-4|two-stage-v4)
    GENERATION_MODE="two-stage"
    GENERATOR_ARGS=(
      --generator-prompt-stage1 "${ROOT_DIR}/docs/prompts/unit-test-generator-prompt-v4-stage1.md"
      --generator-prompt-stage2 "${ROOT_DIR}/docs/prompts/unit-test-generator-prompt-v4-stage2.md"
    )
    ;;
  *)
    echo "unknown AGENT_ID: ${AGENT_ID}" >&2
    echo "known agent ids: 1, 2, 3, 4" >&2
    exit 2
    ;;
esac

echo "running evaluator-generation agent"
echo "  agent_id: ${AGENT_ID}"
echo "  generation_mode: ${GENERATION_MODE}"
echo "  model: ${MODEL}"
echo "  run_name: ${RUN_NAME}"
echo "  project_source: ${PROJECT_SOURCE}"
echo "  reference_project: ${REFERENCE_PROJECT}"
echo "  output: generated_tests under the created run workspace"

python "${ROOT_DIR}/scripts/run_gemini_test_generator.py" \
  --generation-mode "${GENERATION_MODE}" \
  --run-name "${RUN_NAME}" \
  --runs-root "${RUNS_ROOT}" \
  --model "${MODEL}" \
  --gemini-thinking-level "${GEMINI_THINKING_LEVEL}" \
  --key-index "${KEY_INDEX}" \
  --project-source "${PROJECT_SOURCE}" \
  --agent-prompt "${AGENT_PROMPT}" \
  --reference-project "${REFERENCE_PROJECT}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --generation-retries "${GENERATION_RETRIES}" \
  --skip-validation \
  "${GENERATOR_ARGS[@]}"
