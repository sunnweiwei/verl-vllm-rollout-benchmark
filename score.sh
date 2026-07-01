#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENERATED_TESTS="${1:-${GENERATED_TESTS:-}}"
if [[ -z "${GENERATED_TESTS}" ]]; then
  echo "usage: ./score.sh /path/to/generated_tests" >&2
  echo "or: GENERATED_TESTS=/path/to/generated_tests ./score.sh" >&2
  exit 2
fi
if [[ ! -d "${GENERATED_TESTS}" ]]; then
  echo "generated tests directory does not exist: ${GENERATED_TESTS}" >&2
  exit 2
fi

RUNS_ROOT="${RUNS_ROOT:-/mnt/data/projects/verl-vllm-testgen-runs}"
RUN_NAME="${RUN_NAME:-score-generated-tests}"
PROJECT_SOURCE="${PROJECT_SOURCE:-/mnt/data/projects/verl-vllm-capability-benchmark/task/verl}"
AGENT_PROMPT="${AGENT_PROMPT:-/mnt/data/projects/verl-vllm-capability-benchmark/task/PROMPT.md}"
REFERENCE_PROJECT="${REFERENCE_PROJECT:-/mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0}"
COUNTER_REFERENCE_PROJECT="${COUNTER_REFERENCE_PROJECT:-/mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project}"
CANDIDATE_RUNS_ROOT="${CANDIDATE_RUNS_ROOT:-/mnt/data/projects/verl-vllm-gemini-runs}"
MAX_CANDIDATE_PATCHES="${MAX_CANDIDATE_PATCHES:--1}"
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-240}"
SCORE_MANUAL_SUITE="${SCORE_MANUAL_SUITE:-0}"

EXTRA_ARGS=()
case "${SCORE_MANUAL_SUITE}" in
  1|true|TRUE|yes|YES)
    EXTRA_ARGS+=(--score-manual-suite)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "unknown SCORE_MANUAL_SUITE value: ${SCORE_MANUAL_SUITE}" >&2
    exit 2
    ;;
esac

echo "scoring generated evaluator"
echo "  generated_tests: ${GENERATED_TESTS}"
echo "  run_name: ${RUN_NAME}"
echo "  project_source: ${PROJECT_SOURCE}"
echo "  reference_project: ${REFERENCE_PROJECT}"
echo "  counter_reference: ${COUNTER_REFERENCE_PROJECT}"
echo "  candidate_runs_root: ${CANDIDATE_RUNS_ROOT}"
echo "  max_candidate_patches: ${MAX_CANDIDATE_PATCHES}"

python "${ROOT_DIR}/scripts/run_gemini_test_generator.py" \
  --run-name "${RUN_NAME}" \
  --runs-root "${RUNS_ROOT}" \
  --project-source "${PROJECT_SOURCE}" \
  --agent-prompt "${AGENT_PROMPT}" \
  --reference-project "${REFERENCE_PROJECT}" \
  --existing-generated-tests "${GENERATED_TESTS}" \
  --candidate-runs-root "${CANDIDATE_RUNS_ROOT}" \
  --max-candidate-patches "${MAX_CANDIDATE_PATCHES}" \
  --no-default-validation-targets \
  --validation-target "counter_reference=${COUNTER_REFERENCE_PROJECT}" \
  --test-timeout-sec "${TEST_TIMEOUT_SEC}" \
  "${EXTRA_ARGS[@]}"
