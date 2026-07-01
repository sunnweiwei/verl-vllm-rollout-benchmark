#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENERATED_TESTS="${1:-${GENERATED_TESTS:-}}"
if [[ -z "${GENERATED_TESTS}" ]]; then
  echo "usage: ./coverage.sh /path/to/generated_tests [label=/path/to/reference ...]" >&2
  echo "or: GENERATED_TESTS=/path/to/generated_tests ./coverage.sh" >&2
  exit 2
fi
shift || true
if [[ ! -d "${GENERATED_TESTS}" ]]; then
  echo "generated tests directory does not exist: ${GENERATED_TESTS}" >&2
  exit 2
fi

RUNS_ROOT="${RUNS_ROOT:-/mnt/data/projects/verl-vllm-testgen-runs}"
RUN_NAME="${RUN_NAME:-coverage-generated-tests}"
PROJECT_SOURCE="${PROJECT_SOURCE:-/mnt/data/projects/verl-vllm-capability-benchmark/task/verl}"
TEST_IMAGE="${TEST_IMAGE:-verl-vllm-rollout-benchmark:cpu-tests}"
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-300}"
NO_DEFAULT_REFERENCES="${NO_DEFAULT_REFERENCES:-0}"
COVERAGE_INCLUDE_REGEX="${COVERAGE_INCLUDE_REGEX:-}"
COVERAGE_EXCLUDE_REGEX="${COVERAGE_EXCLUDE_REGEX:-}"
PYTEST_TARGETS="${PYTEST_TARGETS:-}"
COVERAGE_DOCKER_GPUS="${COVERAGE_DOCKER_GPUS:-}"
COVERAGE_DOCKER_IPC="${COVERAGE_DOCKER_IPC:-}"
COVERAGE_DOCKER_SHM_SIZE="${COVERAGE_DOCKER_SHM_SIZE:-}"
COVERAGE_DOCKER_ENV="${COVERAGE_DOCKER_ENV:-}"
COVERAGE_DOCKER_VOLUME="${COVERAGE_DOCKER_VOLUME:-}"

EXTRA_ARGS=()
case "${NO_DEFAULT_REFERENCES}" in
  1|true|TRUE|yes|YES)
    EXTRA_ARGS+=(--no-default-references)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "unknown NO_DEFAULT_REFERENCES value: ${NO_DEFAULT_REFERENCES}" >&2
    exit 2
    ;;
esac

for reference_spec in "$@"; do
  EXTRA_ARGS+=(--reference "${reference_spec}")
done

if [[ -n "${COVERAGE_INCLUDE_REGEX}" ]]; then
  IFS=',' read -r -a include_regexes <<< "${COVERAGE_INCLUDE_REGEX}"
  for include_regex in "${include_regexes[@]}"; do
    [[ -n "${include_regex}" ]] && EXTRA_ARGS+=(--include-path-regex "${include_regex}")
  done
fi

if [[ -n "${COVERAGE_EXCLUDE_REGEX}" ]]; then
  IFS=',' read -r -a exclude_regexes <<< "${COVERAGE_EXCLUDE_REGEX}"
  for exclude_regex in "${exclude_regexes[@]}"; do
    [[ -n "${exclude_regex}" ]] && EXTRA_ARGS+=(--exclude-path-regex "${exclude_regex}")
  done
fi

if [[ -n "${PYTEST_TARGETS}" ]]; then
  IFS=',' read -r -a pytest_targets <<< "${PYTEST_TARGETS}"
  for pytest_target in "${pytest_targets[@]}"; do
    [[ -n "${pytest_target}" ]] && EXTRA_ARGS+=(--pytest-target "${pytest_target}")
  done
fi

if [[ -n "${COVERAGE_DOCKER_GPUS}" ]]; then
  EXTRA_ARGS+=(--docker-gpus "${COVERAGE_DOCKER_GPUS}")
fi
if [[ -n "${COVERAGE_DOCKER_IPC}" ]]; then
  EXTRA_ARGS+=(--docker-ipc "${COVERAGE_DOCKER_IPC}")
fi
if [[ -n "${COVERAGE_DOCKER_SHM_SIZE}" ]]; then
  EXTRA_ARGS+=(--docker-shm-size "${COVERAGE_DOCKER_SHM_SIZE}")
fi

if [[ -n "${COVERAGE_DOCKER_ENV}" ]]; then
  IFS=',' read -r -a docker_envs <<< "${COVERAGE_DOCKER_ENV}"
  for docker_env in "${docker_envs[@]}"; do
    [[ -n "${docker_env}" ]] && EXTRA_ARGS+=(--docker-env "${docker_env}")
  done
fi

if [[ -n "${COVERAGE_DOCKER_VOLUME}" ]]; then
  IFS=',' read -r -a docker_volumes <<< "${COVERAGE_DOCKER_VOLUME}"
  for docker_volume in "${docker_volumes[@]}"; do
    [[ -n "${docker_volume}" ]] && EXTRA_ARGS+=(--docker-volume "${docker_volume}")
  done
fi

echo "diagnosing generated evaluator coverage"
echo "  generated_tests: ${GENERATED_TESTS}"
echo "  run_name: ${RUN_NAME}"
echo "  project_source: ${PROJECT_SOURCE}"
echo "  image: ${TEST_IMAGE}"
if [[ -n "${COVERAGE_INCLUDE_REGEX}" ]]; then
  echo "  include_regex: ${COVERAGE_INCLUDE_REGEX}"
fi
if [[ -n "${COVERAGE_EXCLUDE_REGEX}" ]]; then
  echo "  exclude_regex: ${COVERAGE_EXCLUDE_REGEX}"
fi
if [[ -n "${PYTEST_TARGETS}" ]]; then
  echo "  pytest_targets: ${PYTEST_TARGETS}"
fi
if [[ -n "${COVERAGE_DOCKER_GPUS}" ]]; then
  echo "  docker_gpus: ${COVERAGE_DOCKER_GPUS}"
fi
if [[ -n "${COVERAGE_DOCKER_IPC}" ]]; then
  echo "  docker_ipc: ${COVERAGE_DOCKER_IPC}"
fi
if [[ -n "${COVERAGE_DOCKER_SHM_SIZE}" ]]; then
  echo "  docker_shm_size: ${COVERAGE_DOCKER_SHM_SIZE}"
fi

python "${ROOT_DIR}/scripts/diagnose_reference_coverage.py" \
  --generated-tests "${GENERATED_TESTS}" \
  --run-name "${RUN_NAME}" \
  --runs-root "${RUNS_ROOT}" \
  --project-source "${PROJECT_SOURCE}" \
  --image "${TEST_IMAGE}" \
  --test-timeout-sec "${TEST_TIMEOUT_SEC}" \
  "${EXTRA_ARGS[@]}"
