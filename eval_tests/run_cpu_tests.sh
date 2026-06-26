#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/verl-checkout" >&2
  exit 2
fi

SOURCE_DIR="$(cd "$1" && pwd)"
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IMAGE="${TEST_IMAGE:-verl-vllm-rollout-benchmark:cpu-tests}"
WORK_DIR="${EVAL_WORK_DIR:-$(mktemp -d /tmp/verl-vllm-cpu-eval.XXXXXX)}"

cleanup() {
  if [[ -z "${EVAL_WORK_DIR:-}" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$WORK_DIR/verl"
(cd "$SOURCE_DIR" && tar --exclude='./.git' --exclude='./__pycache__' -cf - .) | (cd "$WORK_DIR/verl" && tar -xf -)

mkdir -p "$WORK_DIR/verl/tests/eval_vllm"
cp "$EVAL_ROOT/tests/"*.py "$WORK_DIR/verl/tests/eval_vllm/"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER=eval \
  -e LOGNAME=eval \
  -e RAY_TMPDIR=/tmp/ray \
  -v "$WORK_DIR/verl:/workspace/verl" \
  -w /workspace/verl \
  "$TEST_IMAGE" \
  bash -lc 'python -m pytest -q tests/eval_vllm'
