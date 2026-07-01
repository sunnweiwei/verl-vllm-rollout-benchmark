#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/verl-checkout" >&2
  exit 2
fi

SOURCE_DIR="$(cd "$1" && pwd)"
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_IMAGE="${TEST_IMAGE:-verl-vllm-rollout-benchmark:cpu-tests}"
WORK_DIR="${EVAL_WORK_DIR:-$(mktemp -d /tmp/verl-vllm-gpu-eval.XXXXXX)}"

cleanup() {
  if [[ -z "${EVAL_WORK_DIR:-}" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$WORK_DIR/verl"
(cd "$SOURCE_DIR" && tar --exclude='./.git' --exclude='./__pycache__' -cf - .) | (cd "$WORK_DIR/verl" && tar -xf -)

mkdir -p "$WORK_DIR/verl/tests/eval_vllm_gpu"
cp "$EVAL_ROOT/gpu_tests/"*.py "$WORK_DIR/verl/tests/eval_vllm_gpu/"

if [[ -n "${VERL_GPU_TEST_MODEL:-}" ]]; then
  MODEL_DIR="$(cd "$VERL_GPU_TEST_MODEL" && pwd)"
else
  MODEL_DIR="$WORK_DIR/tiny-random-qwen2"
  mkdir -p "$MODEL_DIR"
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER=eval \
    -e LOGNAME=eval \
    -v "$EVAL_ROOT/make_tiny_gpu_model.py:/workspace/make_tiny_gpu_model.py:ro" \
    -v "$MODEL_DIR:/workspace/model" \
    "$TEST_IMAGE" \
    python /workspace/make_tiny_gpu_model.py /workspace/model
fi

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size="${GPU_TEST_SHM_SIZE:-16g}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER=eval \
  -e LOGNAME=eval \
  -e RAY_TMPDIR=/tmp/ray \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e VERL_GPU_TEST_MODEL=/workspace/model \
  -e VERL_GPU_TEST_GPUS="${VERL_GPU_TEST_GPUS:-1}" \
  -e VERL_GPU_TEST_TP="${VERL_GPU_TEST_TP:-1}" \
  -e VERL_GPU_TEST_PROMPT_LEN="${VERL_GPU_TEST_PROMPT_LEN:-24}" \
  -e VERL_GPU_TEST_RESPONSE_LEN="${VERL_GPU_TEST_RESPONSE_LEN:-4}" \
  -e VERL_GPU_TEST_MAX_MODEL_LEN="${VERL_GPU_TEST_MAX_MODEL_LEN:-40}" \
  -e VERL_GPU_TEST_MAX_NUM_SEQS="${VERL_GPU_TEST_MAX_NUM_SEQS:-2}" \
  -e VERL_GPU_TEST_GENERATE_TOKENS="${VERL_GPU_TEST_GENERATE_TOKENS:-2}" \
  -e VERL_GPU_TEST_DTYPE="${VERL_GPU_TEST_DTYPE:-float16}" \
  -e VERL_GPU_TEST_GPU_MEMORY_UTILIZATION="${VERL_GPU_TEST_GPU_MEMORY_UTILIZATION:-0.05}" \
  -e VERL_GPU_TEST_ENFORCE_EAGER="${VERL_GPU_TEST_ENFORCE_EAGER:-1}" \
  -e VERL_GPU_TEST_TRUST_REMOTE_CODE="${VERL_GPU_TEST_TRUST_REMOTE_CODE:-0}" \
  -e VERL_GPU_PYTEST_ARGS="${VERL_GPU_PYTEST_ARGS:-tests/eval_vllm_gpu}" \
  -v "$WORK_DIR/verl:/workspace/verl" \
  -v "$MODEL_DIR:/workspace/model:ro" \
  -w /workspace/verl \
  "$TEST_IMAGE" \
  bash -lc 'python -m pytest -q -o cache_dir=/tmp/pytest_cache ${VERL_GPU_PYTEST_ARGS}'
