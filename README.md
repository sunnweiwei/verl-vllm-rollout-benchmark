# VERL vLLM Rollout Benchmark Task

This repository contains a benchmark task for adding vLLM rollout support to a sanitized VERL checkout.

## Contents

- `task/PROMPT.md`: the task prompt shown to the coding agent.
- `task/verl`: sanitized VERL source tree with vLLM support removed.
- `task/vllm-src`: local vLLM source tree for the agent to inspect or use.
- `docker/Dockerfile.eval`: Docker environment built from the official VERL runtime image.

## Build

```bash
docker build -f docker/Dockerfile.eval -t verl-vllm-rollout-benchmark:task-v0 .
```

## Run

```bash
docker run --rm -it verl-vllm-rollout-benchmark:task-v0
```

Inside the container:

- Prompt: `/workspace/PROMPT.md`
- VERL task repo: `/workspace/verl`
- vLLM source: `/workspace/vllm-src`

The image removes preinstalled vLLM package remnants from the base runtime and does not include git history for the task sources.

## Gemini Candidate Runner

`scripts/run_gemini_candidate.py` runs one Gemini CLI candidate in the task Docker, exports a patch, applies it to a fresh task checkout, and runs the CPU eval tests in the separate test Docker.

It expects a patched Gemini CLI runtime at `/mnt/data/projects/deep-swe/third_party/gemini_cli_runtime` and reads secrets only from runtime environment variables:

```bash
GEMINI_API_KEYS="key1,key2" scripts/run_gemini_candidate.py \
  --model gemini-3.1-pro-preview \
  --run-name gemini31pro-preview-candidate
```

Run artifacts are written outside this repository under `/mnt/data/projects/verl-vllm-gemini-runs/`.
