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
