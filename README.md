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

It expects a patched Gemini CLI runtime at `/mnt/data/projects/deep-swe/third_party/gemini_cli_runtime`. It reads API keys from `GEMINI_API_KEY`, `GEMINI_API_KEYS`, or the default key file `/mnt/data/projects/verl-vllm-secrets/gemini_api_keys.txt`:

```bash
GEMINI_API_KEYS="key1,key2" scripts/run_gemini_candidate.py \
  --model gemini-3.1-pro-preview \
  --run-name gemini31pro-preview-candidate
```

Run artifacts are written outside this repository under `/mnt/data/projects/verl-vllm-gemini-runs/`.

## Gemini Test Generator Runner

`scripts/run_gemini_test_generator.py` runs Gemini CLI to generate evaluator tests from exactly three inputs:

- `task/verl` as `/workspace/project`
- `task/PROMPT.md` as `/workspace/AGENT_PROMPT.md`
- the official reference checkout as `/workspace/reference_project`

It writes generated tests under `/workspace/generated_tests`, validates them against both reference and incomplete projects in the CPU test image, deletes Gemini CLI state, and records static audits. The audit checks hard structural patterns and test imports or string patches that resolve only through `/workspace/reference_project`.

```bash
GEMINI_API_KEYS="key1,key2" scripts/run_gemini_test_generator.py \
  --model gemini-3.5-flash \
  --run-name gemini35flash-testgen \
  --num-runs 4 \
  --repair-rounds 1 \
  --max-candidate-patches 12 \
  --score-manual-suite
```

Run artifacts are written under `/mnt/data/projects/verl-vllm-testgen-runs/`. Each run writes
`score_table.md` with generated-test counts and pass counts for baseline, reference, discovered
candidate patches, and the hand-written evaluator suite when `--score-manual-suite` is enabled.

To score an already generated evaluator without invoking Gemini:

```bash
scripts/run_gemini_test_generator.py \
  --run-name score-existing \
  --existing-generated-tests /path/to/generated_tests \
  --max-candidate-patches -1 \
  --score-manual-suite
```

If `--repair-rounds` is set, the runner converts reference/baseline/candidate scores and static audit findings into
`/workspace/EVALUATOR_FEEDBACK.md`, then asks Gemini to revise `/workspace/generated_tests` in place. This makes the
generation loop a validate-and-repair pipeline instead of a single-shot prompt.

The generic workflow is documented in `docs/test-generator-pipeline.md`.
