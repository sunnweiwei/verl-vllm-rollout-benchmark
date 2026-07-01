# Evaluator-Generation Agents

Use `./run.sh <agent_id>` to run a registered evaluator-generation agent. This only generates pytest tests under the created run workspace; it does not score them. The agent id is the stable experiment handle; internal generation modes and prompt paths are implementation details behind that id.

## Registered Agents

| Agent id | Prompt files | Runner mode | Notes |
| --- | --- | --- | --- |
| `1` | `docs/prompts/unit-test-generator-prompt-v1.md` | `single` | Historical single-prompt evaluator generator. |
| `2` | `docs/prompts/unit-test-generator-prompt-v2.md` | `single` | Historical single-prompt evaluator generator. |
| `3` | `docs/prompts/unit-test-generator-prompt-v3.md` | `single` | Adds the alternative-implementation litmus test and final import/name self-check. |
| `4` | `docs/prompts/unit-test-generator-prompt-v4-stage1.md`, `docs/prompts/unit-test-generator-prompt-v4-stage2.md` | `two-stage` | Project-first Stage 1 plus reference-oracle Stage 2. Full prompts are sent directly on stdin. Stage 1 and Stage 2 run in one persistent Docker container with explicit Gemini session resume. |

## Common Usage

```bash
./run.sh 4
```

Score a generated evaluator separately:

```bash
./score.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

Run reference coverage diagnostics separately:

```bash
./coverage.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

For this VERL/vLLM task, use a task-focused path filter when you want the denominator to be only vLLM/rollout-related implementation diffs rather than the full release diff:

```bash
COVERAGE_INCLUDE_REGEX='^verl/workers/rollout/,^verl/workers/config/rollout.py$,^verl/third_party/vllm/,^verl/utils/vllm/,^verl/utils/modelopt/,^verl/utils/qat/,^verl/utils/profiler/' \
  ./coverage.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

Additional known-good references can be appended as `label=/path/to/reference`. The coverage route is diagnostic only; it does not affect `score.sh` pass/fail.

To diagnose one pytest file or node instead of the whole generated suite, set `PYTEST_TARGETS` to a comma-separated list of paths or node ids relative to `generated_tests`:

```bash
PYTEST_TARGETS='test_rollout_behavior.py::test_generation_returns_logprobs' \
  ./coverage.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

Useful environment overrides:

```bash
MODEL=gemini-3.5-flash KEY_INDEX=1 ./run.sh 4
MAX_CANDIDATE_PATCHES=8 RUN_NAME=agent-4-sample ./run.sh 4
GENERATION_RETRIES=2 ./run.sh 4
```

`GENERATION_RETRIES` defaults to `1`. It only retries Gemini generation when the CLI stream/subprocess/timeout layer fails. It does not retry because a generated evaluator has low coverage, fails the reference, fails the counter reference, or scores poorly; those are evaluator-quality outcomes that should be reported by `score.sh`.

Keep historical agent prompt files intact. Add a new agent id when changing prompt semantics enough that old runs should remain reproducible.

## Codex Comparison Runner

Codex can run the same v4 two-stage prompt flow through:

```bash
python scripts/run_codex_test_generator.py \
  --generation-mode two-stage \
  --run-name evalgen-agent-4-codex-gpt55 \
  --model gpt-5.5 \
  --project-source /mnt/data/projects/verl-vllm-capability-benchmark/task/verl \
  --agent-prompt /mnt/data/projects/verl-vllm-capability-benchmark/task/PROMPT.md \
  --reference-project /mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0 \
  --generator-prompt-stage1 docs/prompts/unit-test-generator-prompt-v4-stage1.md \
  --generator-prompt-stage2 docs/prompts/unit-test-generator-prompt-v4-stage2.md \
  --timeout-sec 0 \
  --skip-validation
```

The Codex runner keeps one Docker container alive across both stages and resumes the exact Stage 1 Codex thread in Stage 2 using the recorded thread id. Scoring remains a separate `score.sh /path/to/generated_tests` step.
