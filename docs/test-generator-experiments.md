# Test Generator Experiments

This document records attempts to generate evaluation tests automatically with Gemini CLI.

## Current Generic Prompt

The reusable prompts live in `docs/prompts/`. The runners default to the latest versioned prompt.

The prompt is intentionally codebase-agnostic. It defines standard workspace inputs:

- `/workspace/project`: incomplete project candidates edit.
- `/workspace/AGENT_PROMPT.md`: requested missing capability, visible to candidate agents.
- `/workspace/reference_project`: known-correct implementation, used only as a behavior oracle.
- `/workspace/generated_tests/`: generated test output.

The automation entry point is `scripts/run_gemini_test_generator.py`. It prepares exactly those three
task inputs, runs Gemini CLI inside the CPU evaluation image, writes generated evaluator files to
`/workspace/generated_tests`, validates against reference and incomplete projects, deletes Gemini CLI
state, and records a small static audit.

Use `--num-runs N` to generate several independent evaluator candidates in one batch. Each run gets its
own workspace, reference/baseline validation logs, static audit, and validation summary.

Use `--max-candidate-patches N` to discover previous `candidate.patch` files from
`/mnt/data/projects/verl-vllm-gemini-runs`, and use `--score-manual-suite` to score the hand-written
evaluator on the same targets. Each run writes `score_table.md` with rows for the generated evaluator
and hand-written evaluator, columns for baseline, reference, and candidate patches, plus unit-test counts.

The default generation mode is a single prompt. Legacy staged generation remains available with
`--generation-mode staged`, but it is not the default path for prompt iteration.

## Runs

### v1: no reference project

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-20260627-075753`

Result:

- Generated 5 test files plus README and runner.
- Baseline result in CPU test image: `4 failed, 1 passed, 10 skipped`.
- Reference result in CPU test image: `2 failed, 3 passed, 10 skipped`.

Main issues:

- Heavy structural assertions against registry internals and class names.
- Many `pytest.skip` calls when the missing capability was absent.
- Mocked the behavior under test and asserted configured mock returns.
- Included tautological assertions such as `or True`.
- Runner attempted package installation instead of using the evaluation image.

### v2: stricter anti-skip and anti-tautology prompt

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-v2-20260627-080452`

Result:

- Generated one test file plus README and runner.
- Baseline result in CPU test image: `2 failed, 1 passed, 5 skipped`.
- Reference result in CPU test image with normalized user env: `3 failed, 5 passed`.

Main issues:

- Still used private registry checks and exact class names.
- Still skipped most behavior tests when the missing capability was absent.
- Built invented configs that failed against the correct reference implementation.
- Reference failures showed the tests were not fair.

### v3: reference project provided as behavior oracle

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-v3-reference-20260627-081202`

Result:

- Gemini CLI ran inside `verl-vllm-capability-benchmark:task-v0`.
- Generated `generated_tests/README.md`, `generated_tests/run_tests.sh`, and `generated_tests/tests/test_vllm_rollout.py`.
- The CLI process was stopped manually after files were written because it stopped making progress.
- Baseline result in CPU test image: `1 passed, 5 failed, 11 errors`.
- Reference result in CPU test image: `15 passed, 2 failed`.

What improved:

- Runner accepts `PROJECT_UNDER_TEST`, so the same generated tests can target baseline or reference.
- No skip-heavy behavior; baseline fails loudly.
- Added broader checks for config serialization, LoRA rank handling, adapter setup, and async replica delegation.

Remaining blockers:

- Still asserts reference-specific implementation details such as exact class names and module/helper paths.
- Some tests target helper functions discovered only in the reference implementation, not public behavior required by the feature request.
- External vLLM platform behavior was not mocked consistently when vLLM was installed in the CPU test image, causing a reference failure.
- Async actor mocks returned plain dictionaries where the reference implementation awaited remote-call results, causing a reference failure.
- Gemini printed environment variables during exploration; temporary CLI state was deleted and logs were redacted afterward.

## Prompt Changes After v3

The prompt now explicitly requires:

- no reference-only helper/class/module assertions unless public context requires them,
- consistent mocking of optional external dependencies even when installed,
- async mocks that preserve awaitable/call semantics,
- no environment or secret dumps during generation,
- no claims that reference passes unless actually verified.

### v4: exactly three inputs, reference validation inside CPU image

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-v4-three-inputs-20260627-233302`

Result:

- Gemini CLI ran inside `verl-vllm-rollout-benchmark:cpu-tests`.
- Generated 5 test files plus README and runner.
- Reference result in CPU test image: `25 passed`.
- Incomplete project result in CPU test image: `6 failed, 19 errors`.

What improved:

- First generated evaluator that passed the official reference implementation.
- The incomplete project failed loudly for missing vLLM support.
- The tests no longer skipped the requested capability.

Remaining blockers:

- Still asserted exact implementation class names.
- Still imported and tested reference-only vLLM modules/helpers directly.
- Several tests checked server-launch helper details instead of public behavior required by the feature prompt.

### v5: stronger public-surface restrictions

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-v5-public-surface-20260627-234620`

Result:

- Generated one test file plus README, SELF_AUDIT, and runner.
- Reference result in CPU test image: `4 passed`.
- Incomplete project result in CPU test image: `4 failed`.

What improved:

- No direct imports from reference-only vLLM implementation modules.
- No exact class-name or inheritance assertions.
- Uses public rollout selectors and manager/client-level entry points visible in the incomplete project.
- Reference passed cleanly and incomplete project failed for missing vLLM registration/resolution.

Remaining blockers:

- Coverage collapsed to a small set of registry, manager-resolution, and lifecycle routing tests.
- It did not exercise enough downstream behavior from the task prompt, such as generation request/response contracts,
  rollout-output consumers, weight-update semantics, cache/memory lifecycle around update phases, or richer config
  propagation.
- This run showed that the prompt needs an explicit breadth discipline: do not solve the public-surface constraint by
  reducing the evaluator to smoke tests.

Full historical patch scoring for this generated evaluator:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-score-v5-all-candidates-20260628-002459/score_table.md`

Summary:

- Generated evaluator count: 4 tests.
- Hand-written evaluator count: 55 tests.
- Reference: generated `4/4`, hand-written `55/55`.
- Baseline: generated `0/4`, hand-written `25/55`.
- Historical candidates under the generated evaluator mostly scored `2/4`, with one at `0/4`.
- The same historical candidates under the hand-written evaluator ranged from `25/55` to `40/55`.

Interpretation:

- v5 is reference-correct but too coarse for candidate ranking.
- The hand-written evaluator has materially better discrimination across partially correct implementations.
- Future generated evaluators should be judged by this score table pattern, not only by reference pass and baseline fail.

## Prompt Changes After v5

The prompt now adds coverage discipline without specifying a fixed number of tests:

- build a path-effect coverage map from `/workspace/AGENT_PROMPT.md` and the visible project,
- cover every CPU-feasible public workflow affected by the missing capability or document why it is not CPU-feasible,
- treat selection/configuration as only one category,
- exercise downstream consumers after selection when the task asks for first-class integration,
- trace both upstream selectors/configs and downstream consumers of outputs, metadata, errors, lifecycle state, or side effects,
- prefer independent behavior tests with different failure modes over one broad smoke test,
- avoid traditional source coverage of deleted reference modules as an objective.

The runner now enforces these as quality gates:

- reference must pass and baseline must fail,
- `TASK_ANALYSIS.md` must exist and explain the task scope before tests are written,
- `COVERAGE_MAP.md` must exist,
- generated tests must avoid hard structural patterns such as private registries and class-name/inheritance checks,
- when historical patches are supplied, very low score diversity triggers a repair round.

The prompt deliberately does not require any fixed number of tests. The intended fix is judgment quality:
the generator must first reason about public workflows and plausible incomplete implementations, then let
that analysis determine the evaluator's size.

### finalgate: no-leak repair plus reference-only path audit

Run directory:

`/mnt/data/projects/verl-vllm-testgen-runs/testgen-gemini35flash-scope-aware-testgen-finalgate-20260628-025758`

Result:

- Generated evaluator count: 8 tests.
- Hand-written evaluator count: 55 tests.
- Reference: generated `8/8`, hand-written `55/55`.
- Baseline: generated `1/8`, hand-written `25/55`.
- Historical candidates under the generated evaluator ranged from `1/8` to `4/8`.
- The same historical candidates under the hand-written evaluator ranged from `25/55` to `40/55`.
- Final generated evaluator quality issues: none.

Pipeline changes validated by this run:

- Repair rounds hide `/workspace/candidate_targets`, so Gemini sees aggregate scores and audit findings but not candidate implementation source.
- Static audit now ignores `__pycache__`/`.pyc` noise.
- Static audit flags private reflection such as `__module__`, `__ray_actor_class__`, `_underlying_class`, and `sys.modules[...]`.
- Reference-only dotted-path audit catches tests importing or patching modules present only in `/workspace/reference_project`.
- Gemini API keys are read from `/mnt/data/projects/verl-vllm-secrets/gemini_api_keys.txt` by default; Gemini state is removed after runs.

Interpretation:

- This is the first generated evaluator from the automated pipeline that satisfies the current mechanical gates.
- It is still substantially narrower than the hand-written evaluator. It distinguishes candidates better than v5 but still misses many behavior dimensions covered by the 55-test suite.
