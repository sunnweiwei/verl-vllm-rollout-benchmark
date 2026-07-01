# Generic Test-Generator Pipeline

This pipeline generates evaluator tests for benchmarks where a candidate agent must implement a missing capability in an incomplete project.

## Inputs

Each task provides exactly three semantic inputs to the test generator:

- `/workspace/project`: the incomplete project visible to candidate agents.
- `/workspace/AGENT_PROMPT.md`: the feature request visible to candidate agents.
- `/workspace/reference_project`: a known-correct implementation used as a behavior oracle.

The generator must not use hand-written evaluator tests, hidden construction notes, solution notes, or candidate patch source trees.

The runner may additionally receive validation targets such as a counter-reference implementation. These are not visible to the generator. They are known-good scoring targets used only after generation to detect over-specified tests. For the current VERL/vLLM benchmark, the strict counter reference is added automatically when it exists locally; use `--no-default-validation-targets` to disable that default for another task.

## Generation

The current default generator is a single coding-agent prompt. The older staged workflow is still available for experiments, but it follows the same output contract: the model writes tests, and the runner owns execution.

In staged mode, the stages are:

1. **Project-first contract analysis**
   - `PROJECT_CONTRACT_SCAN.md`
   - `TASK_ANALYSIS.md`
   - `PUBLIC_CONTRACTS.md`
   - `INFERRED_CONTRACTS.md`

   During this stage, `/workspace/reference_project` is temporarily hidden so the initial contract scan can only use `/workspace/project` and `/workspace/AGENT_PROMPT.md`.

2. **Reference evidence triage**
   - `REFERENCE_TEST_TRIAGE.md`
   - `NON_CONTRACTS.md`
   - `CONTRACT_DECISION_TABLE.md`
   - `PUBLIC_BEHAVIOR_TARGETS.md`
   - `OPTIONAL_DEPENDENCY_STUBS.md`
   - `IMPORT_COMPATIBILITY_AUDIT.md`

   This is the only stage where `/workspace/reference_project` is visible. Later stages use the triage output rather than reopening reference implementation files.

3. **Behavior and harness design**
   - `BEHAVIOR_INVENTORY.md`
   - `HARNESS_STRATEGY.md`
   - `COVERAGE_MAP.md`

4. **Test planning**
   - `TEST_PLAN.md`
   - `COVERAGE_SUFFICIENCY_AUDIT.md`
   - `TEST_CONTRACT_AUDIT.md`
   - `FAILURE_MODE_AUDIT.md`

5. **Evaluator implementation**
   - `test_*.py`
   - `README.md`
   - `SELF_AUDIT.md`

The generator must not write a custom test runner. If it writes `run_tests.sh`, the benchmark runner ignores it and records that as an extra artifact.

Between stages, the runner writes audit reports. These reports are diagnostic; they do not repair, reject, or rewrite the generated evaluator.

## Validation

After generation, the runner:

- runs the generated pytest files against the reference project with the target project and generated test directory on `PYTHONPATH`,
- runs the same pytest files against the incomplete baseline through the same fixed harness,
- runs the same pytest files against any extra known-good validation targets, such as structurally different counter references,
- optionally applies historical candidate patches and scores each patched project,
- optionally runs the hand-written evaluator for comparison.

Candidate patches are scoring targets only. Gemini does not inspect candidate implementation source trees.

## Audits

The runner records static and semantic audit findings such as:

- structural-looking oracles,
- import-only or construction-only checks,
- mock-call-count assertions,
- reference-only module paths,
- reference-only symbols,
- `NON_CONTRACTS.md` reuse,
- placeholder model paths,
- optional dependency or network/download patterns.

These findings are not hard blockers. They are review signals because the same pattern may be correct or incorrect depending on the task context and public contracts.

## Acceptance Signals

A generated evaluator is promising when:

- the reference project passes with zero failures and zero errors,
- each extra validation target passes with zero failures and zero errors,
- the incomplete baseline fails for the missing capability,
- the generated suite contains real pytest tests,
- the analysis artifacts explain the public behavior surface and CPU harness choices,
- audit findings are either absent or justified by public/inferred contracts,
- optional candidate patch scores show meaningful differences among partial implementations.

The score table is diagnostic, not a replacement for review. A generated evaluator can pass reference and fail baseline while still being too shallow.

Every generated evaluator report should include pass-rate rows for:

- the initial incomplete codebase (`baseline` in the runner table; expected to fail, usually near 0 passed);
- the official reference (`reference`; expected all passed);
- all counter-reference validation targets (`counter_reference` for this benchmark; expected all passed);
- every candidate or baseline variant supplied through `--candidate-patch` or `--max-candidate-patches`, with each variant reported separately.

## Porting To Another Task

For a new benchmark, replace:

- `--project-source`
- `--agent-prompt`
- `--reference-project`
- `--no-default-validation-targets`
- optionally `--validation-target`
- optionally `--candidate-runs-root` or `--candidate-patch`

Task-specific knowledge should come from the three semantic inputs. Codebase-specific rule-based blockers should not be added unless the condition is invalid in every possible context.
