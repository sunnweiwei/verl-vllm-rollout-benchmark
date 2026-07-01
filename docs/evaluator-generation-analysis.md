# Evaluator Generation Analysis

This document tracks generated evaluator versions for the VERL vLLM rollout benchmark.

## Candidate Label Map

| ID | Candidate patch |
| --- | --- |
| C1 | `20260626-081247-gemini31pro-preview-candidate2` |
| C2 | `20260626-091059-gemini31pro-v2-candidate1` |
| C3 | `20260626-091100-gemini31pro-v2-candidate2` |
| C4 | `20260626-091100-gemini31pro-v2-candidate3` |
| C5 | `20260626-091100-gemini31pro-v2-candidate4` |
| C6 | `20260626-093319-gemini35flash-v3-candidate1` |
| C7 | `20260626-093319-gemini35flash-v3-candidate2` |
| C8 | `20260626-093319-gemini35flash-v3-candidate4` |
| C9 | `20260627-044228-gemini35flash-parity-candidate1` |
| C10 | `20260627-044228-gemini35flash-parity-candidate2` |
| C11 | `20260627-044228-gemini35flash-parity-candidate3` |

## Score Ledger

The `unit_tests` column below uses the actual pytest test count. Some runner outputs before 2026-06-28 incorrectly reported `0` because the runner only counted tests under `generated_tests/tests/`; that runner bug has been fixed.

| Version | Run dir | unit_tests | baseline | reference | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | Interpretation |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hand-written CPU evaluator | `testgen-gemini35flash-scope-aware-testgen-finalgate-20260628-025758` | 55 | 25/55 | 55/55 | 33/55 | 39/55 | 36/55 | 39/55 | 39/55 | 40/55 | 38/55 | 37/55 | 27/55 | 38/55 | 25/55 | Best current discriminator. |
| Gemini generated, finalgate prompt | `testgen-gemini35flash-scope-aware-testgen-finalgate-20260628-025758` | 8 | 1/8 | 8/8 | 3/8 | 4/8 | 3/8 | 3/8 | 3/8 | 3/8 | 4/8 | 3/8 | 1/8 | 3/8 | 3/8 | Some signal, but shallow. |
| Prompt v3 | `testgen-score-existing-v3-all-candidates-20260628-044525` | 23 | 0/3 | 23/23 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | Invalid as a discriminator: baseline/candidates hit collection errors. |
| Prompt v4 | `testgen-score-existing-v4-all-candidates-20260628-044647` | 10 | 0/10 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | Invalid as a discriminator: all non-reference targets fail fixture setup. |
| Prompt v5, contract-first | `testgen-gemini35flash-contract-first-prompt-v5-20260628-045604` | 11 | 0/11 | 11/11 | 1/11 | 1/11 | 1/11 | 0/11 | 2/11 | 1/11 | 1/11 | 1/11 | 0/11 | 1/11 | 1/11 | Reference passes, but evaluator over-constrains reference class/module shape. |
| Prompt v5b, evidence-locked | `testgen-gemini35flash-evidence-locked-prompt-v5b-20260628-051037` | 10 | 2/10 | 10/10 | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 6/10 | 2/10 | 5/10 | 5/10 | Rejected by quality audit despite useful score spread. |
| Prompt v5c, no-structure guard | `testgen-gemini35flash-no-structure-prompt-v5c-20260628-052746` | 5 | not scored | not scored | - | - | - | - | - | - | - | - | - | - | - | Aborted: Gemini entered an open-ended self-repair loop and repeatedly started Ray clusters. |
| Prompt v5d, bounded behavior | `testgen-gemini35flash-bounded-behavior-prompt-v5d-20260628-053802` | 6 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Invalid; static guard blocked pytest because tests and planning docs still used reference-only vLLM modules. |
| Prompt v5e, no-shell contract-doc guard | `testgen-gemini35flash-no-shell-contract-doc-prompt-v5e-20260628-054550` | 5 | 1/5 | 3/5 | - | - | - | - | - | - | - | - | - | - | - | Invalid; no shell loop, but tests still reused reference-only bare symbols and triggered model-download IO. |
| Prompt v5e static replay after new guard | `testgen-score-existing-v5e-static-guard-20260628-055309` | 5 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | New guard catches `dummy_path`, reference-only bare symbols, and `NON_CONTRACTS.md` reuse before pytest. |
| Prompt v5f, symbol guard prompt | `testgen-gemini35flash-symbol-guard-prompt-v5f-20260628-055346` | 9 | not accepted | not accepted | - | - | - | - | - | - | - | - | - | - | - | Invalid; CLI still used shell/web because deprecated `--allowed-tools` did not enforce the intended sandbox. |
| Prompt v5f static replay after parser fix | `testgen-score-existing-v5f-static-guard2-20260628-060133` | 9 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | New symbol/contract audits block real issues without the earlier noisy `verl/workers/rollout` false positives. |
| Prompt v5g, admin policy prompt | `testgen-gemini35flash-admin-policy-prompt-v5g-20260628-060225` | 5 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Tool policy works (`tool_usage_audit=none`), but prompt still lets reference-only bare symbols leak into docs/tests. |
| Prompt v5h, quarantine vocabulary prompt | `testgen-gemini35flash-quarantine-vocab-prompt-v5h-20260628-060801` | 3 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Much cleaner; only blocked for `get_device_uuid` leakage and placeholder model path usage. |
| Prompt v5h static replay after placeholder guard | `testgen-score-existing-v5h-static-guard-20260628-061450` | 3 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | New guard catches both reference-only `get_device_uuid` and `mock_model_path` before pytest. |
| Prompt v5i, placeholder guard prompt | `testgen-gemini35flash-placeholder-guard-prompt-v5i-20260628-061514` | 6 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Invalid; regressed to `ServerAdapter/equivalent` phrasing and mock-call-list assertions. |
| Prompt v5i static replay after call-list guard | `testgen-score-existing-v5i-static-guard-20260628-061932` | 6 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | New guard catches `call_args_list` plus reference-name/equivalent leakage before pytest. |
| Prompt v6, staged inventory/plan/tests | `testgen-gemini35flash-staged-v6-20260628-072623` | 12 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Staging broadened behavior analysis, but tests still used subclass/interface oracles, hidden wrapper unwrapping, global import shims, and a forbidden shell-tool attempt. |
| Prompt v7, staged stricter structure guard | `testgen-gemini35flash-staged-v7-20260628-073908` | 7 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Safer but too shallow: it generated only rollout config tests and misclassified public replacement names from `NON_CONTRACTS.md` as quarantined. |
| Prompt v8, public-boundary harness prompt | `testgen-gemini35flash-staged-v8-20260628-075100` | 10 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Coverage widened to lifecycle/cache/weight/generation, but pytest leaked `ServerAdapter` and `BucketedWeightSender` reference-only symbols. |
| Prompt v9, reference-string guard prompt | `testgen-gemini35flash-staged-v9-20260628-080137` | 9 | static-blocked | static-blocked | - | - | - | - | - | - | - | - | - | - | - | Still invalid: stage plan kept subclass/interface oracles, final pytest used `pytest.skip`, and a generic `initialize` symbol false positive was exposed. |
| Prompt v10, staged semantic gates | `testgen-gemini35flash-staged-gated-v10-20260628-081433` | n/a | aborted | aborted | - | - | - | - | - | - | - | - | - | - | - | Stage gates worked: stage 1 repair removed subclass blockers; stage 2 gate caught subclass/reference-helper leakage, then repair still attempted forbidden shell and left mock-call oracles, so the run was manually stopped before invalid pytest generation continued. |

## Prompt v3 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `SELF_AUDIT.md`
- `README.md`
- `test_vllm_cli_args_on_cpu.py`
- `test_vllm_integration.py`

Main problems:

- `test_vllm_cli_args_on_cpu.py` imports `verl.workers.rollout.vllm_rollout.utils` at module import time. On the incomplete project this module is absent, so pytest collection stops before most tests run.
- Several tests assert reference implementation shape rather than behavior: exact `__name__`, `RolloutReplicaRegistry._registry`, `_get_server_name_prefix`, and concrete `ServerAdapter`.
- Many tests start from modules that exist only in the reference implementation rather than from a stable public path in the task project.
- The suite has 23 tests, but most of the apparent coverage is gated behind the first missing import. Candidate score diversity is `[0]`, so it cannot rank candidate quality.

Workflow problems that caused v3:

- The model treated `reference_project` as a source of required APIs, not only an oracle for behavior.
- The model wrote tests before proving that each required module path was visible from the task project or prompt.
- The validation only checked that baseline failed; it did not distinguish a meaningful test failure from a collection-time import failure.

## Prompt v4 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `SELF_AUDIT.md`
- `README.md`
- `test_vllm_rollout_cpu.py`

Main problems:

- `PUBLIC_CONTRACTS.md` explicitly says contracts may come from either `/workspace/project` or `/workspace/reference_project`. This violates the intended rule. Reference-only implementation details became required APIs.
- The autouse fixture patches `verl.workers.rollout.vllm_rollout.vllm_rollout.get_device_uuid` and `get_device_id`. On baseline and all candidates without that exact module path, fixture setup fails before test bodies run.
- The tests still assert exact concrete names: `ServerAdapter`, `vLLMReplica`, `vLLMHttpServer`.
- The tests call private/internal methods and state: `_ensure_server_handle`, `_server_address`, exact Ray actor names, exact ZMQ socket formatting, `sys.modules` sweeping.
- Several tests verify mocked method calls rather than public behavioral effects. For example, weight update is judged by `BucketedWeightSender` and `clear_kv_cache.remote` calls, not by a public update/synchronization outcome.
- The model spent many repair iterations fixing brittle mocks until reference passed. This made the evaluator overfit to the reference implementation and still fail every non-reference candidate at setup.

Workflow problems that caused v4:

- `PUBLIC_CONTRACTS.md` was not forced to cite project/prompt-only evidence. The model was allowed to launder reference implementation names into "contracts."
- Reference validation became a local repair target. The model repeatedly patched mocks to satisfy the reference instead of stepping back and simplifying tests to public behavior.
- The workflow lacked a hard acceptance rule that baseline and candidates must collect the entire test suite and then fail through ordinary assertions, not autouse fixture setup errors.
- The workflow did not require a "negative path audit" after writing tests: remove autouse fixtures and top-level patches that touch missing-capability modules.

## Recommended Next Prompt Changes

- Make `PUBLIC_CONTRACTS.md` schema strict: every required module path, class name, method name, config field, return field, or error type must cite `/workspace/project` or `/workspace/AGENT_PROMPT.md`. Reference evidence can only fill expected behavior for an already-established public contract.
- Add a `NON_CONTRACTS.md` section: list reference-only names observed but forbidden to assert directly.
- Add a baseline collection gate: baseline must collect all generated tests. Missing capability should fail inside individual tests with clear assertions, not in module import, fixture setup, autouse patches, or collection.
- Ban autouse fixtures that patch missing-capability modules. Fixtures may patch external dependencies or existing project modules only.
- Require tests to import candidate-only modules inside the test body and treat absence as one specific test failure, not a suite-level setup error.
- Require a `FAILURE_MODE_AUDIT.md`: for each test, describe how baseline should fail and why a partial implementation could pass/fail. Reject any test whose baseline failure is "module path missing during setup."
- Prefer testing through existing backend-agnostic selectors, config merge/validation paths, trainers/controllers/request builders, and result consumers.
- Avoid exact concrete class names unless the task prompt or existing project public API requires them. Prefer "selected backend can be constructed and used through the shared public interface."
- Avoid mock-call assertions as primary oracle. If mocking is needed, assert externally visible data passed through the public boundary: generated request payloads, returned result objects, error semantics, state transitions, output metadata, or config effects.
- After generation, always run post-hoc candidate scoring for all historical patches and record the score table. Use this as diagnostics, not as a repair objective.

## Prompt v5 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`

Main problems:

- The prompt improved workflow compliance: the model produced the requested analysis files and reference/baseline validation ran cleanly.
- The tests still asserted reference implementation identity: `ServerAdapter`, `vLLMReplica`, direct imports from `verl.workers.rollout.vllm_rollout`, and reference-specific patch targets.
- `SELF_AUDIT.md` claimed reference-only details were not asserted, but the pytest file contained `__name__`, `issubclass`, exact reference module imports, and mock-call assertions.
- Candidate scores were very low and mostly clustered because many tests failed before reaching behavior-level differences.

Workflow problems that caused v5:

- The prompt asked for a contract ledger but did not force a mechanical trace from pytest code back to that ledger.
- The runner's reference-only dotted-path audit only scanned `generated_tests/tests/`, so it missed root-level `generated_tests/test_vllm_rollout.py`. This audit bug has been fixed.

## Prompt v5b Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`

Main problems:

- The added `TEST_CONTRACT_AUDIT.md` surfaced the right workflow idea, but the model still laundered reference-only APIs into "public-contract" entries. Example: it marked `verl.workers.rollout.vllm_rollout.utils.build_cli_args_from_config` public because the task was about vLLM, not because project/prompt evidence required that module.
- The model treated neighboring backend implementation conventions as requirements for the new backend. `ServerAdapter` appears in SGLang/TRTLLM paths, but that does not by itself require a vLLM implementation to use the same concrete class name.
- To force reference success, the tests created a broad fake dependency universe through `sys.modules` and `sys.meta_path`, then patched reference-only internals. This is not a behavior evaluator.
- The fixed runner correctly rejected the suite with quality issues:
  - structural/private-pattern usage: `__name__`, `issubclass`, `hasattr(`, `sys.modules[`;
  - reference-only imports/patch targets under `verl.workers.rollout.vllm_rollout.*`.
- Candidate scores had more spread than v5, but the spread is not trustworthy because several tests are dominated by structural/interface checks and reference-only patches.

Prompt/workflow changes implied by v5b:

- Neighboring examples may establish shared behavior and shared public interface expectations, but not concrete names for the new capability unless the prompt/project explicitly says so.
- The generated pytest code should be invalid if it contains hard-forbidden tokens such as `sys.modules[`, `__name__`, `issubclass`, or reference-only module paths. A prose self-audit is not enough.
- Reference validation should not be repaired by adding a fake global dependency universe. If a reference path cannot run on CPU without broad dependency injection, the test should move up to a public boundary or be recorded as a CPU gap.

## Prompt v5c Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`

Main problems:

- The model did perform the requested staged analysis, but the executable tests regressed to structure checks: `__mro__` base-class assertions were used as the selector oracle.
- The tests used mock-call state (`.called`) as the main oracle for weight update, cache management, and abort/resume behavior.
- The tests patched project behavior directly with `patch.object(replica_class, "launch_servers", ...)`, which bypasses the lifecycle behavior being evaluated.
- The tests still patched reference-only internals such as `verl.workers.rollout.vllm_rollout.vllm_rollout.BucketedWeightSender`.
- The tests started real Ray clusters via `ray.init` and `@ray.remote`. Gemini then entered a slow self-repair loop, repeatedly running reference tests, adding config fields, and restarting Ray.
- Reference validation never stabilized: intermediate runs failed on invalid OmegaConf/dataclass setup, async actor local-mode limitations, and frozen dataclass mutation. The runner process was terminated before candidate scoring to avoid wasting the machine.

Workflow problems that caused v5c:

- The prompt said not to use structural checks, but the hard invalid-token list did not include `__mro__`, `.called`, or `patch.object`.
- The validation instructions still encouraged Gemini to chase reference success. It did not have a clear stop rule for "same brittle setup class keeps failing."
- The prompt did not explicitly say that CPU tests should avoid starting real distributed runtimes when fake public-boundary handles can expose the behavior.
- The test plan allowed "assert class types" and "mock server receives call" language, which the model converted directly into pytest assertions.

Changes made after v5c:

- Added hard static-audit patterns for `__mro__`, `.called`, and `patch.object`.
- Added a quality issue when generated CPU tests start a real Ray runtime with `ray.init(` or define Ray actors with `@ray.remote`.
- Updated the generator prompt to reject inheritance/interface assertions unless they are paired with downstream behavior, reject `patch.object` repairs on project objects, forbid `.called` as a primary oracle, and bound validation to one focused revision pass.

## Prompt v5d Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`

Main problems:

- The model still laundered reference-only names into public planning artifacts. Examples:
  - `TASK_ANALYSIS.md` said `get_rollout_class("vllm", "async")` should point to `verl.workers.rollout.vllm_rollout.ServerAdapter`.
  - `COVERAGE_MAP.md` targeted `build_cli_args_from_config` and `BucketedWeightSender/BucketedWeightReceiver`, even though those were reference-only or already listed as non-contracts.
  - `TEST_CONTRACT_AUDIT.md` incorrectly marked `verl.workers.rollout.vllm_rollout.*` imports as public contracts.
- The pytest file imported reference-only modules directly:
  - `verl.workers.rollout.vllm_rollout.vllm_rollout`
  - `verl.workers.rollout.vllm_rollout.utils`
  - `verl.workers.rollout.vllm_rollout.bucketed_weight_transfer`
- The tests used structural/smoke or overfit checks, such as `type(rollout_cls) is type`, exact error-message strings tied to the reference implementation, `adapter.zmq_handle` socket-name substrings, and mock call-list inspection as the oracle.
- Gemini ran pytest inside the generation phase, then started repairing reference failures by trial-and-error config changes. The run was terminated after it entered a second self-repair loop.
- A replay with the improved static guard produced `exit 126` before pytest:
  - `.called` in `test_vllm_rollout.py`
  - reference-only imports/patches in `test_vllm_rollout.py`
  - reference-only module paths in `TASK_ANALYSIS.md`, `TEST_PLAN.md`, `TEST_CONTRACT_AUDIT.md`, and `FAILURE_MODE_AUDIT.md`

Changes made after v5d:

- Removed `run_shell_command` from Gemini's allowed tools for evaluator generation. Gemini now only reads/searches/writes; the outer runner owns all validation.
- Changed the prompt's validation step into a design-review step: do not execute pytest, package imports, model downloads, runtime probes, or distributed-runtime startup while generating.
- Added explicit prompt rules that analysis artifacts are part of the evaluator contract and cannot describe reference-only names as public, expected, or user-visible.
- Added a document-level reference-only contract audit for `TASK_ANALYSIS.md`, `PUBLIC_CONTRACTS.md`, `COVERAGE_MAP.md`, `TEST_PLAN.md`, `TEST_CONTRACT_AUDIT.md`, `FAILURE_MODE_AUDIT.md`, `README.md`, and `SELF_AUDIT.md`, excluding `NON_CONTRACTS.md`.

## Prompt v5e Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`

Main problems:

- Removing shell execution worked: Gemini no longer entered a pytest/Ray self-repair loop during generation.
- The generated evaluator was still too small and shallow: 5 tests, mostly config/selector/construction checks.
- Reference failed 2 tests because `HFModelConfig(path="dummy_path")` attempted HuggingFace tokenizer lookup. This violates hermetic CPU-test expectations.
- The model laundered reference-only bare symbols rather than dotted module paths:
  - `COVERAGE_MAP.md` used `vLLMReplica` and `ServerAdapter` as test strategy/behavior targets.
  - `TEST_PLAN.md` and pytest asserted `zmq_handle`, even though `NON_CONTRACTS.md` listed it as private reference state.
- Static dotted-path audit was insufficient because these failures did not require full module paths.

Changes made after v5e:

- Added static audit patterns for `dummy_path`, HuggingFace downloader APIs, and direct `requests.get/post` calls in generated tests.
- Added a generic reference-only symbol audit: extract class/function definitions from Python files that exist only in `/workspace/reference_project`, then flag their reuse in contract-bearing docs or pytest code.
- Added a generic `NON_CONTRACTS.md` reuse audit: extract backticked symbols from `NON_CONTRACTS.md` and flag reuse in `TASK_ANALYSIS.md`, `PUBLIC_CONTRACTS.md`, `COVERAGE_MAP.md`, `TEST_PLAN.md`, `TEST_CONTRACT_AUDIT.md`, `FAILURE_MODE_AUDIT.md`, or pytest files.
- Wired both new audits into pre-pytest static validation blockers, quality issues, repair feedback, and validation summaries.
- Strengthened the generator prompt with hermetic CPU-test rules: no placeholder remote model paths, no model hub downloads, and explicit reasoning about config/model/tokenizer constructors that may perform IO.

## Prompt v5f Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- The Gemini CLI did not honor the intended read/write-only constraint from deprecated `--allowed-tools`. The stream shows forbidden tools:
  - `run_shell_command` at 05:57:25, 05:57:31, 05:57:38, 05:58:07, and 05:58:27.
  - `google_web_search` at 05:57:48.
- The generated tests still targeted reference shape and private state:
  - `ServerAdapter` and `BucketedWeightSender` appeared in coverage/test plan/test code.
  - `zmq_handle` was listed in `NON_CONTRACTS.md` and then asserted in the test suite.
  - `.called` and `assert_called` appeared in the pytest file.
- Gemini ran the evaluator itself and tried to debug reference failures. Reference failed because `HFModelConfig()` with default path `???` triggered HuggingFace model/config lookup.
- The first `NON_CONTRACTS.md` parser was too noisy: it split dotted paths into `verl`, `workers`, and `rollout`, and picked up public alternatives such as `update_weights`.

Changes made after v5f:

- Switched Gemini execution from deprecated `--allowed-tools` to a generated `--admin-policy` TOML file:
  - allow only `read_file`, `read_many_files`, `write_file`, `grep_search`, `glob`, `list_directory`, and `replace`;
  - deny every other tool, including shell and web tools.
- Added a stream-json forbidden tool usage audit. If Gemini attempts any tool outside the allowed read/search/write set, validation is blocked before pytest and recorded in summary/repair feedback.
- Refined `NON_CONTRACTS.md` parsing:
  - Markdown tables only contribute symbols from the first column;
  - dotted paths contribute only the final class/function symbol;
  - lowercase package path components are ignored.
- Strengthened the prompt to forbid web/external docs during generation and to move heavyweight model/tokenizer config construction tests to higher public boundaries or CPU gaps.

## Prompt v5g Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- The admin policy solved the tool isolation problem. `tool_usage_audit_attempt0.txt` was `none`, and Gemini did not run shell/web/pytest during generation.
- The evaluator was still invalid because reference-only bare symbols leaked outside `NON_CONTRACTS.md`:
  - `TASK_ANALYSIS.md` mentioned `ServerAdapter`.
  - `COVERAGE_MAP.md` targeted `vLLMReplica` and `ServerAdapter`.
  - `test_vllm_rollout.py` used `vLLMReplica` and `ServerAdapter` in generated test code.
- The suite shrank back to 5 tests and remained dominated by selector/implementation-construction behavior, not broad public workflows.

Changes made after v5g:

- Added a "quarantined vocabulary" rule to the generator prompt:
  - once a reference-only name appears in `NON_CONTRACTS.md`, it may not appear in any later analysis file, pytest function name, docstring, comment, string assertion, mock class, or variable intended to mirror the reference implementation;
  - later docs/tests must use neutral public-language descriptions such as "the selected rollout class" or "the backend implementation."
- Added explicit invalidity rules for planned tests that need to instantiate the selected missing-capability implementation using reference-only constructors, private fields, heavyweight model/tokenizer loading, Ray actor layouts, or socket/shared-memory internals.

## Prompt v5h Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout_on_cpu.py`
- `run_tests.sh`

Main problems:

- The quarantined-vocabulary prompt substantially improved leakage:
  - `COVERAGE_MAP.md` used neutral names such as `selected_rollout` and `selected_replica`.
  - No broad `ServerAdapter` / `vLLMReplica` leakage appeared outside `NON_CONTRACTS.md`.
- The generated suite was still too small: 3 tests.
- The test code still used one reference-only helper path indirectly:
  - `monkeypatch.setattr("vllm.platforms.current_platform.get_device_uuid", ...)`
  - The static symbol audit mapped `get_device_uuid` to the reference-only utility file.
- The tests used `HFModelConfig(path="mock_model_path")`, which is another placeholder model path likely to trigger model/tokenizer lookup.

Changes made after v5h:

- Added `mock_model_path`, `fake_model_path`, and `???` to static audit patterns and hard blockers.
- Updated the prompt's hermetic-test section to explicitly ban those placeholder model identifiers alongside `dummy_path`.

## Prompt v5i Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- Tool isolation remained fixed (`tool_usage_audit=none`).
- The model regressed from v5h's neutral wording by writing `ServerAdapter/equivalent` in `COVERAGE_MAP.md`; this still makes a quarantined reference name part of the evaluator contract.
- The pytest file used `call_args_list` as a mock-call oracle for lifecycle behavior. That is still structural/mock-call testing rather than behavior at a public boundary.
- The generated tests attempted to instantiate selected backend classes with `HFModelConfig(path=str(tmp_path))`; this avoids a literal placeholder string, but likely still lacks the local tokenizer/config files required by the constructor.

Changes made after v5i:

- Added `call_args_list` and `call_count` to static audit patterns and hard blockers.
- Updated the prompt to forbid using quarantined names followed by "or equivalent", "/equivalent", "like", "such as", or parenthetical aliases.
- Updated the prompt to ban `call_args_list` and `call_count` as primary mock-call oracles.

## Prompt v6 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `BEHAVIOR_INVENTORY.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `COVERAGE_SUFFICIENCY_AUDIT.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- The staged workflow improved the analysis depth. `BEHAVIOR_INVENTORY.md` covered generation request mapping, prompt length validation, output packaging, prompt logprobs, memory/cache lifecycle, abort/resume, weight sync, server manager behavior, and neighboring-backend regression behavior.
- The executable tests remained structurally overfit. They imported shared base interfaces and used manual parent-class traversal as the selector oracle rather than exercising downstream public behavior.
- The tests peeled open selected runtime wrappers through `_underlying_class`, which makes hidden implementation structure part of the evaluator contract.
- The tests created a broad fake package tree by mutating `sys.modules`, so collection success depended on import shims rather than public project behavior.
- Gemini attempted `run_shell_command` during stage 3. The admin policy denied the tool, but the attempt itself shows the generator still tried to runtime-probe the environment.
- `NON_CONTRACTS.md` incorrectly mixed reference-only names with shared project-visible interfaces, and then those names leaked into planning/test files.

Changes made after v6:

- Strengthened the prompt to say shared project-visible interfaces belong in `PUBLIC_CONTRACTS.md`, not `NON_CONTRACTS.md`, while still forbidding subclass/interface-conformance as the test oracle.
- Added staged prompt reminders that selector rows are only activation coverage unless they reach a downstream public workflow.
- Added hard audit blockers for `__bases__` and all `sys.modules` usage, not just `sys.modules[`.
- Clarified that tests must not unwrap hidden actor classes or selected implementation wrappers.

## Prompt v7 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `BEHAVIOR_INVENTORY.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `COVERAGE_SUFFICIENCY_AUDIT.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_rollout_config.py`
- `run_tests.sh`

Main problems:

- The stricter structure guard pushed Gemini away from subclass and wrapper tests, but it over-corrected: the executable suite contained only 7 `RolloutConfig` tests.
- The suite did not exercise generation output packaging, agent-loop consumers, manager/client routing, cache/lifecycle controls, weight/update boundaries, async resume/abort behavior, or training consumers.
- The model marked too many downstream behaviors as CPU gaps instead of looking for public-boundary CPU harnesses such as fake server handles, synthetic public output objects, bounded local Ray actors, or downstream consumers.
- The `NON_CONTRACTS.md` parser was too broad. It quarantined backticked names in "How to test instead" guidance, including public replacement concepts such as `TokenOutput`, `RolloutReplica`, and `server_handle`.

Changes made after v7:

- Refined `NON_CONTRACTS.md` parsing to extract quarantined symbols only from item headings, leading backticked list items, or table first columns, not from public replacement prose.
- Relaxed the runner's hard block on controlled local Ray usage; `ray.init(` and `@ray.remote` are still audited, but no longer automatically invalidate a generated evaluator.
- Changed the prompt from pure prohibition to a public-boundary harness strategy: before marking downstream behavior as a CPU gap, the generator must consider fake server/client handles, synthetic public output objects, bounded local CPU runtime fixtures, downstream consumers, and monkeypatches at heavyweight external launch boundaries.
- Replaced the broad `sys.modules` hard block with targeted blockers for wholesale mutation patterns (`sys.modules.update`, `sys_modules_dict`) while allowing narrow optional-dependency stubs outside the missing capability.

## Prompt v8 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `BEHAVIOR_INVENTORY.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `COVERAGE_SUFFICIENCY_AUDIT.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- The public-boundary harness prompt improved coverage direction. The inventory and plan included lifecycle, colocated/standalone startup, cache controls, weight synchronization, async generation, and abort behavior.
- The executable suite grew to 10 tests, but static validation blocked before pytest.
- The tests still leaked reference-only implementation names:
  - `ServerAdapter` appeared inside an asserted exact error-message string.
  - `BucketedWeightSender` appeared in a dynamic module patch used to force weight sync onto a CPU path.
- The generated tests attempted to patch reference-only helpers instead of moving weight-sync assertions up to a public checkpoint/manager/client boundary.
- The stage-2 plan still described selector rows as inheritance checks and some lifecycle rows as "method was called" assertions, although the final blocker was reference-name leakage.

Changes made after v8:

- Extended reference-only dotted-path audit to scan arbitrary string constants in pytest code, not only imports and string patch targets. This catches dynamic lookups such as `sys.modules.get("reference.only.path")`.
- Updated the prompt and stage prompts to forbid reference-only names in comments, docstrings, error strings, fake class names, dynamic module lookups, and monkeypatch targets.
- Added explicit guidance that exact exception text containing implementation class/helper names is not a valid oracle.
- Added explicit guidance that reference-only transfer/helper classes must not be patched to create CPU tests; tests should move to public controller/client/consumer effects instead.

## Prompt v9 Findings

Generated files:

- `TASK_ANALYSIS.md`
- `PUBLIC_CONTRACTS.md`
- `NON_CONTRACTS.md`
- `BEHAVIOR_INVENTORY.md`
- `COVERAGE_MAP.md`
- `TEST_PLAN.md`
- `COVERAGE_SUFFICIENCY_AUDIT.md`
- `TEST_CONTRACT_AUDIT.md`
- `FAILURE_MODE_AUDIT.md`
- `README.md`
- `SELF_AUDIT.md`
- `test_vllm_rollout.py`
- `run_tests.sh`

Main problems:

- The stage-1 and stage-2 artifacts still contained structural selector oracles, e.g. "assert it is a subclass of `RolloutReplica`" and "subclassing `BaseRollout`".
- The final pytest file used `pytest.skip` when the missing capability was absent. That makes the baseline avoid failing for the missing behavior and is invalid for this benchmark.
- The broad reference-only symbol audit produced one false positive on the common symbol `initialize`, pulled from a reference-only performance test file.
- The result confirms that a natural-language prompt alone is not enough: Gemini repeatedly carries invalid planning rows into pytest even after the prompt says not to.

Changes made after v9:

- Added a staged semantic gate. After each staged Gemini call, the runner now audits planning artifacts for subclass/interface oracles, mock-call oracles, implementation-specific exact error-message assertions, reference-only contract docs, reference-only symbols, and non-contract reuse.
- Added per-stage repair prompts. If a gate fails, Gemini gets targeted blocker feedback and must repair the current stage before the pipeline moves on.
- Added stage-3 gate checks for invalid pytest patterns such as `pytest.skip`, `issubclass`, `isinstance`, `_underlying_class`, `__bases__`, mock-call oracles, broad import-state mutation, and hidden wrapper introspection.
- Added `--stage-repair-rounds` with default `1`.
- Added `initialize` to the generic reference-only symbol skip list to avoid that false positive.

## Prompt v10 Findings

Generated files before manual stop:

- Stage 1 analysis artifacts were generated, failed the new semantic gate, then were repaired.
- Stage 2 planning artifacts were generated, failed the gate, then were repaired once.
- The run was stopped during stage 3 because stage 2 repair had already made the run invalid by attempting a forbidden shell tool.

What improved:

- The stage 1 gate caught the recurring selector/subclass mistake immediately:
  - `BEHAVIOR_INVENTORY.md` had "Returns a valid class that is a subclass of `BaseRollout`".
  - `COVERAGE_MAP.md` had selector rows that asserted subclass relationships.
- The stage 1 repair succeeded. The repaired coverage map preserved downstream behaviors such as generation output, cache lifecycle, weight update, abort/resume, and manager behavior while removing the subclass gate blockers.
- The stage 2 gate caught the same class of mistakes when they reappeared in the test plan, plus `vLLMReplica` and `BucketedWeightSender` leakage.

Remaining problems:

- The stage 2 repair attempted `run_shell_command`, so the run became invalid even though the admin policy denied the tool.
- After one repair, the stage 2 plan still had mock-call oracles such as asserting that `wait_for_requests_to_drain`, `sleep`, and `wake_up` were called.
- The current staged runner continues after unrepaired stage blockers. This wastes time and can still generate invalid pytest.

Next pipeline change:

- Make unrepaired stage gate blockers a hard stop that writes a score/summary table instead of continuing to the next stage.
- Either increase `--stage-repair-rounds` for stage 2 or split "harness strategy" into its own stage before writing `TEST_PLAN.md`.

## Hard-Stop Test Findings

Runs:

- `testgen-gemini35flash-stage-hardstop-test-20260628-083221`
- `testgen-gemini35flash-stage-hardstop-repair-test-20260628-083606`
- `testgen-replay-hardstop-generated-static-20260628-085102`

Results:

- With `--stage-repair-rounds 0`, stage 1 stopped immediately after detecting selector/subclass oracles in `BEHAVIOR_INVENTORY.md` and `COVERAGE_MAP.md`. The runner wrote `score_table.md` and `validation_summary.json` with `status=stage_gate_failed` and did not continue to stage 2 or pytest.
- With the default `--stage-repair-rounds 1`, stage 1 failed then repaired successfully; stage 2 passed; stage 3 failed once on a doc-only reference-only module path and repaired successfully; the generated evaluator then ran pytest.
- The generated evaluator had 8 tests. Baseline scored `2/8`, reference scored `4/8`, so it is still invalid.
- Reference failures came from two concrete issues:
  - `HFModelConfig(path="/tmp/test_model_path", load_tokenizer=False)` still caused Transformers to load missing config metadata.
  - Neighboring backend resolution imported optional `sglang`, which is absent from the evaluator image.
- A replay of that generated evaluator after adding `/tmp/test_model_path` to the static audit now blocks before pytest:
  - baseline: `error: static validation blockers before pytest: /tmp/test_model_path: test_vllm_rollout_on_cpu.py`
  - reference: same error.

Changes made after the hard-stop tests:

- Stage gate failures now hard-stop the run and still write `score_table.md`, `validation_summary.json`, and `validation_summary_stage_gate_failure.json`.
- Added prompt guidance that model/tokenizer config constructors must not use arbitrary paths unless required local files are created.
- Added prompt guidance that neighboring-backend regression tests must not require optional backend packages missing from the evaluator image.
- Added `/tmp/test_model_path` to static hard blockers.

## Prompt v12 Pipeline Change

Reason for change:

- Broad rule-based blockers were too easy to overfit and could reject valid task-context contracts.
- Structural-looking patterns such as imports, class names, `isinstance`, monkeypatches, or reference symbols may be legitimate when they are part of a public boundary or are inferable from neighboring project paths.
- Candidate patch score diversity is useful as a diagnostic, but should not become a repair target.

Changes made:

- Rewrote `docs/unit-test-generator-prompt.md` as v12 around contract judgment rather than hard forbidden patterns.
- Added explicit `INFERRED_CONTRACTS.md` semantics: inferred names/signatures are allowed when justified by `/workspace/project` or `/workspace/AGENT_PROMPT.md`, not by reference code alone.
- Changed staged generation to four stages: contract analysis, behavior/harness design, test planning, evaluator implementation.
- Removed stage repair and final validation repair from the active pipeline.
- Static and semantic audits now write reports only. They do not hard-stop generation, block pytest, or trigger Gemini repair.
- The remaining acceptance signals are behavioral: reference must pass, baseline must fail, tests must exist, and analysis artifacts should explain coverage and harness choices.

## Prompt Versioning Update

Prompt revisions are now preserved under `docs/prompts/` instead of overwriting a single prompt file.

- `unit-test-generator-prompt-v1.md` preserves the user-edited outer-behavior prompt from commit `08710f0`.
- `unit-test-generator-prompt-v2.md` is copied from v1 with minimal edits: executable pytest tests and `run_tests.sh` are the only required outputs, the direct-testing language is softened, and coverage/audit notes are optional support material.

The v2 change removes hard requirements for coverage/audit documents. Those artifacts are useful when a generated evaluator has gaps or needs explanation, but they should not be treated as the core output. The runner now defaults to v2 and no longer fails a generated evaluator just because optional support documents are absent.

## Prompt v12 Smoke Result

Run:

- `testgen-gemini35flash-v12-no-repair-smoke-20260628-094804`

Result:

| evaluator | unit_tests | baseline | reference |
| --- | --- | --- | --- |
| generated | 38 | 0/6 (exit 2) | 0/6 (exit 2) |

Main problems:

- With hard blockers and repair disabled, Gemini still laundered reference implementation names into `INFERRED_CONTRACTS.md`.
- The generated evaluator directly imported reference-only implementation modules such as `verl.workers.rollout.vllm_rollout.utils` and `verl.workers.rollout.vllm_rollout.vllm_async_server`.
- The generated `conftest.py` stubbed `ray` as a root mock but did not provide import-compatible `ray.actor`, so both baseline and reference failed during pytest collection.
- The generated plan included subclass/interface checks and mock-call oracles despite the v12 prompt asking for contextual justification.

Changes made after v12:

- Rewrote the prompt as v13 with an explicit Project-First Rule.
- Added required `PROJECT_CONTRACT_SCAN.md`.
- Changed staged generation from four stages to five stages.
- Stage 1 now temporarily hides `/workspace/reference_project`, forcing the first contract scan to use only `/workspace/project` and `/workspace/AGENT_PROMPT.md`.
- Stage 2 restores the reference project and uses it only for behavior evidence and non-contract triage.
- Added collection reliability guidance: avoid top-level imports of missing capability modules unless project/prompt evidence requires the exact path, and make optional-dependency stubs import-compatible with submodules used by project imports.

## Prompt v13 Smoke Result

Run:

- `testgen-gemini35flash-v13-project-first-low-smoke-20260628-100413`

Result:

| evaluator | unit_tests | baseline | reference |
| --- | --- | --- | --- |
| generated | 29 | 8/29 (exit 1) | 24/29 (exit 1) |

What improved:

- The generated suite collected and ran against both baseline and reference.
- The baseline failed many missing-capability tests while the reference passed most tests, so the project-first stage helped compared with v12.

Remaining problems:

- Reference still failed 5 tests.
- The generated suite still used class inheritance, method presence, module-name matching, and `NotImplementedError` as primary behavior tests.
- `NON_CONTRACTS.md` items such as `build_cli_args_from_config` and `vLLMHttpServer` were reused in behavior inventory, coverage, and pytest code.
- Some tests used placeholder model paths such as `???`, triggering Hugging Face validation in the reference project.

Changes made after v13:

- Rewrote prompt as v14.
- Added explicit rule that `NON_CONTRACTS.md` items cannot be behavior IDs, coverage targets, pytest filenames, test names, imports, monkeypatch targets, or assertion subjects.
- Added an observable-behavior row schema for `BEHAVIOR_INVENTORY.md`, `COVERAGE_MAP.md`, and `TEST_PLAN.md`.
- Moved inheritance/method-presence/module-name/helper-existence checks into compatibility notes only; they should not be primary tests unless observed through a real project workflow.
- Added `???` to invalid placeholder model paths and recommended minimal plain objects when real HF metadata is irrelevant.

## Prompt v14 Smoke Result

Run:

- `testgen-gemini35flash-v14-observable-low-smoke-20260628-102601`

Result:

| evaluator | unit_tests | baseline | reference |
| --- | --- | --- | --- |
| generated | 7 | 0/7 (exit 1) | 0/7 (exit 1) |

Main problems:

- The generated evaluator shrank to only 7 tests.
- Reference failed all tests.
- Tests still imported and monkeypatched reference-only modules such as `verl.workers.rollout.vllm_rollout.vllm_async_server`.
- Optional dependency stubs were incomplete: the fake `ray` module lacked attributes used by project import-time type annotations, such as `ray.ObjectRef`.

Changes made after v14:

- Rewrote prompt as v15.
- Reference project is now visible only during Stage 2 reference triage. Stage 1 and Stages 3-5 temporarily hide `/workspace/reference_project`.
- Stage 3-5 prompts explicitly require using `REFERENCE_TEST_TRIAGE.md`, `PROJECT_CONTRACT_SCAN.md`, and planning artifacts rather than reopening reference implementation files.
- Collection guidance now requires optional-dependency stubs to include submodules and attributes used by project imports and type annotations.

## Prompt v15 Smoke Result

Run:

- `testgen-gemini35flash-v15-reference-hidden-low-smoke-20260628-104039`

Result:

| evaluator | unit_tests | baseline | reference |
| --- | --- | --- | --- |
| generated | 10 | 1/10 (exit 1) | 2/10 (exit 1) |

What improved:

- Direct pytest imports of reference-only vLLM implementation modules were reduced.
- Stage 1 no longer wrote an obvious subclass oracle into the project-first scan.

Remaining problems:

- Reference still failed 8/10 tests.
- `NON_CONTRACTS.md` items such as `vllm_async_server` still leaked into later docs.
- Optional vLLM stubs were still incomplete: reference import paths needed root-level symbols such as `vllm.LLM`.
- The model continued to use registry class resolution and subclass checks as primary tests.

Changes made after v15:

- Rewrote prompt as v16.
- Added required `CONTRACT_DECISION_TABLE.md`, classifying every candidate target as `TEST_TARGET`, `SUPPORTING_EVIDENCE`, or `NON_CONTRACT`.
- Added required `OPTIONAL_DEPENDENCY_STUBS.md`, recording root symbols, submodules, type-annotation attributes, and minimal behavior for faked external dependencies.
- Stage 3-5 now must use only `TEST_TARGET` rows and must implement external stubs from the stub plan.

## Prompt v16-v20 Smoke Results

| prompt/run | thinking | unit tests | baseline | reference | main outcome |
| --- | --- | ---: | --- | --- | --- |
| v16 `testgen-gemini35flash-v16-contract-table-low-smoke-20260628-105738` | LOW | 8 | 0/8 (exit 1) | 0/8 (exit 1) | Regressed: reference failed all tests; concrete reference classes were still used as targets. |
| v17 `testgen-gemini35flash-v17-public-behavior-targets-low-smoke-20260628-111400` | LOW | 8 | 0/8 (exit 1) | 0/8 (exit 1) | Added `PUBLIC_BEHAVIOR_TARGETS.md`; still failed because fake `vllm` missed `LLM` and tests kept subclass oracles. |
| v18 `testgen-gemini35flash-v18-import-audit-low-smoke-20260628-113006` | LOW | 8 | 0/8 (exit 1) | 0/8 (exit 1) | Added `IMPORT_COMPATIBILITY_AUDIT.md`; fixed root-symbol thinking but still missed `vllm.outputs` and guessed `RolloutConfig` kwargs. |
| v19 `testgen-gemini35flash-v19-api-import-low-smoke-20260628-114526` | LOW | 9 | 0/9 (exit 1) | 0/9 (exit 1) | Fixed `vllm.outputs` and removed bad `RolloutConfig` kwargs; still imported reference-only `build_cli_args_from_config` and missed `ray.experimental`. |
| v20 `testgen-gemini35flash-v20-public-entry-import-low-smoke-20260628-120047` | LOW | 7 | 0/6 (exit 2) | 0/6 (exit 2) | Regressed to collection errors; generated stub missed `FlexibleArgumentParser` definition and still imported reference-only helper. |
| v20 `testgen-gemini35flash-v20-public-entry-import-high-smoke-20260628-121606` | HIGH | 8 | 0/8 (exit 1) | 0/8 (exit 1) | HIGH thinking did not fix the core behavior: still used subclass oracles, reference-only helper imports, and guessed config usage. |

Current conclusion:

- Prompt-only iteration improved isolated issues, but Gemini still repeatedly turns reference evidence into implementation contracts.
- The most persistent failures are:
  - reference-only helper/module imports in pytest, especially `verl.workers.rollout.vllm_rollout.utils.build_cli_args_from_config`,
  - structural oracle drift (`issubclass`, class names, registry resolution as the main assertion),
  - incomplete recursive import audits for transitive public project imports such as `ray.experimental.state.api`,
  - unsafe broad fallback stubs that can corrupt unrelated library imports, such as torch/tensordict inspection.
- Raising Gemini thinking from LOW to HIGH did not materially improve this task.

Next pipeline direction:

- Stop adding more prose constraints to the same five-stage flow.
- Replace the current staged flow with a stricter artifact schema where stages exchange machine-checkable tables rather than free-form markdown narratives.
- Add a generator self-check stage that rewrites its own plan before pytest generation, but only using task inputs and static artifact inspection, not candidate patch scores.
- Keep rule-based audits diagnostic unless the violation is globally invalid, but expose those diagnostics directly to the generator in the next stage as review feedback rather than hidden repair logic.

## Codex GPT-5.5 Staged Smoke Result

Run:

- `testgen-codex-gpt55-v20-staged-smoke-20260628-202019`

Invocation:

- `python scripts/run_codex_test_generator.py --run-name codex-gpt55-v20-staged-smoke --num-runs 1 --max-candidate-patches 0 --test-timeout-sec 240 --timeout-sec 1800`

Result:

| generator | unit tests | baseline | reference |
| --- | ---: | --- | --- |
| Codex GPT-5.5 staged | 7 | 0/7 (exit 1) | 0/7 (exit 1) |

What improved over Gemini:

- Codex produced much stronger intermediate analysis artifacts: project-first contract scan, reference test triage, public behavior targets, harness strategy, coverage map, test plan, and failure-mode audit.
- The plan correctly identified broad behavior families that Gemini often missed: replica lifecycle, tokenized client routing, agent-loop/DataProto conversion, rollout logprob consumers, weight refresh, checkpoint lifecycle, standalone generation, and existing-backend preservation.
- The generated pytest avoided direct imports of the reference-only vLLM implementation package and did not use obvious subclass/module-name oracles as the only assertion.

Why the generated evaluator is still unusable:

- Reference failed every generated test.
- All original reference failures hit the same harness problem: the fake external `vllm` package missed transitive LoRA submodules required by the reference selector import path, first `vllm.lora.lora_model` / `vllm.lora.models`.
- A manual diagnostic patch adding those two modules exposed the next missing transitive stub, `vllm.lora.utils.get_adapter_absolute_path`, so this was not a one-line typo. The generated import-compatibility harness was incomplete.
- The implementation shrank the Stage 4 plan from 16 planned test intents to 7 pytest functions, leaving important planned coverage unimplemented: engine kwargs propagation, direct sampling-param propagation, bounded PPO/GRPO dataflow, checkpoint-worker adapter update, fully async partial continuation, and teacher prompt-logprob workflow.

Interpretation:

- This run does not prove the problem is only Gemini. Codex understood the task and coverage goals much better than Gemini, but the current prompt/pipeline still did not force a reference-passing executable evaluator.
- The strongest current signal is that model capability matters for analysis quality, while the evaluator-generation pipeline still needs a better executable-harness verification loop, especially for recursive optional-dependency stubs and for ensuring planned coverage is actually implemented.

## Prompt v21-v22 Single-Prompt Execution Results

Changes after the Codex run:

- Replaced the five-stage prompt with a single coding-agent prompt.
- Removed the instruction that prohibited shell/pytest.
- Updated the Gemini runner policy to allow `run_shell_command`.
- Made reference validation an explicit hard stop: the generator must run `PROJECT_UNDER_TEST=/workspace/reference_project /workspace/generated_tests/run_tests.sh` and fix the evaluator until it exits 0.
- Added a second hard stop in v22: the incomplete project must collect tests and fail during test execution; collection/import errors are evaluator bugs.
- Simplified quality gates to match the single-prompt workflow: generated pytest exists, reference passes, baseline fails, and required summary files exist.

Runs:

| prompt/run | unit tests | baseline | reference | outcome |
| --- | ---: | --- | --- | --- |
| v21 `testgen-gemini35flash-v21-single-exec-high-20260628-210313` | 31 | 0/9 (exit 2) | 31/31 (exit 0) | Fixed reference validation. Baseline failed during collection because tests had top-level imports from missing vLLM modules. |
| v22 `testgen-gemini35flash-v22-single-exec-high-baseline-collect-20260628-211158` | 9 | 0/9 (exit 1) | 9/9 (exit 0) | Fixed baseline collection: all tests collect and fail during execution on the incomplete project. |

What improved:

- Gemini now actually runs local commands inside the Docker workspace.
- It used the reference project as an executable oracle and repaired the evaluator from failures.
- v21 reached `31 passed` on reference after one repair.
- v22 reached the stronger validation target: reference passes and baseline collects then fails with ordinary test failures.

Remaining serious problem:

- v22 still tests reference-only internals. The generated pytest imports or targets `verl.workers.rollout.vllm_rollout.*`, `build_cli_args_from_config`, `get_vllm_max_lora_rank`, `extract_prompt_logprobs`, `BucketedWeightSender`, and `vLLMReplica`.
- That means the evaluator is still not behavior-only enough. It proves the new execution loop works, but it is not yet a good benchmark evaluator.
- The next prompt revision should make reference-only module/helper usage a hard self-check, while still allowing the generator to run pytest. The intended failure for the baseline should be through public selectors/callers that exist in `/workspace/project`, not through direct requirements on the reference implementation package layout.

## Runner Ownership Correction

The evaluator generator should not own the test execution entrypoint.

The earlier prompt and runner required the model to emit `generated_tests/run_tests.sh`. That made validation depend on model-authored harness behavior. In the latest Gemini run, the generated shell script ignored the runner's `PROJECT_UNDER_TEST` environment variable and selected the target project from a positional argument. The runner therefore reported the reference result using the wrong target.

The pipeline has been corrected:

- the generator is instructed to write pytest tests and optional helper files only;
- `run_tests.sh` is no longer required and is ignored if generated;
- the runner always executes `python -m pytest -q /workspace/generated_tests` itself;
- the runner sets `PYTHONPATH=/workspace/<target>:/workspace/generated_tests` and `PROJECT_UNDER_TEST=/workspace/<target>` for each target.

With this design, every solution is tested through the same benchmark-owned interface, and generated test quality is no longer confounded with generated harness quality.

Extra known-good validation targets are now first-class runner targets. For example, a structurally different counter reference can be passed as:

```bash
--validation-target counter_reference=/path/to/alt_reference_project
```

These targets are hidden from the generator and are expected to pass after generation. They make structure-binding failures visible even when the official reference passes.

For the current VERL/vLLM benchmark, the strict counter reference is now auto-added by default when `/mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project` exists. Future evaluator-generation reports should always include:

- initial incomplete codebase pass rate;
- official reference pass rate;
- counter-reference pass rate;
- pass rates for each candidate/baseline variant.

## Prompt v3 Small Revision

`unit-test-generator-prompt-v3.md` adds only a narrow alternative-implementation litmus test and a final import/name self-check on top of v2. The goal is to reduce tests that pass only because the official reference uses a particular internal module path, class name, helper name, or file layout.

Attempted run `testgen-gemini35flash-v3-alt-litmus-20260629-085511` was inconclusive: Gemini repeatedly returned 503/high-demand and fetch failures, and the run was interrupted before the patched CLI's indefinite retry behavior could complete, so no score table was available. Gemini runner defaults now leave generation timeout disabled (`--timeout-sec 0`) so transient API failures are handled by the patched CLI retry loop unless a caller explicitly opts into a hard timeout.

## v4 Two-Stage Prompt Design

`unit-test-generator-prompt-v4-stage1.md` and `unit-test-generator-prompt-v4-stage2.md` split the v3 prompt without changing the core evaluator contract. Stage 1 hides `/workspace/reference_project` and asks the model to write the first pytest evaluator from only the candidate prompt and incomplete project, while explicitly encouraging local experiments against existing CPU tests, neighboring backends, public configs, and downstream consumers. Stage 2 restores `/workspace/reference_project` and resumes the same Gemini session with an explicit UUID session id, then asks it to refine the existing evaluator using the reference only as an executable behavior oracle.

This design is meant to reduce reference-structure contamination. A behavior may be added in Stage 2 only when it is inferable from `/workspace/AGENT_PROMPT.md`, `/workspace/project`, or neighboring public behavior; reference-only module paths, class names, helper functions, actors, and file layouts remain non-contracts. The runner exposes this as `--generation-mode two-stage`; the old single-stage and five-stage modes remain available.

The earlier evaluator-generation runs used a short stdin wrapper that told Gemini to read a prompt file. That was weaker than directly sending the full evaluator-generation prompt as the user message, because file-read instructions can be treated as lower-priority context or skipped. The runner now writes the full prompt into the stdin prompt file for single-stage, staged, and two-stage generation, while also saving a copy under `/workspace/EVALUATOR_GENERATOR_PROMPT*.md` for debugging.

The two-stage runner now keeps one Docker container alive across both Gemini CLI invocations. Stage 1 runs with `/workspace/reference_project` hidden, using `--session-id <uuid>` and the full Stage 1 prompt on stdin. The host then restores `/workspace/reference_project` into the mounted workspace, and Stage 2 runs inside the same container with `--resume <uuid>` and the full Stage 2 prompt on stdin. This preserves container-local state while keeping the Stage 1 reference boundary clean.

Evaluator generation should now be launched through `./run.sh <agent_id>` rather than ad hoc inline command strings. `run.sh` only generates pytest files under the created run workspace. Scoring those generated tests is a separate step through `./score.sh /path/to/generated_tests`. The stable agent registry is documented in `docs/evaluator-generation-agents.md`; agent `4` is the current two-stage direct-prompt persistent-container evaluator generator.

First attempted `./run.sh 4` run: `/mnt/data/projects/verl-vllm-testgen-runs/testgen-evalgen-agent-4-gemini-3.5-flash-20260629-112128`. Stage 1 generated four tests. Stage 2 started, read reference vLLM files, then Gemini emitted `Invalid stream: The model returned an empty response or malformed tool call` with final stream status `error`. The CLI process still exited zero, so the old runner incorrectly proceeded to scoring. The resulting generated tests failed on baseline, official reference, counter reference, and all candidate variants with the same 4/4 failure pattern, so this run is not a valid two-stage-refined evaluator result.

The runner now treats Gemini stream `type=error` or final result `status=error` as generation failure. The standard `run.sh` entrypoint passes `--skip-validation`, so generation and scoring are not mixed. Use `score.sh` for reproducible re-scoring without invoking Gemini.

Second `./run.sh 4` rerun: `/mnt/data/projects/verl-vllm-testgen-runs/testgen-evalgen-agent-4-gemini-3.5-flash-20260629-114139`. This run completed successfully at the Gemini stream level:

- Stage 1 stream status: success, 41 tool calls, 611s.
- Stage 2 stream status: success, 54 tool calls, 1667s.
- No `Invalid stream` / malformed tool call error occurred. The earlier stream error was therefore not reproduced by this rerun.
- The long silent periods in Stage 2 were Gemini model calls, not local pytest or shell hangs; the stream later resumed and produced more tool calls.

Generated evaluator result after independent `score.sh`:

| evaluator | unit tests | baseline | reference | counter reference |
| --- | ---: | --- | --- | --- |
| agent 4 rerun | 5 | 1/5 (exit 1) | 5/5 (exit 0) | 3/5 (exit 1) |

Candidate patch scores ranged from 1/5 to 3/5, with most candidates at 3/5. This gives some separation but the evaluator is invalid because the counter reference fails.

The counter-reference failures and static audit show the root problem clearly: the generated tests still import or patch reference-only internals such as `verl.workers.rollout.vllm_rollout.vllm_async_server`, `vLLMHttpServer`, `vLLMReplica`, and `BucketedWeightSender`. These are official-reference implementation details, not task-level behavior contracts. The tests pass the official reference because they match its layout, but reject a structurally different implementation that provides the intended capability.

Operational conclusion:

- The rerun did not confirm a deterministic Gemini CLI stream failure.
- The runner-side stream-error detection is still necessary because the previous run did produce a final stream `status=error` while the CLI process exited zero.
- The generation workflow remains too slow and too prone to reference-structure contamination. Future prompt changes should focus on preventing direct reference-only module imports/patches while preserving reference execution as an oracle.

Third `./run.sh 4` rerun after adding a mild coverage-size hint and generation retry support: `/mnt/data/projects/verl-vllm-testgen-runs/testgen-evalgen-agent-4-gemini-3.5-flash-20260629-220154`.

- Stage 1 stream status: success, 47 tool calls, 205s.
- Stage 2 stream status: success, 58 tool calls, 1024s.
- No Gemini stream error occurred, so the retry path was not exercised.
- Final evaluator still contained only 7 pytest nodes.

Generated evaluator score:

| evaluator | unit tests | baseline | reference | counter reference | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent 4 third rerun | 7 | 0/7 (exit 1) | 7/7 (exit 0) | 4/7 (exit 1) | 4/7 | 4/7 | 4/7 | 4/7 | 4/7 | 4/7 | 4/7 | 4/7 | 0/7 | 4/7 | 2/7 |

Main findings:

- The coverage hint slightly increased the test count from 5 to 7, but this is still far below the expected coverage for this task.
- The evaluator again overfit to official-reference internals. Static audit found direct reference-only paths and symbols including `verl.workers.rollout.vllm_rollout.vllm_async_server`, `verl.workers.rollout.vllm_rollout.vllm_rollout.BucketedWeightSender`, `vLLMHttpServer`, `vLLMReplica`, and `extract_prompt_logprobs`.
- Counter reference failed exactly on the tests that hard-coded the official reference package layout. This confirms the problem is evaluator structure-binding, not merely an incomplete alternate implementation.
- The candidate score spread is not trustworthy: most candidates get 4/7 because they satisfy registry/lifecycle smoke checks but fail the reference-only adapter/generation tests.

Prompt implication:

- The two-stage prompt still lets Stage 2 convert reference-observed implementation details into executable test targets. The next change should keep reference execution as an oracle, but explicitly require every import, patch target, and called symbol in pytest to be available or inferable from `/workspace/project` or `/workspace/AGENT_PROMPT.md`, not from `/workspace/reference_project`.

## Codex GPT-5.5 Two-Stage v4 Result

A Codex runner was added for the same v4 two-stage prompt shape used by Gemini agent 4. Stage 1 hides `/workspace/reference_project`; Stage 2 restores it and resumes the same Codex thread in the same persistent Docker container.

Run:

- Generation: `/mnt/data/projects/verl-vllm-testgen-runs/testgen-evalgen-agent-4-codex-gpt55-20260629-223053`
- Score: `/mnt/data/projects/verl-vllm-testgen-runs/testgen-score-generated-tests-20260629-224036`

Generated evaluator score:

| evaluator | pytest nodes | baseline | reference | counter reference | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Codex GPT-5.5 two-stage v4 | 8 | 3/8 (exit 1) | 8/8 (exit 0) | 8/8 (exit 0) | 6/8 | 6/8 | 6/8 | 6/8 | 6/8 | 6/8 | 6/8 | 5/8 | 3/8 | 6/8 | 6/8 |

Main findings:

- This is the first generated evaluator in this line of experiments that passes both the official reference and the structurally different counter reference.
- It avoids direct official-reference `verl.workers.rollout.vllm_rollout.*` imports and patch targets. `reference_only_dotted_paths` was empty.
- It exercises more realistic public behavior than the recent Gemini outputs: normal Hydra rollout config selection, async rollout client token/logprob return behavior, load-balancer release on generation failure, checkpoint-worker weight update path, and `LLMServerManager` startup for hybrid and standalone vLLM replicas.
- The score spread is still limited: most imperfect candidates score 6/8, one scores 5/8, and the weakest remains at 3/8. This is useful but still much less discriminating than the hand-written CPU evaluator.
- The static audit reports `sys.meta_path` and `MetaPathFinder` in `conftest.py`; here they are used to provide an import-compatible fake external `vllm` package, not to rewrite project internals. This should be reviewed but did not cause counter-reference failure.
- `reference_only_symbols` reports `_FakeServer`, but inspection shows it is a local fake class in the generated test file whose name collides with a reference-test helper. This appears to be an audit false positive, not reference leakage.

Interpretation:

- Codex is materially better than Gemini for this evaluator-generation prompt: it followed the project-first boundary better, used reference execution to repair evaluator bugs, and did not overfit to the official reference package layout.
- The remaining issue is coverage depth. Eight pytest nodes are not enough for the full vLLM integration task, and candidate scores show that many partial implementations still pass most of the suite. The next prompt/agent improvement should push Codex toward broader independent behavior targets without sacrificing the counter-reference property.

## Reference Coverage Diagnostic Route

Added a separate coverage diagnostic entrypoint instead of mixing coverage into `score.sh`. `score.sh` remains the pass/fail evaluator scorer. `coverage.sh` runs a generated pytest suite against one or more known-good references and reports how much of each reference's implementation diff is activated.

The diagnostic compares the incomplete task project against each reference target, computes changed Python lines, then runs pytest inside the CPU test Docker image under Python's stdlib `trace` module. It reports changed-line coverage plus rough changed function/class activation. No extra package such as `coverage.py` is required.

Default references:

- `official_reference`: `/mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0`
- `counter_reference`: `/mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project`

Usage:

```bash
./coverage.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

Additional references can be appended as `label=/path/to/reference`. For the VERL/vLLM task, use this focused denominator to avoid unrelated release-diff noise:

```bash
COVERAGE_INCLUDE_REGEX='^verl/workers/rollout/,^verl/workers/config/rollout.py$,^verl/third_party/vllm/,^verl/utils/vllm/,^verl/utils/modelopt/,^verl/utils/qat/,^verl/utils/profiler/' \
  ./coverage.sh /mnt/data/projects/verl-vllm-testgen-runs/<run>/workspace/generated_tests
```

The same route can diagnose a single pytest node by setting `PYTEST_TARGETS` to a comma-separated list of files or node ids relative to `generated_tests`.

Codex GPT-5.5 two-stage v4 focused diagnostic:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-codex-vllm-focused-classfix-20260629-225501`
- Report: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-codex-vllm-focused-classfix-20260629-225501/coverage_report.md`

| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_reference | 0 | 25 | 3721 | 326 | 8.8% | 2/189 | 1.1% | 9/17 | 52.9% |
| counter_reference | 0 | 22 | 3710 | 282 | 7.6% | 2/191 | 1.0% | 8/14 | 57.1% |

Interpretation:

- The Codex-generated evaluator is behaviorally much healthier than the recent Gemini-generated evaluators because it passes both official and counter references.
- The coverage diagnostic quantifies the remaining weakness: it activates less than 10% of the task-focused changed implementation lines on both references.
- The current denominator still includes optional or GPU-heavy vLLM/modelopt/QAT/profiler paths. This is useful as a broad diagnostic, but future task metadata may need tiers such as CPU-must-cover versus GPU/optional-gap to make coverage numbers easier to interpret.

## Hand-Written CPU Evaluator Status

The hand-written evaluator lives at `/mnt/data/projects/verl-vllm-evaluator-generation-workbench-publish/eval_tests`. It contains 55 CPU pytest cases and is the source of the historical `manual_results` score rows.

Historical candidate scores from `/mnt/data/projects/verl-vllm-testgen-runs/testgen-score-v5-all-candidates-20260628-002459/validation_summary.json`:

| target | score |
| --- | ---: |
| baseline | 25/55 |
| official reference | 55/55 |
| C1 `20260626-081247-gemini31pro-preview-candidate2` | 33/55 |
| C2 `20260626-091059-gemini31pro-v2-candidate1` | 39/55 |
| C3 `20260626-091100-gemini31pro-v2-candidate2` | 36/55 |
| C4 `20260626-091100-gemini31pro-v2-candidate3` | 39/55 |
| C5 `20260626-091100-gemini31pro-v2-candidate4` | 39/55 |
| C6 `20260626-093319-gemini35flash-v3-candidate1` | 40/55 |
| C7 `20260626-093319-gemini35flash-v3-candidate2` | 38/55 |
| C8 `20260626-093319-gemini35flash-v3-candidate4` | 37/55 |
| C9 `20260627-044228-gemini35flash-parity-candidate1` | 27/55 |
| C10 `20260627-044228-gemini35flash-parity-candidate2` | 38/55 |
| C11 `20260627-044228-gemini35flash-parity-candidate3` | 25/55 |

Counter-reference check added later:

- Command: `TEST_IMAGE=verl-vllm-rollout-benchmark:cpu-tests /mnt/data/projects/verl-vllm-evaluator-generation-workbench-publish/eval_tests/run_cpu_tests.sh /mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project`
- Result: `42/55`, exit 1.
- The 13 failures all come from generation-contract tests that import `verl.workers.rollout.vllm_rollout` through `vllm_eval_helpers.py`. This is official-reference package-layout binding, so this hand-written evaluator is not counter-reference compatible.

Focused coverage after fixing `coverage.sh` to run tests from each reference checkout root:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-cwdfix-20260629-233610`
- Report: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-cwdfix-20260629-233610/coverage_report.md`

| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_reference | 0 | 25 | 3721 | 439 | 11.8% | 12/189 | 6.3% | 10/17 | 58.8% |
| counter_reference | 1 | 22 | 3710 | 283 | 7.6% | 2/191 | 1.0% | 8/14 | 57.1% |

Interpretation:

- It remains the strongest discriminator among the evaluated suites by candidate score spread.
- It covers more official-reference vLLM code than the Codex-generated evaluator, especially `vllm_async_server.py` and some lifecycle/generation paths.
- It is not a valid behavior-only oracle under the newer counter-reference criterion because part of the suite requires the official `verl.workers.rollout.vllm_rollout` package layout.

## Hand-Written CPU Evaluator v2

Updated `/mnt/data/projects/verl-vllm-evaluator-generation-workbench-publish/eval_tests` to remove the official-reference package-layout binding from generation-server discovery. The helper now starts from `get_rollout_replica_class("vllm")`, scans the implementation package that registered replica class comes from, and only accepts generation classes defined in those candidate modules. This preserves the behavior target while allowing structurally different implementations such as the counter reference.

Added six behavior-level server lifecycle tests covering wake-up/cache reset, sleep, explicit KV-cache clear, request drain, empty abort-all, and missing abort-request handling. The suite now has 61 CPU pytest cases.

Reference checks:

| target | score |
| --- | ---: |
| baseline | 25/61 |
| official reference | 61/61 |
| counter reference | 61/61 |

Historical candidate scores after the v2 update:

| target | score |
| --- | ---: |
| C1 `20260626-081247-gemini31pro-preview-candidate2` | 33/61 |
| C2 `20260626-091059-gemini31pro-v2-candidate1` | 40/61 |
| C3 `20260626-091100-gemini31pro-v2-candidate2` | 37/61 |
| C4 `20260626-091100-gemini31pro-v2-candidate3` | 41/61 |
| C5 `20260626-091100-gemini31pro-v2-candidate4` | 40/61 |
| C6 `20260626-093319-gemini35flash-v3-candidate1` | 40/61 |
| C7 `20260626-093319-gemini35flash-v3-candidate2` | 38/61 |
| C8 `20260626-093319-gemini35flash-v3-candidate4` | 39/61 |
| C9 `20260627-044228-gemini35flash-parity-candidate1` | 25/61 |
| C10 `20260627-044228-gemini35flash-parity-candidate2` | 39/61 |
| C11 `20260627-044228-gemini35flash-parity-candidate3` | 25/61 |

Candidate score run: `/mnt/data/projects/verl-vllm-testgen-runs/manual-suite-v2-candidate-score-20260630-011302`.

Focused coverage:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-v2-20260630-011036`
- Report: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-v2-20260630-011036/coverage_report.md`

| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_reference | 0 | 25 | 3721 | 467 | 12.6% | 20/189 | 10.6% | 10/17 | 58.8% |
| counter_reference | 0 | 22 | 3710 | 377 | 10.2% | 8/191 | 4.2% | 8/14 | 57.1% |

Interpretation:

- v2 fixes the major behavior-only flaw: both official and counter references pass.
- Coverage improves from official `11.8%` to `12.6%` and counter `7.6%` to `10.2%` under the same focused denominator.
- The denominator still includes optional/GPU-heavy modelopt/QAT/profiler code, so the absolute percentage remains low. The meaningful improvement is concentrated in the CPU-reachable generation request-flow and server lifecycle paths.

## Hand-Written CPU Evaluator v3

Expanded `/mnt/data/projects/verl-vllm-evaluator-generation-workbench-publish/eval_tests` from 61 to 78 CPU pytest cases while keeping the generation-server discovery behavior-only and counter-reference compatible.

Added coverage for:

- default response-budget behavior when requests omit `max_tokens`;
- explicit and default wake-up tags;
- hybrid, colocated, standalone, LoRA-adapter, full-weight, free-cache-disabled, and non-master lifecycle branches;
- logprob disabled requests;
- backend abort finish reasons;
- LoRA request suppression until the adapter is loaded;
- abort-all/resume/profile/abort-request server behavior.

Reference checks:

| target | score |
| --- | ---: |
| baseline | 25/78 |
| official reference | 78/78 |
| counter reference | 78/78 |

Focused coverage with the original stdlib `trace` runner:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-v4-20260630-014536`
- Result: official `13.4%`, counter `10.2%`.

The unchanged counter score exposed a diagnostic bug: stdlib `trace.Trace` was undercounting async coroutine bodies. In particular, the counter-reference service class lived in `runtime.py`, tests executed its async methods, but the report still showed `runtime.py` as `0/759`.

## Async-Aware Coverage Diagnostic

Replaced the stdlib `trace.Trace` runner in `scripts/diagnose_reference_coverage.py` with a lightweight `sys.settrace`/`threading.settrace` runner that only enables line counting for frames whose filename is under `PROJECT_UNDER_TEST`. This keeps tracing practical while counting async method bodies.

Async-aware focused coverage for hand-written v3:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-v5-async-aware-20260630-015101`
- Report: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-vllm-focused-v5-async-aware-20260630-015101/coverage_report.md`

| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_reference | 0 | 25 | 3721 | 617 | 16.6% | 26/189 | 13.8% | 12/17 | 70.6% |
| counter_reference | 0 | 22 | 3710 | 644 | 17.4% | 29/191 | 15.2% | 11/14 | 78.6% |

Focused group breakdown:

| reference | core rollout | vLLM utils | quant/modelopt/QAT | profiler |
| --- | ---: | ---: | ---: | ---: |
| official_reference | 446/1595 (28.0%) | 116/920 (12.6%) | 50/1152 (4.3%) | 3/37 (8.1%) |
| counter_reference | 474/1612 (29.4%) | 116/920 (12.6%) | 50/1149 (4.4%) | 3/27 (11.1%) |

Interpretation:

- The earlier `10%` counter-reference coverage number was partly a tooling artifact, not a CPU-testability upper bound.
- The CPU suite now gives materially better signal on core rollout behavior, but it still does not cover most quantization/modelopt/QAT logic. Those paths are largely optional or tied to real vLLM weight loading, GPU kernels, NPU behavior, or heavy patching flows.
- Further CPU gains should target behavior-level tests for launch argument assembly, profiler bridging, and utility wrappers only if they can be exercised through stable runtime behavior rather than direct helper-name assertions.

## Hand-Written CPU Evaluator v4

Expanded `/mnt/data/projects/verl-vllm-evaluator-generation-workbench-publish/eval_tests` from 78 to 83 CPU pytest cases. The additions stay at public behavior boundaries:

- Qwen2.5-VL consecutive visual-token deduplication before backend generation.
- vLLM server actor address methods and `collective_rpc` forwarding.
- vLLM async rollout adapter registry selection.
- vLLM async rollout adapter `resume(["weights"])` and `release()` lifecycle forwarding through an existing server handle.

Reference checks:

| target | score |
| --- | ---: |
| baseline | 25/83 |
| official reference | 83/83 |
| counter reference | 83/83 |

Focused async-aware coverage:

- Run: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-handwritten-vllm-focused-v9-adapter-lifecycle-20260630-023551`
- Report: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-handwritten-vllm-focused-v9-adapter-lifecycle-20260630-023551/coverage_report.md`

| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_reference | 0 | 25 | 3721 | 634 | 17.0% | 34/189 | 18.0% | 12/17 | 70.6% |
| counter_reference | 0 | 22 | 3710 | 661 | 17.8% | 36/191 | 18.8% | 11/14 | 78.6% |

Interpretation:

- The coverage route is now correct for async/threaded CPU execution and the report explicitly states that it measures changed implementation lines, not whole-repository coverage.
- The absolute line percentage is still low because the focused denominator intentionally includes large optional/platform-specific reference files: `modelopt/vllm_modelopt_patch.py`, `utils/vllm/vllm_fp8_utils.py`, `utils/qat/vllm_patch.py`, `utils/vllm/npu_vllm_patch.py`, and weight-transfer internals. These files account for most uncovered functions.
- Not all uncovered code is impossible on CPU, but much of the remaining uncovered code would require direct helper/transport/quantization internals rather than stable public behavior. The next safe CPU targets are limited: profiler argument bridging if exercised through public profiler controls, rollout adapter update-weight effects if a behavior-level fake transport can be used without patching reference-only classes, and maybe shared-memory weight transfer as a separate lower-level conformance test.
- The current suite already covers the highest-value CPU-reachable behavior surface: backend selection, manager/client generation, sampling/logprob/prompt-logprob semantics, multimodal payload and Qwen visual-token handling, LoRA request gating, routing/MTP metadata, lifecycle/cache/abort/profile operations, rollout adapter lifecycle, checkpoint update ordering, fully async resume behavior, and downstream rollout consumers.

## Hand-Written GPU Evaluator v1

Added a separate GPU evaluator tier under `eval_tests/gpu_tests` plus `eval_tests/run_gpu_tests.sh`. This tier is separate from CPU scoring and requires `VERL_GPU_TEST_MODEL=/path/to/local/model`.

The first GPU test launches a real standalone rollout server through the public `LLMServerManager`/client path with `rollout.name=vllm`. It checks behavior that CPU fakes cannot prove:

- real token generation through the vLLM backend;
- response logprobs and prompt logprobs from a real model;
- explicit `max_tokens`, `max_new_tokens`, default response-budget handling, parallel requests, and full-context rejection;
- global load-balancer drain state after generation;
- abort-all/resume, KV cache lifecycle hooks, and profile hooks against a live rollout replica;
- fully-async rollout client output contract on top of the same live backend.

This tier is designed as a fast GPU smoke test. The runner now creates a tiny randomly initialized Qwen2-compatible HF causal LM when `VERL_GPU_TEST_MODEL` is unset. If a caller overrides the model, it should still use the smallest local vLLM-compatible causal language model that exercises the backend, preferably sub-1B parameters. The defaults are intentionally tiny: one GPU, TP=1, `max_model_len=40`, `max_num_seqs=2`, `gpu_memory_utilization=0.05`, and two generated tokens per request. Larger models, multi-GPU parallelism, long-context generation, LoRA update correctness, and real training-step GPU checks should live in a separate heavier GPU tier.

Validation performed without GPU:

- `python -m py_compile` for the GPU test file;
- `bash -n` for `run_gpu_tests.sh`;
- Docker dry-run in the official test image with no `VERL_GPU_TEST_MODEL`, which cleanly reports `1 skipped`;
- Docker compose check for the Hydra config override helper.

Real GPU checks with the auto-generated tiny random Qwen2 model:

| target | result |
| --- | --- |
| official reference | `1 passed`, 115s, sampled GPU memory about 4.7GB |
| counter reference | `1 passed`, 115s |
| initial incomplete codebase | `1 failed`, fails at missing `rollout.name=vllm` registry entry |

The expected real run command is:

```bash
eval_tests/run_gpu_tests.sh /path/to/candidate/verl
```

## Hand-Written Evaluator Scoreboard With GPU v1

Pass-rate summary:

| suite | pytest nodes | baseline | official reference | counter reference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 83 | 25/83 | 83/83 | 83/83 |
| GPU | 1 | 0/1 | 1/1 | 1/1 |
| CPU + GPU | 84 | 25/84 | 84/84 | 84/84 |

Focused coverage summary:

| suite | official line cov | official funcs | official classes | counter line cov | counter funcs | counter classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 17.0% | 34/189 | 12/17 | 17.8% | 36/191 | 11/14 |
| GPU | 13.3% | 12/189 | 12/17 | 13.9% | 11/191 | 11/14 |
| CPU + GPU | 18.5% | 37/189 | 12/17 | 19.3% | 39/191 | 11/14 |

Coverage runs:

- CPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-handwritten-vllm-focused-v9-adapter-lifecycle-20260630-023551`
- GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-gpu-v1-20260630-031836`
- CPU + GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-combined-v1-20260630-032234`

Interpretation:

- GPU v1 improves the combined focused line coverage from CPU-only official `17.0%` to `18.5%`, and counter `17.8%` to `19.3%`.
- The GPU suite mainly increases real `vllm_async_server` / counter `runtime` activation because it launches a live vLLM backend rather than scripted fakes.
- The focused denominator still includes optional or specialized paths such as modelopt, QAT, NPU patches, and lower-level weight-transfer internals, so the absolute percentage remains intentionally conservative.

## Hand-Written Evaluator v5 / GPU v2

Expanded the hand-written evaluator from 83 to 86 CPU pytest cases and from 1 to 6 GPU pytest cases.

CPU additions:

- async rollout adapter lazy server-handle lookup through the existing Ray actor naming convention;
- nonzero rollout-rank lifecycle gating;
- blocking and non-blocking collective RPC forwarding through the adapter.

GPU additions:

- split the original live-vLLM smoke into 6 pytest nodes sharing one module-scoped live server fixture;
- added explicit OpenAI completion endpoint coverage;
- added prompt-logprob-zero, `ignore_eos`, default-budget, parallel request, missing abort, and lifecycle checks with pass-rate granularity.

Pass-rate summary:

| suite | pytest nodes | baseline | official reference | counter reference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 86 | 25/86 | 86/86 | 86/86 |
| GPU | 6 | 0/6 | 6/6 | 6/6 |
| CPU + GPU | 92 | 25/92 | 92/92 | 92/92 |

Focused coverage summary:

| suite | official line cov | official funcs | official classes | counter line cov | counter funcs | counter classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 17.2% | 36/189 | 12/17 | 18.0% | 38/191 | 11/14 |
| GPU | 13.4% | 13/189 | 12/17 | 14.0% | 12/191 | 11/14 |
| CPU + GPU | 18.7% | 39/189 | 12/17 | 19.5% | 41/191 | 11/14 |

Coverage runs:

- CPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-cpu-v3-20260630-040811`
- GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-gpu-v2-20260630-035101`
- CPU + GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-combined-v3-20260630-041013`

Interpretation:

- The CPU adapter additions modestly improve adapter coverage: official `vllm_rollout.py` and counter `adapter.py` are now at `57/167` changed lines (`34.1%`).
- The live GPU split improves pass-rate diagnosability and adds a real OpenAI HTTP path, but does not materially raise total line coverage because the same live server startup dominates activation.
- The remaining uncovered denominator is now clearly dominated by optional/specialized paths: `modelopt/vllm_modelopt_patch.py`, `utils/qat/vllm_patch.py`, `utils/vllm/vllm_fp8_utils.py`, `utils/vllm/npu_vllm_patch.py`, and lower-level weight-transfer internals. Covering those without over-constraining implementation shape likely requires a separate heavy/special GPU tier for real weight sync, LoRA adapter sync, quantized/QAT model loading, and platform-specific behavior.

## Hand-Written Evaluator v6 / CPU Contract v4

Expanded the hand-written CPU evaluator from 86 to 90 pytest cases.

CPU additions:

- prompt-logprob behavior for `prompt_logprobs=0`, where the backend returns sampled-token prompt logprobs;
- prompt-logprob rank ordering, including ignoring backend entries outside the requested top-k;
- routed-expert outputs stay hidden unless routing replay is enabled;
- abort-all lifecycle can leave cache intact when `reset_prefix_cache=False`.

Pass-rate summary:

| suite | pytest nodes | baseline | official reference | counter reference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 90 | 25/90 | 90/90 | 90/90 |

Broad changed-code coverage diagnostic:

| reference | changed lines | covered lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official reference | 4955 | 698 | 14.1% | 38/255 | 14.9% | 17/32 | 53.1% |
| counter reference | 3710 | 673 | 18.1% | 38/191 | 19.9% | 11/14 | 78.6% |

Coverage run:

- CPU broad changed-code diagnostic: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-cpu-v4-20260630-after-extra-contracts-20260630-081511`

Interpretation:

- The four new tests are behavior-level additions in the existing generation/lifecycle contract layer and pass both official and counter references, so they do not appear to bind to the official implementation structure.
- The broad coverage denominator is larger than the earlier focused reports because it includes all changed reference files, including optional modelopt/QAT/NPU and diagnostic code. Use this run as a broad diagnostic, not as a direct apples-to-apples comparison with the focused vLLM-only coverage rows above.

## Hand-Written Evaluator v7 / GPU Weight Update v3

Expanded the GPU evaluator from 6 to 7 pytest cases by adding a live weight-update behavior test.

GPU addition:

- launch a real standalone vLLM rollout through `LLMServerManager`;
- generate from a fixed prompt and record the response logprob;
- update the live rollout through the public async rollout adapter `update_weights()` using a deterministic zeroed HF weight stream;
- generate again from the same prompt and verify that the logprob distribution reflects the new zeroed weights and that the rollout reports the updated global step.

Pass-rate summary:

| suite | pytest nodes | baseline | official reference | counter reference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 90 | 25/90 | 90/90 | 90/90 |
| GPU | 7 | 0/7 | 7/7 | 7/7 |
| CPU + GPU | 97 | 25/97 | 97/97 | 97/97 |

Broad changed-code coverage diagnostic:

| suite | official line cov | official funcs | official classes | counter line cov | counter funcs | counter classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 13.4% | 24/255 | 18/32 | 17.0% | 25/191 | 11/14 |
| CPU + GPU | 17.2% | 49/255 | 17/32 | 22.5% | 51/191 | 11/14 |

Coverage runs:

- GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-gpu-v3-weight-update-20260630-20260630-084318`
- CPU + GPU: `/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-combined-v4-weight-update-20260630-20260630-084731`

Interpretation:

- The new GPU test covers real weight streaming and live backend reload behavior without importing the official worker-extension or transport internals directly.
- The broad combined counter-reference coverage increases materially because the counter implementation has a more compact changed-code denominator and the new test activates its adapter/transport path: `vllm_engine/adapter.py` reaches `96/167` changed lines (`57.5%`), `vllm_engine/transport.py` reaches `104/259` (`40.2%`), and `vllm_engine/runtime.py` reaches `279/759` (`36.8%`).
- The official broad line coverage remains lower because the official changed-code denominator includes many optional/specialized paths such as modelopt, QAT, NPU, and extra checkpoint engines. The new test still raises direct transport coverage: `vllm_rollout/bucketed_weight_transfer.py` reaches `100/255` changed lines (`39.2%`).
