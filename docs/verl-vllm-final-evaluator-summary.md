# VERL vLLM Evaluator Summary

Date: 2026-07-01 UTC

This document summarizes the current evaluator status for the sanitized VERL vLLM rollout task. The main objective is to make the evaluator behavior-based: correct alternative implementations should pass even when they do not reproduce the official reference's internal module names, class names, helper functions, or file layout.

## 1. Counter References And False-Negative Control

The counter-reference process created structurally different but behaviorally valid implementations of the same vLLM rollout capability. These are used to detect evaluator tests that accidentally require the official implementation structure.

Current counter-reference set:

- 4 regular counter references: `run_001` through `run_004`.
- 4 anonymized variants: `run_001_anon` through `run_004_anon`.
- Archive: `/mnt/data/projects/verl-vllm-counterreference-oracle/docs/counterreference_runs_archive.md`.

All eight were validated before evaluator scoring:

| target set | CPU behavior gate | GPU training smoke |
| --- | ---: | --- |
| 8 counter references | `55 passed` each | Qwen3-8B GRPO/vLLM 1-step passed, `training/global_step:1` each |

The counter references exposed a repeated evaluator failure mode: generated tests often passed the official reference but failed a valid alternative implementation because they imported or patched official-private internals.

Representative pre-fix result:

| evaluator | tests | official reference | counter reference | invalid signal |
| --- | ---: | ---: | ---: | --- |
| Gemini agent 4 third rerun | 7 | 7/7 | 4/7 | 3/7 tests, 42.9%, were false negatives against a valid counter reference |

The failing tests hard-coded official implementation details such as:

- `verl.workers.rollout.vllm_rollout.vllm_async_server`
- `vLLMHttpServer`
- `vLLMReplica`
- `BucketedWeightSender`
- `extract_prompt_logprobs`

After using the counter references as a guardrail, the current hand-written evaluator no longer has this failure mode:

| validation target | result |
| --- | ---: |
| `run_001` | 97 passed, 12 skipped |
| `run_002` | 97 passed, 12 skipped |
| `run_003` | 97 passed, 12 skipped |
| `run_004` | 97 passed, 12 skipped |
| `run_001_anon` | 97 passed, 12 skipped |
| `run_002_anon` | 97 passed, 12 skipped |
| `run_003_anon` | 97 passed, 12 skipped |
| `run_004_anon` | 97 passed, 12 skipped |

Run:

```text
/mnt/data/projects/verl-vllm-testgen-runs/coverage-score-eight-counterrefs-after-behavior-relax-v1-gpumem015-20260701-110124
```

Current source audit for common official-private structure oracles is clean: no test imports or asserts direct use of `vllm_rollout.vllm_rollout`, `vllm_rollout.vllm_async_server`, `bucketed_weight_transfer`, `_execute_method`, `patch.object`, `issubclass`, or `__mro__`.

Recent cleanup actions:

- Deleted the direct `_execute_method` adapter test.
- Relaxed lifecycle tests to check observable behavior instead of exact private call shapes:
  - `collective_rpc` payload shape,
  - exact `wake_up` tags,
  - exact `sleep(level=...)`,
  - fake-engine abort/reset internals.

Net effect: the measured false-negative rate from official-structure tests went from 3/7 on the latest problematic generated evaluator to 0/97 non-skipped nodes across all eight current counter references.

## 2. Coverage And False-Positive Control

Coverage is measured against changed implementation lines relative to the sanitized starting codebase, not whole-repository lines. The purpose is diagnostic: if evaluator tests do not activate reference implementation behavior, partial or broken candidates may pass.

Early coverage was too low. The first useful generated/hand-written runs were around 7-13% line coverage, and an early stdlib `trace` implementation undercounted async coroutine bodies. The coverage tool was replaced with a `sys.settrace`/`threading.settrace` runner that correctly counts async and threaded code under `PROJECT_UNDER_TEST`.

The current broad CPU+GPU diagnostic for the main official and original counter reference reached the target range:

| reference | pytest | changed lines | line cov | functions | function cov | classes | class cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official reference | 130 passed, 12 skipped | 2586/4955 | 52.2% | 167/255 | 65.5% | 17/32 | 53.1% |
| original counter reference | 124 passed, 18 skipped | 2079/3710 | 56.0% | 138/191 | 72.3% | 11/14 | 78.6% |

Run:

```text
/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-cpu-gpu-combined-v1-20260701-014346
```

The CPU-only broad diagnostic also reached about 50% on the official reference:

| suite | official changed-line coverage | official function coverage |
| --- | ---: | ---: |
| CPU-only v5 diagnostic | 2485/4955, 50.2% | 161/255, 63.1% |
| CPU+GPU combined | 2586/4955, 52.2% | 167/255, 65.5% |

Run:

```text
/mnt/data/projects/verl-vllm-testgen-runs/coverage-coverage-handwritten-cpu-v5-diagnose-all-20260701-011806
```

Main remaining uncovered function groups in the official reference:

| category | uncovered functions | reason |
| --- | ---: | --- |
| Optional quantization / QAT / FP8 / ModelOpt | 71 | tiny unquantized Qwen2 model does not enter optional quantized reload branches |
| Alternative checkpoint / distributed weight-transfer engines | 44 | KIMI/HCCL/Mooncake, large direct CUDA IPC, shared-memory fallback, and explicit process-group flows are not part of the fast tier |
| vLLM server launch / CLI / branch-specific lifecycle | 43 | evaluator covers manager/client launch and lifecycle, but not every CLI/headless/quant/MTP branch |
| Repository diagnostics / environment checks | 17 | diagnostic utilities, not the vLLM rollout behavior target |
| NPU / Ascend-specific patching | 15 | CUDA H100 runtime does not enter Ascend-specific branches |
| LoRA/FSDP/Megatron/HF/SGLang side paths | 9 | adjacent compatibility code, not always triggered by vLLM rollout tests |
| Training/trainer helpers outside rollout smoke path | 8 | full PPO/GRPO training is not part of the fast evaluator tier |
| Miscellaneous small helpers | 2 | small changed helpers not reached by current behavior scenarios |

Remaining gaps are now explicit and mostly correspond to optional/heavy paths. This reduces false positives because the evaluator no longer only checks registration/config smoke; it activates generation, logprobs, prompt logprobs, multimodal pass-through, lifecycle/cache/abort behavior, fully async dataflow, checkpoint/update ordering, adapter/Lora/MTP/routing metadata, live vLLM generation, and live weight update behavior.

## 3. Candidate Score Discrimination

The current behavior evaluator gives non-uniform scores across historical Gemini candidates. This is the main evidence that the tests have useful resolution instead of collapsing all candidates into the same pass/fail bucket.

Scoring run:

```text
/mnt/data/projects/verl-vllm-testgen-runs/coverage-score-candidates-after-internal-tests-removed-v3-correct-root-gpumem015-20260701-063744
```

Skipped tests are excluded from the pass-rate denominator. Collection/runtime errors are counted as failed scored nodes. Each target had 98 non-skipped scored nodes.

| ID | candidate | pass / scored | pass rate |
| --- | --- | ---: | ---: |
| baseline | incomplete task codebase | 21/98 | 21.4% |
| C1 | `20260626-081247-gemini31pro-preview-candidate2` | 31/98 | 31.6% |
| C2 | `20260626-091059-gemini31pro-v2-candidate1` | 43/98 | 43.9% |
| C3 | `20260626-091100-gemini31pro-v2-candidate2` | 35/98 | 35.7% |
| C4 | `20260626-091100-gemini31pro-v2-candidate3` | 45/98 | 45.9% |
| C5 | `20260626-091100-gemini31pro-v2-candidate4` | 44/98 | 44.9% |
| C6 | `20260626-093319-gemini35flash-v3-candidate1` | 41/98 | 41.8% |
| C7 | `20260626-093319-gemini35flash-v3-candidate2` | 37/98 | 37.8% |
| C8 | `20260626-093319-gemini35flash-v3-candidate4` | 37/98 | 37.8% |
| C9 | `20260627-044228-gemini35flash-parity-candidate1` | 21/98 | 21.4% |
| C10 | `20260627-044228-gemini35flash-parity-candidate2` | 38/98 | 38.8% |
| C11 | `20260627-044228-gemini35flash-parity-candidate3` | 22/98 | 22.4% |

Observed range:

- baseline: 21.4%;
- weakest candidates: 21.4-22.4%;
- middle candidates: 31.6-38.8%;
- strongest candidates: 41.8-45.9%.

This is a meaningful gradient. It matches qualitative inspection: candidates that only add a registry/config stub stay near baseline, while candidates that implement more lifecycle/generation/manager behavior score higher but still fail missing metadata, prompt-logprob, LoRA/MTP/routing, weight-update, or live-backend semantics.

Historical comparison:

- Before removing internal-structure tests, a full candidate score run produced only `1 error in ~3s` for every candidate. That run had no useful resolution because pytest failed during collection/setup through direct reference-only imports.
- After removing/relaxing those tests, the same candidate family reaches behavior assertions and spreads across 21-45 passed nodes.

## 4. Current Method

The current data-generation/evaluator method has three separate checks:

1. Counter references reduce false negatives.
   - A valid alternative implementation must pass.
   - Anonymized counter references specifically reject official-private import paths.
   - If a test passes official but fails counter references, treat it as over-specified unless the name is anchored in the task prompt or remaining project.

2. Coverage diagnostics reduce false positives.
   - Measure changed-code activation on official and counter references.
   - Use coverage to find missing behavior areas, not as a hidden oracle for candidates.
   - Keep broad coverage reports separate from `score.sh`.

3. Candidate score tables measure resolution.
   - Score baseline, official reference, counter references, and historical candidate patches with the same evaluator.
   - The evaluator is useful only if references pass, baseline fails meaningfully, and partial candidates occupy different score bands.

This makes the evaluator substantially more reliable than the earlier generated suites: it is no longer dominated by official-reference structure, it covers much more of the real implementation behavior, and it ranks partial implementations with a clear gradient.
