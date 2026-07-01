# Candidate Analysis Notes

This document tracks implementation patterns and failure modes observed from
candidate solutions for the sanitized VERL vLLM rollout task. The goal is to
calibrate evaluation tests against behavior that is required by the surrounding
codebase, not against the reference implementation's private structure.

## Evaluation Principles

- Test behavior at integration boundaries that are visible from the remaining
  codebase.
- Do not require a specific file name, class name, helper name, constant name, or
  internal architecture unless the remaining code calls it directly or the task
  prompt explicitly asks for it.
- A name/signature is fair to test when it is discoverable from existing call
  sites, shared base classes, backend registries, or configuration schema.
- Prefer tests that exercise manager/client/checkpoint/training behavior over
  tests that inspect implementation details.
- When a test uses a fake backend, the fake should accept multiple reasonable
  implementation strategies where possible. It should assert externally visible
  effects, such as handles being exposed, generated outputs being normalized,
  weights being synchronized, KV cache being cleared, or global step information
  being propagated.
- Separate regression guards for non-vLLM paths from vLLM capability tests so
  pass counts are not misleading.

## Version 1: Gemini Candidate

Run directory:

`/mnt/data/projects/verl-vllm-gemini-runs/20260626-081247-gemini31pro-preview-candidate2`

Patch:

`/mnt/data/projects/verl-vllm-gemini-runs/20260626-081247-gemini31pro-preview-candidate2/candidate.patch`

Observed test result:

`31 passed, 15 failed`

### Fairly Detected Issues

- The rollout config did not expose `engine_kwargs.vllm`, so user-facing config
  overrides such as `actor_rollout_ref.rollout.engine_kwargs.vllm.*` would not
  work.
- Lifecycle methods did not forward all required server operations, and
  `abort_all_requests` did not return aggregate abort information.
- Some generation behavior was missing, including response normalization,
  `max_new_tokens` compatibility, logprob handling, prompt logprob handling,
  multimodal processor kwargs, empty backend output handling, routed experts, MTP
  metrics, and LoRA request behavior.

### Tests That Are Too Structure-Specific

- The current generation helper imports
  `verl.workers.rollout.vllm_rollout.vllm_async_server` and instantiates
  `vLLMHttpServer` directly. A correct implementation could use different files
  or classes.
- The current LoRA generation test imports `VLLM_LORA_INT_ID` directly. The
  behavior that matters is that generation uses the active adapter after it has
  been loaded, not the exact constant name.
- These tests should be refactored to reach the server through the rollout
  registry/manager path or through a fake vLLM backend that observes generation
  calls without assuming internal module names.

### Real Issues Not Well Covered Yet

- Manager startup postconditions: `LLMServerManager` and other remaining
  call sites read `_server_handle` and `_server_address` from rollout replicas.
  Version 1 launches an actor but does not populate these fields or `servers`.
  Existing tests monkeypatch standalone init and therefore miss this.
- Standalone initialization path: base `init_standalone` calls
  `get_ray_class_with_init_args`. Version 1 returns a tuple instead of an object
  that `RayWorkerGroup` can consume. This should be caught by exercising manager
  startup with fake Ray resources, not by asserting the exact return type.
- Colocated reward/teacher paths: reward and teacher managers call
  `init_colocated(resource_pool)` for the selected inference backend. Version 1
  raises `NotImplementedError`. This is a real call-site contract.
- Checkpoint server adapter contract: `CheckpointEngineWorker` creates the
  rollout adapter via `get_rollout_class(name, mode)` and reads
  `server_adapter.is_leader_rank`. Version 1 does not expose that property.
- Weight synchronization semantics: training code passes `peft_config`,
  `base_sync_done`, and `global_steps` into rollout weight updates. Version 1
  drops these values and does not clear KV cache or propagate global steps after
  sync.
- vLLM engine configuration mapping: Version 1 reads non-existent top-level
  config fields such as `vllm_kwargs` and `max_concurrency` instead of using the
  remaining rollout config schema (`engine_kwargs.vllm`, `max_num_seqs`, etc.).
- Safe sleep/release behavior: weight-sync lifecycle must not race with
  in-flight requests. Tests should assert the effect that requests are drained
  before cache release/sleep, while avoiding a reference-only implementation
  requirement.

### Lower Priority Or Needs More Evidence

- A single-request `abort_request` API may be useful, but it should not be part
  of the core CPU score unless a remaining call site requires it. The clearly
  required behavior is `abort_all_requests` plus generation resume for partial
  rollout/checkpoint updates.
- Multi-node launch details should be tested through observable behavior
  (replica/server availability and routing), not through exact actor names.

## Next Candidate Runs

Run several independent Gemini candidates and classify each failure into:

- Required external behavior missing.
- Existing test over-constrains implementation details.
- Ambiguous behavior that needs prompt clarification or more examples.
- Non-vLLM regression.

## Version 2: Four Parallel Gemini Candidates

Runs:

- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-091059-gemini31pro-v2-candidate1`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-091100-gemini31pro-v2-candidate2`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-091100-gemini31pro-v2-candidate3`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-091100-gemini31pro-v2-candidate4`

All four produced patches and completed the current CPU test suite.

Results:

- Candidate 1: `15 failed, 31 passed`
- Candidate 2: `15 failed, 31 passed`
- Candidate 3: `15 failed, 31 passed`
- Candidate 4: `15 failed, 31 passed`

Patch sizes and implementation shapes varied:

- Candidate 1: about 16 KB, adds `vllm_rollout/vllm_async_server.py`
  and `vllm_rollout/vllm_rollout.py`, plus a stray root-level
  `test_vllm_worker.py`.
- Candidate 2: about 17 KB, uses `vllm_rollout/async_vllm_server.py`
  instead of `vllm_async_server.py`.
- Candidate 3: about 18 KB, adds `vllm_rollout/vllm_async_server.py`,
  `vllm_rollout/vllm_rollout.py`, plus stray root-level `test_vllm.py`
  and `test_vllm_rpc.py`.
- Candidate 4: about 44 KB, adds `vllm_rollout/vllm_async_server.py`,
  `vllm_rollout/vllm_rollout.py`, and an extra root rollout file
  `vllm_rollout_server.py`.

### Strong Signal: Current Generation Tests Are Over-Constrained

The generation failures did not reach candidate generation behavior. They failed
in the test helper before calling `generate`:

- Candidate 2 used `async_vllm_server.py`, so importing
  `verl.workers.rollout.vllm_rollout.vllm_async_server` failed.
- Candidates 1, 3, and 4 had `vllm_async_server.py`, but the helper tried to
  monkeypatch `qwen2_5_vl_dedup_image_tokens`, a reference-specific helper name
  that the task prompt and remaining call sites do not require.
- The LoRA generation test imported `verl.workers.rollout.vllm_rollout.utils`
  and `VLLM_LORA_INT_ID`; several candidates had no such file. The required
  behavior is adapter-aware generation, not this constant name.

Conclusion: refactor generation tests before using these failures for scoring.
The tests should instantiate or reach candidate generation through the rollout
registry/manager path or through a flexible fake vLLM backend, and then assert
the `TokenOutput` behavior.

### Strong Signal: Config Contract Is Real And Repeatedly Missed

All four candidates registered the vLLM backend in Python, but all four failed
to add `vllm` under `verl/trainer/config/rollout/rollout.yaml` `engine_kwargs`.
This is a real user-facing config contract because examples and command-line
overrides use `actor_rollout_ref.rollout.engine_kwargs.vllm.*`.

This should remain a test, but we should phrase it as a config behavior test:
vLLM-specific engine kwargs must be accepted and forwarded without dropping
SGLang/TRT-LLM kwargs.

### Strong Signal: Lifecycle Forwarding Is Real And Repeatedly Missed

All four candidates failed the current lifecycle tests because their replica
`abort_all_requests` returned `None`, `sleep`/`release_kv_cache` did not drain
requests first, and `abort_request` was absent.

Keep these as separate judgments:

- `abort_all_requests` returning useful aggregate information is fair because
  partial rollout/checkpoint code needs coordinated abort/resume behavior.
- draining before sleep/cache release is fair as a behavioral safety contract for
  weight sync and cache release.
- single-request `abort_request` should remain lower priority unless a remaining
  non-test call site requires it.

### Strong Signal: Startup/Manager Postconditions Need New Tests

The current passing manager/client tests still monkeypatch vLLM standalone init,
so they do not verify the candidate's actual startup path. Across the four
candidates, several implementations set registry entries but have fragile or
incomplete server launch behavior.

Add tests that exercise `LLMServerManager.create` with fake Ray/vLLM resources
and assert only external effects:

- manager exposes non-empty `server_handles` and `server_addresses`;
- generated requests route through the exposed handle;
- reward/teacher colocated paths can initialize when their inference backend is
  vLLM.

Do not require a specific server file name, actor class name, or actor naming
scheme.

### Strong Signal: Weight Sync Semantics Need Behavior Tests

The candidates vary in how they attempt weight sync:

- some convert all weights to a dict on CPU;
- some bucket weights;
- some set `global_steps`;
- some expose `is_leader_rank`;
- some explicitly reject LoRA adapter sync.

The fair test target is not IPC, ZMQ, or any reference transfer mechanism. The
fair target is:

- all ranks/generators are consumed so trainer-side collectives cannot hang;
- leader-side server receives the updated weights;
- `peft_config` and `base_sync_done` are not dropped when provided by the
  training worker;
- global step information reaches future generation outputs or equivalent
  rollout metadata;
- KV cache is cleared after weights change.

### Implementation Quality Signals

- Candidate 1 and Candidate 3 left ad hoc root-level test files in the source
  tree. This is a quality issue but should not be part of vLLM behavior scoring
  unless we add a general cleanliness check for stray files.
- Candidate 4 added a large `vllm_rollout_server.py` copied from a neighboring
  backend shape. The existence of this file is not wrong by itself; tests should
  judge whether the registered vLLM path works.

## Version 3: Four Gemini 3.5 Flash Candidates

Runs:

- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-093319-gemini35flash-v3-candidate1`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-093319-gemini35flash-v3-candidate2`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-093319-gemini35flash-v3-candidate3`
- `/mnt/data/projects/verl-vllm-gemini-runs/20260626-093319-gemini35flash-v3-candidate4`

Results:

- Candidate 1: produced a 32 KB patch, `15 failed, 31 passed`.
- Candidate 2: produced a 26 KB patch, `15 failed, 31 passed`.
- Candidate 3: timed out after the 2 hour Gemini limit and did not export a
  patch. It had started editing `base.py`, `replica.py`, and
  `vllm_rollout/`, then spent time installing test requirements and running its
  own test.
- Candidate 4: produced a 35 KB patch, `15 failed, 31 passed`.

Patch shapes differed:

- Candidate 1 added `vllm_async_server.py`, `vllm_rollout.py`, and its own
  `tests/workers/rollout/test_vllm_rollout.py`.
- Candidate 2 added only `vllm_rollout.py` under the vLLM rollout package and
  modified `verl/workers/config/rollout.py`.
- Candidate 4 added `vllm_async_server.py`, `vllm_rollout.py`, and
  `vllm_worker_extension.py`, and modified `llm_server.py`.
- Candidate 3 was much slower and attempted to install/run pytest inside the
  task environment before timing out.

### Flash Confirms The Current Score Is Not Discriminative Enough

The completed Flash candidates had different implementation shapes and editing
histories, but all three completed runs received exactly the same score and the
same failed test names. The repeated score is therefore not strong evidence that
the candidates are equally good. It is evidence that the current suite has
large early gates that flatten many distinct implementations into the same
failure bucket.

The same two causes reappeared:

- Real missing behavior: all completed Flash candidates missed the
  `engine_kwargs.vllm` config contract and lifecycle/drain/abort behavior.
- Over-constrained generation harness: generation tests did not reach most
  candidate `generate` implementations. Candidate 2 failed because the helper
  imports `vllm_rollout.vllm_async_server`. Candidates 1 and 4 had that module
  but failed because the helper monkeypatches
  `qwen2_5_vl_dedup_image_tokens`. The LoRA test still imports
  `vllm_rollout.utils.VLLM_LORA_INT_ID`.

### Implications For The Next Test Revision

- Keep the config and lifecycle tests, but split them into smaller, separately
  reported checks so a candidate can receive partial credit for adding config,
  drain, abort aggregation, and server forwarding independently.
- Rewrite generation tests so they enter through a remaining public integration
  point, such as backend registry/manager construction, or through an adapter
  hook supplied by the test harness. Do not import a reference-specific
  `vllm_async_server` module or require private helper names.
- Keep behavior requirements for tokens, logprobs, prompt logprobs,
  multimodal kwargs, LoRA, MTP stats, routed experts, empty outputs, and context
  clamping, but assert those after a flexible fake backend has been connected to
  the candidate implementation.
- Treat candidate-added tests as quality/noise signals only. They should not
  affect vLLM behavior scoring unless they break normal package import or
  execution.

## Test Revision: Flexible Generation Harness

The first revision produced identical `15 failed, 31 passed` results for every
completed patch. The updated CPU suite keeps the same behavioral goals but
removes several reference-specific assumptions:

- Generation tests no longer import a fixed
  `verl.workers.rollout.vllm_rollout.vllm_async_server` module or require a
  class named `vLLMHttpServer`.
- The test harness scans the candidate's `vllm_rollout` package for any
  server-like object with a `generate` method, including classes wrapped with
  `@ray.remote`.
- Optional reference helpers such as `qwen2_5_vl_dedup_image_tokens` are patched
  only when present.
- LoRA generation no longer imports `VLLM_LORA_INT_ID`; the fake backend reports
  any candidate-selected LoRA id as loaded.
- Lifecycle checks are split so wake/sleep/cache/profile forwarding can pass
  separately from abort aggregation and drain ordering.
- Compound generation checks are split: token/logprob handling, stop/global-step
  metadata, `max_new_tokens`, multimodal payloads, and priority forwarding now
  receive separate credit.

Reference VERL v0.8.0 result after the latest revision:

- `55 passed`

Historical candidate results after the latest revision:

- `20260626-081247-gemini31pro-preview-candidate2`: `22 failed, 33 passed`
- `20260626-091059-gemini31pro-v2-candidate1`: `16 failed, 39 passed`
- `20260626-091100-gemini31pro-v2-candidate2`: `19 failed, 36 passed`
- `20260626-091100-gemini31pro-v2-candidate3`: `16 failed, 39 passed`
- `20260626-091100-gemini31pro-v2-candidate4`: `16 failed, 39 passed`
- `20260626-093319-gemini35flash-v3-candidate1`: `15 failed, 40 passed`
- `20260626-093319-gemini35flash-v3-candidate2`: `17 failed, 38 passed`
- `20260626-093319-gemini35flash-v3-candidate4`: `18 failed, 37 passed`

This is still not a full behavioral proof, but it is a meaningful improvement:
the suite now separates clearly weaker stubs from candidates that implemented
more of the generation and lifecycle contracts, while keeping the official
implementation as the passing reference. The remaining universal failures are
real capability gaps shared by these candidates: the rollout config still omits
`engine_kwargs.vllm`, abort/drain lifecycle semantics are incomplete, and
generation still misses stop/global-step normalization, multimodal payload
forwarding, request priority, empty abort outputs, prompt logprobs, routed
experts, LoRA adapter requests, and MTP stats.

## Prompt Revision: Backend Capability Parity

The task prompt was updated to clarify the expected abstraction level without
listing the hidden test features. The new text tells agents to treat SGLang and
TRTLLM as examples of VERL rollout backend public behavior, and to make vLLM a
first-class backend for existing manager/client, trainer, checkpoint/update,
agent-loop, lifecycle, and rollout-output consumer flows.

Four new Gemini 3.5 Flash candidates were run with this revised prompt:

- `20260627-044228-gemini35flash-parity-candidate1`: `28 failed, 27 passed`
- `20260627-044228-gemini35flash-parity-candidate2`: `17 failed, 38 passed`
- `20260627-044228-gemini35flash-parity-candidate3`: `30 failed, 25 passed`
- `20260627-044228-gemini35flash-parity-candidate4`: no patch; the run got
  stuck trying to compile/install the local vLLM tree and was terminated.

The revised prompt did not produce a fully correct solution. Candidate 2 is the
only useful new sample: it passed the manager/client, reward-manager, lifecycle
fan-out, and fully-async client tests, but still failed the rollout config
contract, abort aggregation, drain-before-sleep/release, request-level abort,
and most generation edge contracts. Candidate 1 and candidate 3 were worse than
the previous best candidates, and candidate 4 never reached implementation.

This suggests the parity wording is directionally better for making at least one
candidate read broader VERL flows, but it is not sufficient by itself. Gemini
3.5 Flash still tends to either implement a shallow backend skeleton, over-focus
on local self-tests, or spend time trying to install/compile vLLM instead of
using the provided source tree as reference material.
