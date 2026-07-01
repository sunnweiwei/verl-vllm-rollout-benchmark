# CPU Evaluation Tests

These tests are not part of the agent workspace. They are applied to a fresh
copy of a candidate VERL checkout and run in a separate test container.

Run against a candidate checkout:

```bash
docker build -f eval_tests/Dockerfile.cpu -t verl-vllm-rollout-benchmark:cpu-tests .
eval_tests/run_cpu_tests.sh /path/to/candidate/verl
```

The runner copies the checkout into a temporary workspace, overlays the tests
under `tests/eval_vllm/`, and runs pytest inside a dedicated test image by
default. This test image is separate from the agent workspace image.

The tests focus on public behavior across 90 CPU pytest cases:

- backend/config selection for `rollout.name=vllm`, including reward-model
  rollout, async rollout-adapter registry selection, and unsupported
  PD-disaggregation handling.
- token generation contracts: default and explicit sampling budgets, logprobs,
  prompt logprobs including sampled-token and rank-order cases, abort output,
  global step metadata, multimodal payloads,
  Qwen visual-token deduplication before backend generation, backend actor
  address/RPC methods, LoRA adapter use, routed experts gated by routing
  replay configuration, MTP rollout stats,
  and server-level wake/sleep/cache/abort/resume/profile lifecycle behavior
  across rollout modes and LoRA/full-weight memory levels.
- agent-loop/DataProto contracts: response masks, rollout logprobs,
  extra-field schema stability, routed-expert alignment, and multimodal
  forwarding.
- fully async partial-rollout behavior: abort resume, max-token budget
  updates, global-step min/max tracking, routed-expert merging, and multimodal
  forwarding across resumed calls.
- checkpoint/lifecycle behavior: abort/release/update/resume ordering,
  replica lifecycle fan-out, async rollout-adapter lifecycle forwarding,
  server-handle lazy lookup, rank gating, global step propagation, and
  profiling hooks.
- LoRA/adapter weight sync behavior for naive and non-naive checkpoint paths.
- downstream training consumers of rollout outputs, including rollout
  correction/bypass mode and router replay.

The CPU tests replace the actual vLLM engine launch with scripted Ray rollout
servers, so they do not require GPUs or model weights. They intentionally avoid
checking module paths or concrete implementation class names; the fixed
contract is that VERL can select the backend with `rollout.name=vllm` and then
use compatible manager/client/lifecycle and training-output behavior as other
async rollout backends.

## GPU Evaluation Tests

The GPU suite is intentionally separate from the CPU suite. It launches a real
standalone rollout server through VERL's public rollout manager/client path and
requires a local model directory:

```bash
eval_tests/run_gpu_tests.sh /path/to/candidate/verl
```

If `VERL_GPU_TEST_MODEL` is not set, the runner creates a tiny random Qwen2
HF model in its temporary workspace and uses that model. You can still override
the model explicitly:

```bash
VERL_GPU_TEST_MODEL=/path/to/local/model \
eval_tests/run_gpu_tests.sh /path/to/candidate/verl
```

The GPU suite currently collects 24 pytest nodes and covers real token generation
with response logprobs, prompt logprobs, explicit `max_tokens`,
`max_new_tokens`, default response-budget handling, parallel requests, the
OpenAI-compatible completion endpoint, full-context rejection, load-balancer
drain state, abort/resume, KV cache lifecycle hooks, profile hooks, and the
fully-async rollout client contract. It also updates a live vLLM rollout
through the public async rollout adapter and verifies that the next generation
uses the new weights and reports the updated global step. Extended GPU nodes
cover LoRA adapter weight sync, small-bucket weight transfer, online FP8 vLLM
quantization with a generated tiny compatible model, optional TP=2 and DP=2
execution when at least two GPUs are visible, and a manager-level NCCL
checkpoint-transfer round trip that preserves small and bucket-split tensors.
Optional heavier nodes cover direct large-tensor IPC transfer, speculative/MTP
rollout metrics with a real compatible MTP fixture, FP8 weight-sync, quantized,
QAT, or ModelOpt model generation/update with caller-provided local fixtures,
and KIMI/Mooncake checkpoint-engine round trips when those backends are
installed and explicitly enabled. It still
avoids checking implementation file names or concrete backend classes; the required behavior is that
`rollout.name=vllm` can be used as a real VERL inference backend under the
existing rollout manager/client API.

This is a fast smoke tier, not a throughput benchmark. By default it uses a
tiny randomly initialized Qwen2-compatible causal LM; when overriding the model,
use the smallest local vLLM-compatible causal language model that exercises the
backend, preferably sub-1B parameters. Defaults are intentionally small: one GPU, TP=1,
`max_model_len=40`, `max_num_seqs=2`, `gpu_memory_utilization=0.05`, and two generated tokens per request.
Override these only for a separate heavier GPU tier.
