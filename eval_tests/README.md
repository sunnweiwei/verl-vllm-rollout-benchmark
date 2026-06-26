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

The tests focus on public behavior across 46 CPU pytest cases:

- backend/config selection for `rollout.name=vllm`, including reward-model
  rollout and unsupported PD-disaggregation handling.
- token generation contracts: sampling limits, logprobs, prompt logprobs,
  abort output, global step metadata, multimodal payloads, LoRA adapter use,
  routed experts, and MTP rollout stats.
- agent-loop/DataProto contracts: response masks, rollout logprobs,
  extra-field schema stability, routed-expert alignment, and multimodal
  forwarding.
- fully async partial-rollout behavior: abort resume, max-token budget
  updates, global-step min/max tracking, routed-expert merging, and multimodal
  forwarding across resumed calls.
- checkpoint/lifecycle behavior: abort/release/update/resume ordering,
  replica lifecycle fan-out, global step propagation, and profiling hooks.
- LoRA/adapter weight sync behavior for naive and non-naive checkpoint paths.
- downstream training consumers of rollout outputs, including rollout
  correction/bypass mode and router replay.

The CPU tests replace the actual vLLM engine launch with scripted Ray rollout
servers, so they do not require GPUs or model weights. They intentionally avoid
checking module paths or concrete implementation class names; the fixed
contract is that VERL can select the backend with `rollout.name=vllm` and then
use compatible manager/client/lifecycle and training-output behavior as other
async rollout backends.
