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

The tests focus on public behavior:

- `rollout.name=vllm` can be initialized through the normal `LLMServerManager`.
- VERL's rollout client can generate through that backend and returns
  `TokenOutput` data without leaking in-flight load-balancer state.
- Fully async generation resumes after an aborted partial rollout.
- The rollout replica lifecycle methods used by checkpoint/update phases are
  forwarded to rollout servers.
- Existing rollout configuration still exposes backend-specific engine kwargs.

The CPU tests replace the actual vLLM engine launch with scripted Ray rollout
servers, so they do not require GPUs or model weights. They intentionally avoid
checking module paths or concrete implementation class names; the fixed
contract is that VERL can select the backend with `rollout.name=vllm` and then
use the same manager/client/lifecycle entry points as other async rollout
backends.
