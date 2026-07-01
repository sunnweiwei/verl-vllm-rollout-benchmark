# Feature request: add vLLM rollout support

We use VERL for PPO/GRPO-style RL training, and we want to add vLLM as a supported rollout backend.

Users should be able to select it with:

```bash
actor_rollout_ref.rollout.name=vllm
```

and then use the normal VERL trainer and generation flows without switching to a separate workflow.

Please integrate vLLM into the existing async rollout path. The implementation should cover the usual training lifecycle: starting rollout workers or servers, generating from tokenized prompts, returning generated token ids and optional logprob information through the existing rollout client flow, refreshing rollout model weights during training, aborting or resuming in-flight generation when needed, and releasing or restoring cache/memory around update phases.

In practice, this backend will sit behind VERL's existing rollout manager/client APIs and will be used by the same trainer, checkpoint/update, fully async rollout, agent-loop, and rollout-output consumer code as the other backends. When those flows pass through generation options, multimodal inputs, lifecycle calls, cache or weight-update state, or optional generation metadata, the vLLM path should preserve the same behavior instead of silently dropping information.

Existing SGLang, TRTLLM, and HF rollout paths should keep working. Add only the configuration and public surface that are needed for vLLM to fit naturally into the current VERL backend model, and avoid hardcoding behavior for a single model, script, or test fixture.
