import asyncio
from contextlib import contextmanager
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _require_gpu_model() -> str:
    import torch

    model_path = os.environ.get("VERL_GPU_TEST_MODEL")
    if not model_path:
        pytest.skip("VERL_GPU_TEST_MODEL is required for GPU rollout tests")
    if not Path(model_path).exists():
        pytest.skip(f"VERL_GPU_TEST_MODEL does not exist: {model_path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU rollout tests")
    return model_path


def _compose_real_vllm_config(model_path: str):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = os.path.abspath("verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    gpus = int(os.environ.get("VERL_GPU_TEST_GPUS", "1"))
    tp = int(os.environ.get("VERL_GPU_TEST_TP", "1"))
    max_prompt_len = int(os.environ.get("VERL_GPU_TEST_PROMPT_LEN", "24"))
    max_response_len = int(os.environ.get("VERL_GPU_TEST_RESPONSE_LEN", "4"))
    max_model_len = int(os.environ.get("VERL_GPU_TEST_MAX_MODEL_LEN", str(max_prompt_len + max_response_len + 12)))
    max_num_seqs = int(os.environ.get("VERL_GPU_TEST_MAX_NUM_SEQS", "2"))

    updates = {
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": gpus,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.trust_remote_code": _bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
        "actor_rollout_ref.model.load_tokenizer": True,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.model.lora.merge": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.nnodes": 1,
        "actor_rollout_ref.rollout.n_gpus_per_node": gpus,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": tp,
        "actor_rollout_ref.rollout.data_parallel_size": 1,
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
        "actor_rollout_ref.rollout.prompt_length": max_prompt_len,
        "actor_rollout_ref.rollout.response_length": max_response_len,
        "actor_rollout_ref.rollout.max_model_len": max_model_len,
        "actor_rollout_ref.rollout.max_num_seqs": max_num_seqs,
        "actor_rollout_ref.rollout.max_num_batched_tokens": int(
            os.environ.get("VERL_GPU_TEST_MAX_NUM_BATCHED_TOKENS", str(max_model_len * max_num_seqs))
        ),
        "actor_rollout_ref.rollout.dtype": os.environ.get("VERL_GPU_TEST_DTYPE", "float16"),
        "actor_rollout_ref.rollout.gpu_memory_utilization": float(
            os.environ.get("VERL_GPU_TEST_GPU_MEMORY_UTILIZATION", "0.05")
        ),
        "actor_rollout_ref.rollout.enforce_eager": _bool_env("VERL_GPU_TEST_ENFORCE_EAGER", True),
        "actor_rollout_ref.rollout.skip_tokenizer_init": False,
        "actor_rollout_ref.rollout.load_format": "auto",
        "actor_rollout_ref.rollout.disable_log_stats": False,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.enable_prefix_caching": True,
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": int(
            os.environ.get("VERL_GPU_TEST_UPDATE_BUCKET_MB", "16")
        ),
        "actor_rollout_ref.rollout.engine_kwargs.vllm": {},
        "async_training.partial_rollout": True,
        "data.max_prompt_length": max_prompt_len,
        "data.max_response_length": max_response_len,
    }
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


def _tokenize_prompt(model_path: str) -> list[int]:
    from transformers import AutoTokenizer

    trust_remote_code = _bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    prompt = os.environ.get("VERL_GPU_TEST_PROMPT", "2 + 3 =")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_ids = prompt_ids[: int(os.environ.get("VERL_GPU_TEST_PROMPT_LEN", "24"))]
    if not prompt_ids:
        pytest.fail("GPU test prompt produced no token ids")
    return prompt_ids


def _assert_finite(values: list[float]) -> None:
    assert values
    for value in values:
        assert isinstance(value, float)
        assert math.isfinite(value)


def _coverage_env_vars() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key == "PYTHONPATH" or key.startswith("VERL_EVAL_COVERAGE_")
    }


def _runtime_env_vars() -> dict[str, str]:
    return {
        **_coverage_env_vars(),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO"),
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "1"),
    }


@contextmanager
def _temporary_env(**updates: str):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _zeroed_hf_weight_items(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
        local_files_only=True,
    )
    for name, tensor in model.state_dict().items():
        yield name, torch.zeros_like(tensor, device="cpu").contiguous()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def real_vllm_context():
    model_path = _require_gpu_model()

    import ray

    from verl.workers.rollout.llm_server import LLMServerManager

    prompt_ids = _tokenize_prompt(model_path)
    config = _compose_real_vllm_config(model_path)
    generate_tokens = int(os.environ.get("VERL_GPU_TEST_GENERATE_TOKENS", "2"))

    ray.shutdown()
    ray.init(
        num_gpus=int(os.environ.get("VERL_GPU_TEST_GPUS", "1")),
        ignore_reinit_error=True,
        runtime_env={"env_vars": _runtime_env_vars()},
    )

    manager = None
    try:
        maybe_manager = LLMServerManager.create(config=config, worker_group=None, rollout_resource_pool=None)
        manager = await maybe_manager if inspect.isawaitable(maybe_manager) else maybe_manager
        yield SimpleNamespace(
            model_path=model_path,
            prompt_ids=prompt_ids,
            config=config,
            generate_tokens=generate_tokens,
            manager=manager,
            client=manager.get_client(),
        )
    finally:
        if manager is not None:
            for replica in manager.get_replicas():
                try:
                    await replica.abort_all_requests()
                    await replica.resume_generation()
                except Exception:
                    pass
        ray.shutdown()


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_manager_starts_one_standalone_replica(real_vllm_context):
    manager = real_vllm_context.manager
    assert len(manager.get_addresses()) == 1
    assert len(manager.get_replicas()) == 1
    status = await manager.global_load_balancer.get_status.remote()
    assert status["active_servers"] == 1
    assert status["total_inflight"] == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_token_generation_logprobs_and_sampling_budgets(real_vllm_context):
    ctx = real_vllm_context

    output = await ctx.client.generate(
        request_id="gpu-basic-generation",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": ctx.generate_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "prompt_logprobs": 1,
        },
    )
    assert 1 <= len(output.token_ids) <= ctx.generate_tokens
    assert output.stop_reason == "completed"
    _assert_finite(output.log_probs)
    assert len(output.log_probs) == len(output.token_ids)
    assert len(output.extra_fields["prompt_logprobs"]) == len(ctx.prompt_ids)
    assert all(len(row) == 1 for row in output.extra_fields["prompt_logprobs"])

    prompt_only_logprobs = await ctx.client.generate(
        request_id="gpu-prompt-logprob-zero",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "prompt_logprobs": 0,
        },
    )
    assert prompt_only_logprobs.stop_reason == "completed"
    assert len(prompt_only_logprobs.extra_fields["prompt_logprobs"]) == len(ctx.prompt_ids)
    assert all(len(row) == 1 for row in prompt_only_logprobs.extra_fields["prompt_logprobs"])

    max_new_tokens_output = await ctx.client.generate(
        request_id="gpu-max-new-tokens",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_new_tokens": ctx.generate_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "ignore_eos": True,
        },
    )
    assert 1 <= len(max_new_tokens_output.token_ids) <= ctx.generate_tokens
    assert max_new_tokens_output.stop_reason == "completed"
    _assert_finite(max_new_tokens_output.log_probs)

    default_budget_output = await ctx.client.generate(
        request_id="gpu-default-response-budget",
        prompt_ids=ctx.prompt_ids,
        sampling_params={"temperature": 0.0, "top_p": 1.0},
    )
    assert 1 <= len(default_budget_output.token_ids) <= ctx.config.actor_rollout_ref.rollout.response_length
    assert default_budget_output.stop_reason == "completed"
    assert default_budget_output.log_probs is None


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_parallel_requests_and_load_balancer_drain(real_vllm_context):
    ctx = real_vllm_context
    parallel_outputs = await asyncio.gather(
        *[
            ctx.client.generate(
                request_id=f"gpu-parallel-{idx}",
                prompt_ids=ctx.prompt_ids,
                sampling_params={
                    "max_tokens": ctx.generate_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "logprobs": True,
                },
            )
            for idx in range(2)
        ]
    )
    for parallel_output in parallel_outputs:
        assert 1 <= len(parallel_output.token_ids) <= ctx.generate_tokens
        assert parallel_output.stop_reason == "completed"
        _assert_finite(parallel_output.log_probs)

    status = await ctx.manager.global_load_balancer.get_status.remote()
    assert status["total_inflight"] == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_openai_completion_endpoint(real_vllm_context):
    from openai import AsyncOpenAI

    ctx = real_vllm_context
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        os.environ.pop(key, None)

    openai_client = AsyncOpenAI(api_key="eval", base_url=f"http://{ctx.manager.get_addresses()[0]}/v1")
    completion = await openai_client.completions.create(
        model=ctx.model_path,
        prompt="2 + 3 =",
        max_tokens=ctx.generate_tokens,
        temperature=0.0,
        logprobs=1,
    )

    assert completion.id
    assert completion.choices
    choice = completion.choices[0]
    assert choice.finish_reason in {"length", "stop"}
    assert choice.logprobs is not None


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_context_limit_and_lifecycle_controls(real_vllm_context):
    ctx = real_vllm_context
    full_context_prompt = ctx.prompt_ids + [ctx.prompt_ids[-1]] * (
        ctx.config.actor_rollout_ref.rollout.max_model_len - len(ctx.prompt_ids)
    )
    with pytest.raises(Exception) as no_budget_error:
        await ctx.client.generate(
            request_id="gpu-no-generation-budget",
            prompt_ids=full_context_prompt,
            sampling_params={"max_tokens": 1, "temperature": 0.0},
        )
    assert "leaves no room" in str(no_budget_error.value) or "maximum context length" in str(no_budget_error.value)

    await ctx.manager.start_profile(trace_id="gpu-rollout-smoke")
    await ctx.manager.stop_profile()

    replica = ctx.manager.get_replicas()[0]
    abort_report = await replica.abort_all_requests()
    assert set(abort_report) >= {"aborted_count", "request_ids"}
    await replica.resume_generation()
    missing_abort = await replica.abort_request("gpu-missing-request")
    assert missing_abort.get("aborted") is False
    await replica.clear_kv_cache()
    await replica.release_kv_cache()
    await replica.resume_kv_cache()
    await replica.sleep()
    await replica.wake_up()

    resumed_output = await ctx.client.generate(
        request_id="gpu-lifecycle-generate-after-resume",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": ctx.generate_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
        },
    )
    assert 1 <= len(resumed_output.token_ids) <= ctx.generate_tokens
    assert resumed_output.stop_reason == "completed"
    _assert_finite(resumed_output.log_probs)


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_fully_async_client_contract(real_vllm_context):
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient

    ctx = real_vllm_context
    fully_async_client = ctx.manager.get_client(client_cls=FullyAsyncLLMServerClient)
    resumed_output = await fully_async_client.generate(
        request_id="gpu-fully-async-contract",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_new_tokens": ctx.generate_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
        },
    )

    assert 1 <= len(resumed_output.token_ids) <= ctx.generate_tokens
    assert resumed_output.stop_reason in {"completed", "length"}
    _assert_finite(resumed_output.log_probs)
    assert len(resumed_output.log_probs) == len(resumed_output.token_ids)
    assert set(resumed_output.extra_fields) >= {"global_steps", "min_global_steps", "max_global_steps"}


@pytest.mark.asyncio(loop_scope="module")
async def test_real_vllm_weight_update_changes_generation_distribution_and_step(real_vllm_context):
    from transformers import AutoTokenizer

    from verl.workers.rollout.base import get_rollout_class

    ctx = real_vllm_context
    before = await ctx.client.generate(
        request_id="gpu-weight-update-before",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
        },
    )
    _assert_finite(before.log_probs)

    rollout_cls = get_rollout_class("vllm", "async")
    with _temporary_env(RANK="0", RAY_LOCAL_WORLD_SIZE=str(int(os.environ.get("VERL_GPU_TEST_GPUS", "1")))):
        rollout = rollout_cls(
            config=ctx.config.actor_rollout_ref.rollout,
            model_config=ctx.config.actor_rollout_ref.model,
            device_mesh=None,
            replica_rank=0,
        )
        await rollout.update_weights(_zeroed_hf_weight_items(ctx.model_path), global_steps=123)

    after = await ctx.client.generate(
        request_id="gpu-weight-update-after",
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "prompt_logprobs": 0,
        },
    )
    _assert_finite(after.log_probs)

    tokenizer = AutoTokenizer.from_pretrained(
        ctx.model_path,
        trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
        local_files_only=True,
    )
    expected_uniform_logprob = -math.log(len(tokenizer))
    assert abs(after.log_probs[0] - expected_uniform_logprob) < 0.25
    assert abs(after.log_probs[0] - before.log_probs[0]) > 0.05
    assert after.extra_fields["global_steps"] == 123
