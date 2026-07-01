import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import gc
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_real_vllm_rollout_gpu import (
    _assert_finite,
    _bool_env,
    _compose_real_vllm_config,
    _coverage_env_vars,
    _require_gpu_model,
    _temporary_env,
    _tokenize_prompt,
    _zeroed_hf_weight_items,
)


async def _ray_shutdown_cooldown() -> None:
    import ray

    ray.shutdown()
    gc.collect()
    cooldown = float(os.environ.get("VERL_GPU_TEST_CONTEXT_COOLDOWN_SEC", "12"))
    if cooldown > 0:
        await asyncio.sleep(cooldown)


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


def _cuda_device_count() -> int:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU rollout tests")
    return torch.cuda.device_count()


def _require_cuda_count(count: int) -> None:
    available = _cuda_device_count()
    if available < count:
        pytest.skip(f"requires at least {count} visible CUDA devices, got {available}")


@asynccontextmanager
async def _managed_vllm_context(
    model_path: str,
    *,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    config_updates: dict[str, object] | None = None,
):
    import ray
    from omegaconf import OmegaConf

    from verl.workers.rollout.llm_server import LLMServerManager

    prompt_ids = _tokenize_prompt(model_path)
    config = _compose_real_vllm_config(model_path)
    updates = {
        "trainer.n_gpus_per_node": num_gpus,
        "actor_rollout_ref.rollout.n_gpus_per_node": num_gpus,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": tensor_parallel_size,
        "actor_rollout_ref.rollout.data_parallel_size": data_parallel_size,
        "actor_rollout_ref.rollout.max_num_seqs": int(os.environ.get("VERL_GPU_TEST_MAX_NUM_SEQS", "2")),
        "actor_rollout_ref.rollout.max_num_batched_tokens": int(
            os.environ.get(
                "VERL_GPU_TEST_MAX_NUM_BATCHED_TOKENS",
                str(config.actor_rollout_ref.rollout.max_model_len * int(os.environ.get("VERL_GPU_TEST_MAX_NUM_SEQS", "2"))),
            )
        ),
        "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes": int(
            os.environ.get("VERL_GPU_TEST_KV_CACHE_BYTES", str(64 << 20))
        ),
    }
    if config_updates:
        updates.update(config_updates)
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)

    await _ray_shutdown_cooldown()
    ray.init(
        num_gpus=num_gpus,
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
            generate_tokens=int(os.environ.get("VERL_GPU_TEST_GENERATE_TOKENS", "2")),
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
        await _ray_shutdown_cooldown()


async def _generate_one(ctx, request_id: str):
    output = await ctx.client.generate(
        request_id=request_id,
        prompt_ids=ctx.prompt_ids,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "prompt_logprobs": 0,
        },
    )
    assert output.stop_reason == "completed"
    assert len(output.token_ids) == 1
    _assert_finite(output.log_probs)
    return output


async def _sync_zeroed_weights_through_rollout(ctx, *, global_steps: int):
    from verl.workers.rollout.base import get_rollout_class

    rollout_cls = get_rollout_class("vllm", "async")
    with _temporary_env(RANK="0", RAY_LOCAL_WORLD_SIZE="1"):
        rollout = rollout_cls(
            config=ctx.config.actor_rollout_ref.rollout,
            model_config=ctx.config.actor_rollout_ref.model,
            device_mesh=None,
            replica_rank=0,
        )
        await rollout.update_weights(_zeroed_hf_weight_items(ctx.model_path), global_steps=global_steps)


def _make_lora_update(model_path: str, rank: int = 2):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
        local_files_only=True,
    )
    peft_model = get_peft_model(model, lora_config)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                param.fill_(0.10)
            elif "lora_B" in name:
                param.fill_(0.10)

    state = get_peft_model_state_dict(peft_model)
    peft_config = {key: value for key, value in asdict(lora_config).items() if value is not None}
    tensors = [(name, tensor.detach().cpu().contiguous()) for name, tensor in state.items()]
    del peft_model, model
    gc.collect()
    return peft_config, tensors


def _make_random_qwen2_model(
    output_dir: Path,
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 160,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
) -> str:
    if output_dir.exists():
        return str(output_dir)

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    base_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "+",
        "-",
        "*",
        "/",
        "=",
        "?",
        ".",
        ",",
        ":",
        "Question",
        "Answer",
        "hello",
        "world",
    ]
    vocab = {token: idx for idx, token in enumerate(base_tokens)}
    for idx in range(len(vocab), vocab_size):
        vocab[f"tok{idx}"] = idx
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=True,
    )
    model = Qwen2ForCausalLM(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    del model
    gc.collect()
    return str(output_dir)


def _make_large_vocab_model(output_dir: Path) -> str:
    return _make_random_qwen2_model(output_dir, vocab_size=8192)


def _make_fp8_block_model(output_dir: Path) -> str:
    return _make_random_qwen2_model(
        output_dir,
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
    )


@pytest.mark.asyncio
async def test_real_vllm_lora_adapter_weight_sync_changes_generation_behavior():
    pytest.importorskip("peft")
    model_path = _require_gpu_model()
    rank = int(os.environ.get("VERL_GPU_TEST_LORA_RANK", "2"))

    async with _managed_vllm_context(
        model_path,
        config_updates={
            "actor_rollout_ref.model.lora_rank": rank,
            "actor_rollout_ref.model.lora_alpha": rank * 8,
            "actor_rollout_ref.model.target_modules": ["q_proj", "v_proj"],
            "actor_rollout_ref.model.lora.rank": rank,
            "actor_rollout_ref.model.lora.alpha": rank * 8,
            "actor_rollout_ref.model.lora.merge": False,
            "actor_rollout_ref.model.lora.target_modules": ["q_proj", "v_proj"],
        },
    ) as ctx:
        before = await _generate_one(ctx, "gpu-lora-before")

        from verl.workers.rollout.base import get_rollout_class

        peft_config, lora_tensors = _make_lora_update(ctx.model_path, rank=rank)
        rollout_cls = get_rollout_class("vllm", "async")
        with _temporary_env(RANK="0", RAY_LOCAL_WORLD_SIZE="1"):
            rollout = rollout_cls(
                config=ctx.config.actor_rollout_ref.rollout,
                model_config=ctx.config.actor_rollout_ref.model,
                device_mesh=None,
                replica_rank=0,
            )
            await rollout.update_weights(
                iter(lora_tensors),
                peft_config=peft_config,
                base_sync_done=True,
                global_steps=231,
            )

        after = await _generate_one(ctx, "gpu-lora-after")
        assert after.extra_fields["global_steps"] == 231
        changed_token = after.token_ids[0] != before.token_ids[0]
        changed_logprob = abs(after.log_probs[0] - before.log_probs[0]) > 1e-4
        assert changed_token or changed_logprob


@pytest.mark.asyncio
async def test_real_vllm_small_bucket_weight_sync_preserves_update_contract():
    model_path = _require_gpu_model()
    async with _managed_vllm_context(
        model_path,
        config_updates={"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 1},
    ) as ctx:
        before = await _generate_one(ctx, "gpu-small-bucket-before")

        from transformers import AutoTokenizer

        await _sync_zeroed_weights_through_rollout(ctx, global_steps=321)

        after = await _generate_one(ctx, "gpu-small-bucket-after")
        tokenizer = AutoTokenizer.from_pretrained(
            ctx.model_path,
            trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
            local_files_only=True,
        )
        assert after.extra_fields["global_steps"] == 321
        assert abs(after.log_probs[0] + math.log(len(tokenizer))) < 0.25
        assert abs(after.log_probs[0] - before.log_probs[0]) > 0.05


@pytest.mark.asyncio
async def test_real_vllm_direct_ipc_large_tensor_weight_sync_preserves_update_contract(tmp_path: Path):
    if not _bool_env("VERL_GPU_TEST_ENABLE_DIRECT_IPC_LARGE", False):
        pytest.skip("VERL_GPU_TEST_ENABLE_DIRECT_IPC_LARGE disabled")
    _require_gpu_model()
    model_path = _make_large_vocab_model(tmp_path / "large-direct-ipc-model")
    async with _managed_vllm_context(
        model_path,
        config_updates={"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 1},
    ) as ctx:
        before = await _generate_one(ctx, "gpu-direct-ipc-before")

        from transformers import AutoTokenizer

        from verl.workers.rollout.base import get_rollout_class

        rollout_cls = get_rollout_class("vllm", "async")
        with _temporary_env(RANK="0", RAY_LOCAL_WORLD_SIZE="1"):
            rollout = rollout_cls(
                config=ctx.config.actor_rollout_ref.rollout,
                model_config=ctx.config.actor_rollout_ref.model,
                device_mesh=None,
                replica_rank=0,
            )
            await rollout.update_weights(_zeroed_hf_weight_items(ctx.model_path), global_steps=654)

        after = await _generate_one(ctx, "gpu-direct-ipc-after")
        tokenizer = AutoTokenizer.from_pretrained(
            ctx.model_path,
            trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
            local_files_only=True,
        )
        assert after.extra_fields["global_steps"] == 654
        assert abs(after.log_probs[0] + math.log(len(tokenizer))) < 0.25
        assert abs(after.log_probs[0] - before.log_probs[0]) > 0.05


@pytest.mark.asyncio
async def test_real_vllm_online_fp8_quantization_generation_contract(tmp_path: Path):
    if not _bool_env("VERL_GPU_TEST_ENABLE_FP8", True):
        pytest.skip("VERL_GPU_TEST_ENABLE_FP8 disabled")
    _require_gpu_model()
    model_path = _make_fp8_block_model(tmp_path / "fp8-block-model")

    async with _managed_vllm_context(
        model_path,
        config_updates={
            "actor_rollout_ref.rollout.quantization": "fp8",
            "actor_rollout_ref.rollout.dtype": os.environ.get("VERL_GPU_TEST_FP8_DTYPE", "auto"),
        },
    ) as ctx:
        output = await _generate_one(ctx, "gpu-fp8-generation")
        assert output.token_ids


@pytest.mark.asyncio
async def test_real_vllm_online_fp8_weight_sync_quantizes_updated_weights(tmp_path: Path):
    if not _bool_env("VERL_GPU_TEST_ENABLE_FP8_WEIGHT_SYNC", False):
        pytest.skip("VERL_GPU_TEST_ENABLE_FP8_WEIGHT_SYNC disabled")
    if not _bool_env("VERL_GPU_TEST_ENABLE_FP8", True):
        pytest.skip("VERL_GPU_TEST_ENABLE_FP8 disabled")
    _cuda_device_count()
    model_path = _make_fp8_block_model(tmp_path / "fp8-weight-sync-model")

    async with _managed_vllm_context(
        model_path,
        config_updates={
            "actor_rollout_ref.rollout.quantization": "fp8",
            "actor_rollout_ref.rollout.dtype": os.environ.get("VERL_GPU_TEST_FP8_DTYPE", "auto"),
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 1,
        },
    ) as ctx:
        before = await _generate_one(ctx, "gpu-fp8-sync-before")
        await _sync_zeroed_weights_through_rollout(ctx, global_steps=877)
        after = await _generate_one(ctx, "gpu-fp8-sync-after")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            ctx.model_path,
            trust_remote_code=_bool_env("VERL_GPU_TEST_TRUST_REMOTE_CODE", False),
            local_files_only=True,
        )
        assert after.extra_fields["global_steps"] == 877
        assert abs(after.log_probs[0] + math.log(len(tokenizer))) < 0.35
        assert abs(after.log_probs[0] - before.log_probs[0]) > 0.05


@pytest.mark.asyncio
async def test_real_vllm_speculative_rollout_reports_draft_acceptance_metrics():
    if not _bool_env("VERL_GPU_TEST_ENABLE_SPECULATIVE", False):
        pytest.skip("VERL_GPU_TEST_ENABLE_SPECULATIVE disabled")
    model_path = _require_gpu_model()
    method = os.environ.get("VERL_GPU_TEST_SPECULATIVE_METHOD", "ngram")

    async with _managed_vllm_context(
        model_path,
        config_updates={
            "actor_rollout_ref.model.mtp.enable": True,
            "actor_rollout_ref.model.mtp.enable_rollout": True,
            "actor_rollout_ref.model.mtp.method": method,
            "actor_rollout_ref.model.mtp.num_speculative_tokens": 1,
            "actor_rollout_ref.rollout.disable_log_stats": False,
        },
    ) as ctx:
        output = await ctx.client.generate(
            request_id="gpu-speculative-generation",
            prompt_ids=ctx.prompt_ids,
            sampling_params={
                "max_tokens": max(2, ctx.generate_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": True,
            },
        )
        assert output.stop_reason == "completed"
        _assert_finite(output.log_probs)
        assert set(output.extra_fields) >= {
            "spec_num_draft_tokens",
            "spec_num_accepted_tokens",
            "spec_num_verify_steps",
        }
        assert output.extra_fields["spec_num_draft_tokens"] >= 0
        assert output.extra_fields["spec_num_accepted_tokens"] >= 0
        assert output.extra_fields["spec_num_verify_steps"] >= 0


@pytest.mark.asyncio
async def test_real_vllm_tensor_parallel_generation_on_two_gpus_when_available():
    if not _bool_env("VERL_GPU_TEST_ENABLE_MULTI_GPU", True):
        pytest.skip("VERL_GPU_TEST_ENABLE_MULTI_GPU disabled")
    _require_cuda_count(2)
    model_path = _require_gpu_model()

    async with _managed_vllm_context(model_path, num_gpus=2, tensor_parallel_size=2) as ctx:
        assert len(ctx.manager.get_replicas()) == 1
        await _generate_one(ctx, "gpu-tp2-generation")


@pytest.mark.asyncio
async def test_real_vllm_data_parallel_generation_on_two_gpus_when_available():
    if not _bool_env("VERL_GPU_TEST_ENABLE_MULTI_GPU", True):
        pytest.skip("VERL_GPU_TEST_ENABLE_MULTI_GPU disabled")
    _require_cuda_count(2)
    model_path = _require_gpu_model()

    async with _managed_vllm_context(model_path, num_gpus=2, data_parallel_size=2) as ctx:
        outputs = await asyncio.gather(
            *[_generate_one(ctx, f"gpu-dp2-generation-{idx}") for idx in range(4)]
        )
        assert len(outputs) == 4
        status = await ctx.manager.global_load_balancer.get_status.remote()
        assert status["total_inflight"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_name", "dtype_env_name"),
    [
        ("VERL_GPU_TEST_QUANT_MODEL", "VERL_GPU_TEST_QUANT_DTYPE"),
        ("VERL_GPU_TEST_QAT_MODEL", "VERL_GPU_TEST_QAT_DTYPE"),
        ("VERL_GPU_TEST_MODELOPT_MODEL", "VERL_GPU_TEST_MODELOPT_DTYPE"),
    ],
)
async def test_real_vllm_optional_quantized_model_generation_contract(env_name: str, dtype_env_name: str):
    model_path = os.environ.get(env_name)
    if not model_path:
        pytest.skip(f"{env_name} is not configured")
    if not Path(model_path).exists():
        pytest.skip(f"{env_name} does not exist: {model_path}")
    _cuda_device_count()

    async with _managed_vllm_context(
        model_path,
        config_updates={"actor_rollout_ref.rollout.dtype": os.environ.get(dtype_env_name, "auto")},
    ) as ctx:
        output = await _generate_one(ctx, f"gpu-{env_name.lower()}-generation")
        assert output.token_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_name", "dtype_env_name"),
    [
        ("VERL_GPU_TEST_QUANT_MODEL", "VERL_GPU_TEST_QUANT_DTYPE"),
        ("VERL_GPU_TEST_QAT_MODEL", "VERL_GPU_TEST_QAT_DTYPE"),
        ("VERL_GPU_TEST_MODELOPT_MODEL", "VERL_GPU_TEST_MODELOPT_DTYPE"),
    ],
)
async def test_real_vllm_optional_quantized_model_weight_sync_contract(env_name: str, dtype_env_name: str):
    model_path = os.environ.get(env_name)
    if not model_path:
        pytest.skip(f"{env_name} is not configured")
    if not Path(model_path).exists():
        pytest.skip(f"{env_name} does not exist: {model_path}")
    _cuda_device_count()

    async with _managed_vllm_context(
        model_path,
        config_updates={
            "actor_rollout_ref.rollout.dtype": os.environ.get(dtype_env_name, "auto"),
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 1,
        },
    ) as ctx:
        before = await _generate_one(ctx, f"gpu-{env_name.lower()}-sync-before")
        await _sync_zeroed_weights_through_rollout(ctx, global_steps=932)
        after = await _generate_one(ctx, f"gpu-{env_name.lower()}-sync-after")
        assert after.extra_fields["global_steps"] == 932
        assert after.token_ids
        assert abs(after.log_probs[0] - before.log_probs[0]) > 1e-4
