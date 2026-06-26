import pytest
from omegaconf import OmegaConf

from vllm_eval_helpers import ScriptedRolloutServer, make_config, patch_vllm_standalone_launch


@pytest.mark.asyncio
async def test_vllm_backend_runs_through_manager_and_client(monkeypatch):
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [
        {
            "token_ids": [31, 32, 33],
            "log_probs": [-0.3, -0.2, -0.1],
            "stop_reason": "completed",
            "extra_fields": {"global_steps": 7, "source": "fake-vllm"},
        }
    ]
    created = patch_vllm_standalone_launch(monkeypatch, outputs)

    manager = await LLMServerManager.create(config=make_config())
    assert manager.get_addresses() == ["fake-vllm://replica-0"]
    assert len(created) == 1

    result = await manager.get_client().generate(
        request_id="sticky-user-request",
        prompt_ids=[11, 12],
        sampling_params={"temperature": 0.0, "max_tokens": 3, "logprobs": True},
        image_data=["image-payload"],
        video_data=["video-payload"],
        audio_data=["audio-payload"],
        mm_processor_kwargs={"sampling_rate": 16000},
    )

    assert result.token_ids == [31, 32, 33]
    assert result.log_probs == [-0.3, -0.2, -0.1]
    assert result.stop_reason == "completed"
    assert result.extra_fields["global_steps"] == 7

    calls = await manager.server_handles[0].get_calls.remote()
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt_ids"] == [11, 12]
    assert call["sampling_params"] == {"temperature": 0.0, "max_tokens": 3, "logprobs": True}
    assert call["image_data"] == ["image-payload"]
    assert call["video_data"] == ["video-payload"]
    assert call["audio_data"] == ["audio-payload"]
    assert call["mm_processor_kwargs"] == {"sampling_rate": 16000}
    assert call["request_id"] != "sticky-user-request"

    status = await manager.global_load_balancer.get_status.remote()
    assert status["total_inflight"] == 0
    assert status["active_servers"] == 1


@pytest.mark.asyncio
async def test_vllm_reward_model_rollout_can_use_separate_manager(monkeypatch):
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [{"token_ids": [4], "stop_reason": "completed", "extra_fields": {"global_steps": 2}}]
    created = patch_vllm_standalone_launch(monkeypatch, outputs)
    cfg = make_config()
    cfg.reward.reward_model.enable = True
    cfg.reward.reward_model.rollout.name = "vllm"
    cfg.actor_rollout_ref.rollout = cfg.reward.reward_model.rollout

    manager = await LLMServerManager.create(config=cfg)
    result = await manager.get_client().generate(
        request_id="reward",
        prompt_ids=[1],
        sampling_params={"max_tokens": 1},
    )

    assert result.token_ids == [4]
    assert created[0].config.name == "vllm"


def test_rollout_config_exposes_backend_engine_kwargs_without_dropping_other_backends():
    cfg = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")

    assert "vllm" in cfg.engine_kwargs
    assert "sglang" in cfg.engine_kwargs
    assert "trtllm" in cfg.engine_kwargs

    rollout_cfg = make_config().actor_rollout_ref.rollout
    assert rollout_cfg.name == "vllm"
    assert rollout_cfg.engine_kwargs["vllm"]["sentinel"] == "preserved"
    assert rollout_cfg.engine_kwargs["sglang"]["sentinel"] == "sglang"
    assert rollout_cfg.engine_kwargs["trtllm"]["sentinel"] == "trtllm"


def test_vllm_rejects_prefill_decode_disaggregation_but_sglang_accepts_it():
    from verl.workers.config import DisaggregationConfig, RolloutConfig

    with pytest.raises(ValueError, match="disaggregation.enabled=True"):
        RolloutConfig(name="vllm", disaggregation=DisaggregationConfig(enabled=True))

    cfg = RolloutConfig(name="sglang", disaggregation=DisaggregationConfig(enabled=True))
    assert cfg.disaggregation.enabled is True


def test_non_vllm_rollout_names_remain_selectable_when_installed():
    from verl.workers.rollout.replica import get_rollout_replica_class

    for name in ("sglang", "trtllm"):
        try:
            cls = get_rollout_replica_class(name)
        except (ImportError, ModuleNotFoundError):
            continue
        assert cls is not None


@pytest.mark.asyncio
async def test_load_balancer_preserves_sticky_sessions_and_clears_inflight():
    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    server_a = ScriptedRolloutServer.remote(label="a")
    server_b = ScriptedRolloutServer.remote(label="b")
    lb = GlobalRequestLoadBalancer.remote({"a": server_a, "b": server_b})

    first_id, first_handle = await lb.acquire_server.remote("sample-1")
    second_id, second_handle = await lb.acquire_server.remote("sample-1")
    assert first_id == second_id
    assert first_handle == second_handle

    await lb.release_server.remote(first_id)
    await lb.release_server.remote(second_id)
    status = await lb.get_status.remote()
    assert status["total_inflight"] == 0
