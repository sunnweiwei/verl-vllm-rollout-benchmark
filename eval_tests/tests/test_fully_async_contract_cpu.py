import pytest
import torch

from vllm_eval_helpers import install_fully_async_import_stubs, make_config, patch_vllm_standalone_launch


@pytest.mark.asyncio
async def test_fully_async_client_resumes_after_aborted_generation(monkeypatch):
    install_fully_async_import_stubs(monkeypatch)

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [
        {
            "token_ids": [101, 102],
            "log_probs": [-1.0, -0.8],
            "stop_reason": "aborted",
            "num_preempted": 1,
            "extra_fields": {"global_steps": 3},
        },
        {
            "token_ids": [103],
            "log_probs": [-0.2],
            "stop_reason": "completed",
            "num_preempted": 2,
            "extra_fields": {"global_steps": 4},
        },
    ]
    patch_vllm_standalone_launch(monkeypatch, outputs)

    manager = await LLMServerManager.create(config=make_config(partial_rollout=True))
    client = manager.get_client(client_cls=FullyAsyncLLMServerClient)

    result = await client.generate(
        request_id="resume-me",
        prompt_ids=[1, 2],
        sampling_params={"temperature": 1.0, "max_tokens": 3, "logprobs": True},
    )

    assert result.token_ids == [101, 102, 103]
    assert result.log_probs == [-1.0, -0.8, -0.2]
    assert result.stop_reason == "length"
    assert result.num_preempted == 3
    assert result.extra_fields["global_steps"] == 4
    assert result.extra_fields["min_global_steps"] == 3
    assert result.extra_fields["max_global_steps"] == 4

    calls = await manager.server_handles[0].get_calls.remote()
    assert calls[0]["prompt_ids"] == [1, 2]
    assert calls[0]["sampling_params"]["max_tokens"] == 3
    assert calls[1]["prompt_ids"] == [1, 2, 101, 102]
    assert calls[1]["sampling_params"]["max_tokens"] == 1


@pytest.mark.asyncio
async def test_fully_async_client_stops_on_abort_when_partial_rollout_disabled(monkeypatch):
    install_fully_async_import_stubs(monkeypatch)

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [
        {
            "token_ids": [5],
            "log_probs": [-0.5],
            "stop_reason": "aborted",
            "extra_fields": {"global_steps": 8},
        },
        {"token_ids": [6], "stop_reason": "completed", "extra_fields": {"global_steps": 9}},
    ]
    patch_vllm_standalone_launch(monkeypatch, outputs)
    manager = await LLMServerManager.create(config=make_config(partial_rollout=False))

    result = await manager.get_client(client_cls=FullyAsyncLLMServerClient).generate(
        request_id="no-resume",
        prompt_ids=[1],
        sampling_params={"max_tokens": 3, "logprobs": True},
    )

    assert result.token_ids == [5]
    assert result.stop_reason == "aborted"
    assert result.extra_fields["min_global_steps"] == 8
    assert len(await manager.server_handles[0].get_calls.remote()) == 1


@pytest.mark.asyncio
async def test_fully_async_client_merges_routed_experts_for_new_tokens(monkeypatch):
    install_fully_async_import_stubs(monkeypatch)

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
    from verl.workers.rollout.llm_server import LLMServerManager

    first_routing = torch.tensor([[[1]], [[2]], [[3]]])
    second_routing = torch.tensor([[[7]], [[8]], [[9]], [[10]]])
    outputs = [
        {
            "token_ids": [11, 12],
            "routed_experts": first_routing,
            "stop_reason": "aborted",
            "extra_fields": {"global_steps": 1},
        },
        {
            "token_ids": [13, 14],
            "routed_experts": second_routing,
            "stop_reason": "completed",
            "extra_fields": {"global_steps": 2},
        },
    ]
    patch_vllm_standalone_launch(monkeypatch, outputs)
    manager = await LLMServerManager.create(config=make_config(partial_rollout=True))

    result = await manager.get_client(client_cls=FullyAsyncLLMServerClient).generate(
        request_id="router",
        prompt_ids=[1],
        sampling_params={"max_tokens": 4},
    )

    assert result.token_ids == [11, 12, 13, 14]
    assert result.routed_experts.shape == (5, 1, 1)
    assert result.routed_experts[-2:].flatten().tolist() == [9, 10]


@pytest.mark.asyncio
async def test_fully_async_client_forwards_multimodal_kwargs_on_resume(monkeypatch):
    install_fully_async_import_stubs(monkeypatch)

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [
        {"token_ids": [1], "stop_reason": "aborted", "extra_fields": {"global_steps": 1}},
        {"token_ids": [2], "stop_reason": "completed", "extra_fields": {"global_steps": 2}},
    ]
    patch_vllm_standalone_launch(monkeypatch, outputs)
    manager = await LLMServerManager.create(config=make_config(partial_rollout=True))

    await manager.get_client(client_cls=FullyAsyncLLMServerClient).generate(
        request_id="mm",
        prompt_ids=[9],
        sampling_params={"max_new_tokens": 2},
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"sampling_rate": 8000},
    )

    calls = await manager.server_handles[0].get_calls.remote()
    assert len(calls) == 2
    for call in calls:
        assert call["image_data"] == ["image"]
        assert call["video_data"] == ["video"]
        assert call["audio_data"] == ["audio"]
        assert call["mm_processor_kwargs"] == {"sampling_rate": 8000}


@pytest.mark.asyncio
async def test_fully_async_manager_adds_and_removes_hybrid_replicas(monkeypatch):
    install_fully_async_import_stubs(monkeypatch)

    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager

    manager = FullyAsyncLLMServerManager.__new__(FullyAsyncLLMServerManager)
    manager.hybrid_replicas = {}
    manager.alive_replicas = {}
    manager.alive_addresses = {}
    manager.server_handles = []
    manager.server_addresses = []
    manager.rollout_replicas = []
    manager.last_hybrid_add_time = 0

    from vllm_eval_helpers import ScriptedRolloutServer

    server_actor = ScriptedRolloutServer.remote(label="hybrid")
    replica = type("Replica", (), {"_server_address": "hybrid://0", "_server_handle": server_actor})()
    manager.hybrid_replicas["hybrid_0"] = replica

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    manager.global_load_balancer = GlobalRequestLoadBalancer.remote({"standalone://0": server_actor})

    assert await manager.add_replicas(["hybrid_0"]) == 1
    assert "hybrid_0" in manager.alive_replicas
    status = await manager.global_load_balancer.get_status.remote()
    assert "hybrid://0" in status["registered_handles"]

    assert await manager.remove_replicas(["hybrid_0"]) == 1
    assert "hybrid_0" not in manager.alive_replicas
    status = await manager.global_load_balancer.get_status.remote()
    assert "hybrid://0" not in status["registered_handles"]
