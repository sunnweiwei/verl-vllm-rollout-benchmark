import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_eval_helpers import (
    AnyLoadedLoraSet,
    FakeTokenLogprob,
    make_completion_output,
    make_request_output,
    make_vllm_http_server,
)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _set_server_global_steps(server, steps: int):
    setter = getattr(server, "set_global_steps", None)
    if setter is None:
        server.global_steps = steps
    else:
        await _maybe_await(setter(steps))


def _multimodal_data_from_call(call):
    prompt = call.get("prompt")
    if isinstance(prompt, dict):
        data = prompt.get("multi_modal_data")
        if data is not None:
            return data
    elif prompt is not None:
        data = getattr(prompt, "multi_modal_data", None)
        if data is not None:
            return data

    return call.get("multi_modal_data")


def _mm_processor_kwargs_from_call(call):
    prompt = call.get("prompt")
    if isinstance(prompt, dict):
        data = prompt.get("mm_processor_kwargs")
        if data is not None:
            return data
    elif prompt is not None:
        data = getattr(prompt, "mm_processor_kwargs", None)
        if data is not None:
            return data

    return call.get("mm_processor_kwargs")


def _prompt_token_ids_from_call(call):
    prompt = call.get("prompt")
    if isinstance(prompt, dict):
        return prompt.get("prompt_token_ids")
    if prompt is not None:
        prompt_ids = getattr(prompt, "prompt_token_ids", None)
        if prompt_ids is not None:
            return prompt_ids
    return call.get("prompt_token_ids")


class FakeQwen2VLImageProcessor:
    pass


@pytest.mark.asyncio
async def test_generate_returns_tokens_and_logprobs(monkeypatch):
    final = make_request_output(
        [make_completion_output([7, 8], logprobs=[-0.7, -0.8], finish_reason="length", num_preempted=2)]
    )
    server, engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 2, "temperature": 0.1, "logprobs": True},
        request_id="r1",
    )

    assert output.token_ids == [7, 8]
    assert output.log_probs == [-0.7, -0.8]
    call = engine.generate_calls[0]
    assert call["sampling_params"].max_tokens == 2
    assert call["sampling_params"].temperature == 0.1
    assert call["sampling_params"].logprobs is not None


@pytest.mark.asyncio
async def test_generate_uses_default_response_budget_when_request_omits_max_tokens(monkeypatch):
    final = make_request_output([make_completion_output([4, 5], finish_reason="length")])
    server, engine = make_vllm_http_server(monkeypatch, [final], prompt_length=8, response_length=6, max_model_len=20)

    await server.generate(prompt_ids=[1, 2, 3], sampling_params={"temperature": 0.2}, request_id="default-budget")

    assert engine.generate_calls[0]["sampling_params"].max_tokens == 6


@pytest.mark.asyncio
async def test_generate_caps_default_response_budget_by_configured_total_length(monkeypatch):
    final = make_request_output([make_completion_output([4], finish_reason="length")])
    server, engine = make_vllm_http_server(monkeypatch, [final], prompt_length=8, response_length=6, max_model_len=20)

    await server.generate(prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sampling_params={}, request_id="short-budget")

    assert engine.generate_calls[0]["sampling_params"].max_tokens == 4


@pytest.mark.asyncio
async def test_generate_without_logprobs_does_not_request_or_return_token_logprobs(monkeypatch):
    final = make_request_output([make_completion_output([7], logprobs=[-0.7], finish_reason="stop")])
    server, engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[1],
        sampling_params={"max_tokens": 1, "logprobs": False},
        request_id="no-logprobs",
    )

    assert output.log_probs is None
    assert engine.generate_calls[0]["sampling_params"].logprobs is None


@pytest.mark.asyncio
async def test_generate_normalizes_completion_metadata_and_global_step(monkeypatch):
    final = make_request_output(
        [make_completion_output([7, 8], logprobs=[-0.7, -0.8], finish_reason="length", num_preempted=2)]
    )
    server, _engine = make_vllm_http_server(monkeypatch, [final])
    await _set_server_global_steps(server, 11)

    output = await server.generate(
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 2, "temperature": 0.1, "logprobs": True},
        request_id="r1",
    )

    assert output.stop_reason == "completed"
    assert output.num_preempted == 2
    assert output.extra_fields["global_steps"] == 11


@pytest.mark.asyncio
async def test_generate_maps_backend_abort_finish_reason_to_aborted(monkeypatch):
    final = make_request_output([make_completion_output([7], finish_reason="abort")])
    server, _engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="aborted")

    assert output.token_ids == [7]
    assert output.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_generate_accepts_max_new_tokens_alias(monkeypatch):
    final = make_request_output([make_completion_output([9], finish_reason="stop")])
    server, engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[1, 2],
        sampling_params={"max_new_tokens": 4},
        request_id="r2",
    )

    assert output.token_ids == [9]
    assert engine.generate_calls[0]["sampling_params"].max_tokens == 4


@pytest.mark.asyncio
async def test_generate_clamps_response_to_remaining_context(monkeypatch):
    final = make_request_output([make_completion_output([5], finish_reason="length")])
    server, engine = make_vllm_http_server(monkeypatch, [final], max_model_len=5, response_length=10, prompt_length=4)

    await server.generate(prompt_ids=[1, 2, 3, 4], sampling_params={"max_tokens": 99}, request_id="r3")

    assert engine.generate_calls[0]["sampling_params"].max_tokens == 1


@pytest.mark.asyncio
async def test_generate_raises_when_prompt_leaves_no_context(monkeypatch):
    server, _engine = make_vllm_http_server(monkeypatch, [], max_model_len=3)

    with pytest.raises(ValueError):
        await server.generate(prompt_ids=[1, 2, 3], sampling_params={}, request_id="full")


@pytest.mark.asyncio
async def test_generate_forwards_multimodal_payload_and_processor_kwargs(monkeypatch):
    final = make_request_output([make_completion_output([1], finish_reason="stop")])
    server, engine = make_vllm_http_server(monkeypatch, [final])

    await server.generate(
        prompt_ids=[3],
        sampling_params={"max_tokens": 1},
        request_id="mm",
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"sampling_rate": 16000},
    )

    call = engine.generate_calls[0]
    multimodal_data = _multimodal_data_from_call(call)
    assert multimodal_data == {"image": ["image"], "video": ["video"], "audio": ["audio"]}
    assert _mm_processor_kwargs_from_call(call) == {"sampling_rate": 16000}


@pytest.mark.asyncio
async def test_generate_deduplicates_consecutive_qwen_visual_tokens(monkeypatch):
    final = make_request_output([make_completion_output([1], finish_reason="stop")])
    server, engine = make_vllm_http_server(monkeypatch, [final])
    server.model_config = SimpleNamespace(
        lora_rank=0,
        lora={"rank": 0, "merge": False},
        processor=SimpleNamespace(
            image_processor=FakeQwen2VLImageProcessor(),
            image_token_id=100,
            video_token_id=101,
        ),
    )

    await server.generate(
        prompt_ids=[1, 100, 100, 100, 2, 101, 101, 3],
        sampling_params={"max_tokens": 1},
        request_id="qwen-vl",
        image_data=["image"],
        video_data=["video"],
    )

    assert _prompt_token_ids_from_call(engine.generate_calls[0]) == [1, 100, 2, 101, 3]


def test_generation_server_reports_bound_addresses(monkeypatch):
    server, _engine = make_vllm_http_server(monkeypatch, [])
    server._master_address = "10.0.0.1"
    server._master_port = 1234
    server._dp_rpc_port = 5678
    server._server_address = "127.0.0.1"
    server._server_port = 9012

    assert server.get_master_address() == ("10.0.0.1", 1234, 5678)
    assert server.get_server_address() == ("127.0.0.1", 9012)


@pytest.mark.asyncio
async def test_generation_server_collective_rpc_forwards_to_backend_engine(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.collective_rpc("reload_weights", timeout=3.0, args=("layer",), kwargs={"strict": True})

    collective_calls = [kwargs for name, kwargs in engine.lifecycle_calls if name == "collective_rpc"]
    assert collective_calls
    assert collective_calls[-1]["method"] == "reload_weights"
    assert collective_calls[-1].get("args") in {("layer",), None}
    assert collective_calls[-1].get("kwargs", {"strict": True}) == {"strict": True}


@pytest.mark.asyncio
async def test_generate_forwards_request_priority(monkeypatch):
    final = make_request_output([make_completion_output([1], finish_reason="stop")])
    server, engine = make_vllm_http_server(monkeypatch, [final])

    await server.generate(
        prompt_ids=[3],
        sampling_params={"max_tokens": 1},
        request_id="priority",
        priority=3,
    )

    assert engine.generate_calls[0]["priority"] == 3


@pytest.mark.asyncio
async def test_generate_maps_empty_backend_output_to_aborted(monkeypatch):
    server, _engine = make_vllm_http_server(monkeypatch, [make_request_output([])])

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="abort")

    assert output.token_ids == []
    assert output.stop_reason == "aborted"


@pytest.mark.asyncio
async def test_generate_exposes_prompt_logprobs_when_requested(monkeypatch):
    prompt_logprobs = [
        None,
        {"42": FakeTokenLogprob(-4.2, rank=1), "41": FakeTokenLogprob(-4.1, rank=2)},
        {"43": FakeTokenLogprob(-4.3, rank=1), "44": FakeTokenLogprob(-4.4, rank=2)},
    ]
    final = make_request_output(
        [make_completion_output([8], logprobs=[-0.8], finish_reason="stop")],
        prompt_logprobs=prompt_logprobs,
    )
    server, _engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[10, 11, 12],
        sampling_params={"prompt_logprobs": 2, "logprobs": True},
        request_id="plp",
    )

    assert output.extra_fields["prompt_ids"] == [[42, 41], [43, 44], [0, 0]]
    assert output.extra_fields["prompt_logprobs"] == [[-4.2, -4.1], [-4.3, -4.4], [0.0, 0.0]]


@pytest.mark.asyncio
async def test_generate_exposes_sampled_prompt_logprobs_when_zero_requested(monkeypatch):
    prompt_logprobs = [
        None,
        {"51": FakeTokenLogprob(-5.1, rank=1)},
        {"52": FakeTokenLogprob(-5.2, rank=1)},
    ]
    final = make_request_output(
        [make_completion_output([9], logprobs=[-0.9], finish_reason="stop")],
        prompt_logprobs=prompt_logprobs,
    )
    server, _engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[10, 11, 12],
        sampling_params={"prompt_logprobs": 0, "logprobs": True},
        request_id="plp-zero",
    )

    assert output.extra_fields["prompt_ids"] == [[51], [52], [0]]
    assert output.extra_fields["prompt_logprobs"] == [[-5.1], [-5.2], [0.0]]


@pytest.mark.asyncio
async def test_generate_orders_prompt_logprobs_by_backend_rank(monkeypatch):
    prompt_logprobs = [
        None,
        {
            "61": FakeTokenLogprob(-6.1, rank=2),
            "62": FakeTokenLogprob(-6.2, rank=1),
            "63": FakeTokenLogprob(-6.3, rank=3),
        },
    ]
    final = make_request_output(
        [make_completion_output([7], logprobs=[-0.7], finish_reason="stop")],
        prompt_logprobs=prompt_logprobs,
    )
    server, _engine = make_vllm_http_server(monkeypatch, [final])

    output = await server.generate(
        prompt_ids=[10, 11],
        sampling_params={"prompt_logprobs": 2, "logprobs": True},
        request_id="plp-rank-order",
    )

    assert output.extra_fields["prompt_ids"] == [[62, 61], [0, 0]]
    assert output.extra_fields["prompt_logprobs"] == [[-6.2, -6.1], [0.0, 0.0]]


@pytest.mark.asyncio
async def test_generate_returns_routed_experts_only_when_enabled(monkeypatch):
    routed = torch.ones(3, 2, 1, dtype=torch.long)
    final = make_request_output([make_completion_output([8], finish_reason="stop", routed_experts=routed)])
    server, _engine = make_vllm_http_server(monkeypatch, [final], enable_routing_replay=True)

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="router")

    assert torch.equal(output.routed_experts, routed)


@pytest.mark.asyncio
async def test_generate_does_not_expose_routed_experts_when_replay_disabled(monkeypatch):
    routed = torch.ones(3, 2, 1, dtype=torch.long)
    final = make_request_output([make_completion_output([8], finish_reason="stop", routed_experts=routed)])
    server, _engine = make_vllm_http_server(monkeypatch, [final], enable_routing_replay=False)

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="router-off")

    assert output.routed_experts is None


@pytest.mark.asyncio
async def test_generate_uses_lora_request_only_after_adapter_is_loaded(monkeypatch):
    final = make_request_output([make_completion_output([2], finish_reason="stop")])
    server, engine = make_vllm_http_server(
        monkeypatch,
        [final],
        lora_rank=8,
        loaded_loras=AnyLoadedLoraSet(),
    )

    await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="lora")

    assert engine.generate_calls[0]["lora_request"] is not None


@pytest.mark.asyncio
async def test_generate_does_not_send_lora_request_before_adapter_is_loaded(monkeypatch):
    final = make_request_output([make_completion_output([2], finish_reason="stop")])
    server, engine = make_vllm_http_server(
        monkeypatch,
        [final],
        lora_rank=8,
        loaded_loras=set(),
    )

    await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="lora-missing")

    assert engine.generate_calls[0]["lora_request"] is None


@pytest.mark.asyncio
async def test_generate_requires_mtp_stats_when_mtp_rollout_enabled(monkeypatch):
    mtp = {"enable": True, "enable_rollout": True}
    final = make_request_output([make_completion_output([1], finish_reason="stop")], metrics=None)
    server, _engine = make_vllm_http_server(monkeypatch, [final], mtp=mtp)

    with pytest.raises(RuntimeError, match="request_spec_decode_stats"):
        await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="mtp")


@pytest.mark.asyncio
async def test_generate_exposes_mtp_stats_when_backend_reports_them(monkeypatch):
    stats = SimpleNamespace(num_draft_tokens=5, num_accepted_tokens=3, num_verify_steps=2)
    metrics = SimpleNamespace(request_spec_decode_stats=stats)
    mtp = {"enable": True, "enable_rollout": True}
    final = make_request_output([make_completion_output([1], finish_reason="stop")], metrics=metrics)
    server, _engine = make_vllm_http_server(monkeypatch, [final], mtp=mtp)

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="mtp-ok")

    assert output.extra_fields["spec_num_draft_tokens"] == 5
    assert output.extra_fields["spec_num_accepted_tokens"] == 3
    assert output.extra_fields["spec_num_verify_steps"] == 2


@pytest.mark.asyncio
async def test_generation_server_wake_up_resets_cache(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.wake_up()

    calls = [name for name, _kwargs in engine.lifecycle_calls]
    assert "wake_up" in calls
    assert "reset_prefix_cache" in calls


@pytest.mark.asyncio
async def test_generation_server_wake_up_forwards_explicit_tags(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.wake_up(tags=["weights"])

    assert ("wake_up", {"tags": ["weights"]}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_colocated_wake_up_uses_default_tags_and_resets_cache(monkeypatch):
    from verl.workers.rollout.replica import RolloutMode

    server, engine = make_vllm_http_server(monkeypatch, [])
    server.rollout_mode = RolloutMode.COLOCATED

    await server.wake_up()

    assert any(name == "wake_up" for name, _kwargs in engine.lifecycle_calls)
    assert any(name == "reset_prefix_cache" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_standalone_wake_up_does_not_touch_engine(monkeypatch):
    from verl.workers.rollout.replica import RolloutMode

    server, engine = make_vllm_http_server(monkeypatch, [])
    server.rollout_mode = RolloutMode.STANDALONE

    await server.wake_up()

    assert engine.lifecycle_calls == []


@pytest.mark.asyncio
async def test_generation_server_sleep_releases_engine_memory(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.sleep()

    assert any(name == "sleep" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_lora_adapter_sleep_keeps_base_weights_resident(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [], lora_rank=8)

    await server.sleep()

    assert ("sleep", {"level": 1}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_colocated_sleep_releases_cache_only(monkeypatch):
    from verl.workers.rollout.replica import RolloutMode

    server, engine = make_vllm_http_server(monkeypatch, [])
    server.rollout_mode = RolloutMode.COLOCATED

    await server.sleep()

    assert ("sleep", {"level": 1}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_sleep_skips_when_cache_freeing_disabled(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    server.config.free_cache_engine = False

    await server.sleep()

    assert all(name != "sleep" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_non_master_node_does_not_manage_engine_memory(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    server.node_rank = 1

    await server.wake_up()
    await server.sleep()
    await server.clear_kv_cache()

    assert engine.lifecycle_calls == []


@pytest.mark.asyncio
async def test_generation_server_clear_kv_cache_resets_prefix_cache(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.clear_kv_cache()

    assert any(name == "reset_prefix_cache" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_wait_for_requests_to_drain_uses_engine(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.wait_for_requests_to_drain()

    assert any(name == "wait_for_requests_to_drain" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_abort_all_requests_reports_empty_state(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    engine.output_processor.request_states = {}

    result = await server.abort_all_requests()

    assert result["aborted_count"] == 0
    assert result["request_ids"] == []


@pytest.mark.asyncio
async def test_generation_server_abort_all_requests_pauses_and_clears_active_requests(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    engine.output_processor.request_states = {"r1": object(), "r2": object()}

    result = await server.abort_all_requests(reset_prefix_cache=True)

    assert result["aborted_count"] == 2
    assert result["request_ids"] == ["r1", "r2"]
    assert ("pause_generation", {"wait_for_inflight_requests": False, "clear_cache": True}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_abort_all_requests_can_leave_cache_intact(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    engine.output_processor.request_states = {"r1": object()}

    result = await server.abort_all_requests(reset_prefix_cache=False)

    assert result["aborted_count"] == 1
    assert result["request_ids"] == ["r1"]
    assert ("pause_generation", {"wait_for_inflight_requests": False, "clear_cache": False}) in engine.lifecycle_calls
    assert all(name != "reset_prefix_cache" for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_resume_generation_allows_future_requests(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    await server.resume_generation()

    assert ("resume_generation", {}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_profile_controls_forward_when_profiler_is_active(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    class ActiveProfiler:
        def check_enable(self):
            return True

        def check_this_rank(self):
            return True

        def is_discrete_mode(self):
            return True

    server.profiler_controller = ActiveProfiler()

    await server.start_profile(trace_id="trace-1")
    await server.stop_profile()

    assert ("start_profile", {"trace_id": "trace-1"}) in engine.lifecycle_calls
    assert ("stop_profile", {}) in engine.lifecycle_calls


@pytest.mark.asyncio
async def test_generation_server_profile_controls_do_not_forward_when_profiler_is_inactive(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])

    class InactiveProfiler:
        def check_enable(self):
            return False

        def check_this_rank(self):
            return True

        def is_discrete_mode(self):
            return True

    server.profiler_controller = InactiveProfiler()

    await server.start_profile(trace_id="trace-2")
    await server.stop_profile()

    assert all(name not in {"start_profile", "stop_profile"} for name, _kwargs in engine.lifecycle_calls)


@pytest.mark.asyncio
async def test_generation_server_abort_missing_request_is_reported(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    engine.output_processor.request_states = {}

    result = await server.abort_request("missing-request")

    assert result["aborted"] is False
    assert result.get("request_id", "missing-request") == "missing-request"


@pytest.mark.asyncio
async def test_generation_server_abort_existing_request_reports_and_resets_cache(monkeypatch):
    server, engine = make_vllm_http_server(monkeypatch, [])
    queued = []

    request_state = SimpleNamespace(
        queue=SimpleNamespace(put=lambda output: queued.append(output)),
        make_request_output=lambda *args, **kwargs: {"aborted": True, "kwargs": kwargs},
    )
    engine.output_processor.request_states = {"active-request": request_state}

    result = await server.abort_request("active-request")

    assert result["aborted"] is True
    assert result["request_id"] == "active-request"
