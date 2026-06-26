from types import SimpleNamespace

import pytest
import torch

from vllm_eval_helpers import (
    FakeTokenLogprob,
    make_completion_output,
    make_request_output,
    make_vllm_http_server,
)


@pytest.mark.asyncio
async def test_generate_returns_tokens_logprobs_stop_reason_and_global_step(monkeypatch):
    final = make_request_output(
        [make_completion_output([7, 8], logprobs=[-0.7, -0.8], finish_reason="length", num_preempted=2)]
    )
    server, engine = make_vllm_http_server(monkeypatch, [final])
    await server.set_global_steps(11)

    output = await server.generate(
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 2, "temperature": 0.1, "logprobs": True},
        request_id="r1",
    )

    assert output.token_ids == [7, 8]
    assert output.log_probs == [-0.7, -0.8]
    assert output.stop_reason == "completed"
    assert output.num_preempted == 2
    assert output.extra_fields["global_steps"] == 11
    call = engine.generate_calls[0]
    assert call["sampling_params"].max_tokens == 2
    assert call["sampling_params"].temperature == 0.1
    assert call["sampling_params"].logprobs == 0


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
    assert output.stop_reason == "completed"
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

    with pytest.raises(ValueError, match="leaves no room"):
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
        priority=3,
    )

    call = engine.generate_calls[0]
    prompt = call["prompt"]
    assert call["priority"] == 3
    multimodal_data = prompt["multi_modal_data"] if isinstance(prompt, dict) else prompt.multi_modal_data
    assert multimodal_data == {"image": ["image"], "video": ["video"], "audio": ["audio"]}


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
async def test_generate_returns_routed_experts_only_when_enabled(monkeypatch):
    routed = torch.ones(3, 2, 1, dtype=torch.long)
    final = make_request_output([make_completion_output([8], finish_reason="stop", routed_experts=routed)])
    server, _engine = make_vllm_http_server(monkeypatch, [final], enable_routing_replay=True)

    output = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="router")

    assert torch.equal(output.routed_experts, routed)


@pytest.mark.asyncio
async def test_generate_uses_lora_request_only_after_adapter_is_loaded(monkeypatch):
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID

    final = make_request_output([make_completion_output([2], finish_reason="stop")])
    server, engine = make_vllm_http_server(
        monkeypatch,
        [final],
        lora_rank=8,
        loaded_loras={VLLM_LORA_INT_ID},
    )

    await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 1}, request_id="lora")

    assert engine.generate_calls[0]["lora_request"] is not None


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
