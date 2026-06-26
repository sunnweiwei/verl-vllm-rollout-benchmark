from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vllm_eval_helpers import make_agent_loop_worker


@pytest.mark.asyncio
async def test_agent_loop_postprocess_builds_training_batch_with_logprobs():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker = make_agent_loop_worker(prompt_length=5, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[3, 4],
        response_ids=[5, 6, 7],
        response_mask=[1, 1, 1],
        response_logprobs=[-0.5, -0.6, -0.7],
        reward_score=1.25,
        num_turns=2,
        metrics=AgentLoopMetrics(num_preempted=2),
        extra_fields={"global_steps": 9, "turn_scores": [], "tool_rewards": []},
    )

    internal = await worker._agent_loop_postprocess(output, validate=False, raw_prompt=[{"role": "user"}])
    batch = worker._postprocess([internal])

    assert batch.batch["prompts"].shape == (1, 5)
    assert batch.batch["responses"].shape == (1, 4)
    assert batch.batch["input_ids"].shape == (1, 9)
    assert batch.batch["response_mask"].tolist() == [[1, 1, 1, 0]]
    torch.testing.assert_close(
        batch.batch["rollout_log_probs"],
        torch.tensor([[-0.5, -0.6, -0.7, 0.0]], dtype=torch.float32),
    )
    assert batch.batch["rm_scores"].shape == (1, 4)
    assert batch.non_tensor_batch["global_steps"][0] == 9
    assert batch.non_tensor_batch["__num_turns__"].tolist() == [2]
    assert batch.meta_info["metrics"][0]["num_preempted"] == 2


@pytest.mark.asyncio
async def test_agent_loop_postprocess_masks_tool_observation_tokens():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker = make_agent_loop_worker(prompt_length=4, response_length=6)
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[10, 11, 12, 13, 14],
        response_mask=[1, 1, 0, 0, 1],
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [0.1], "tool_rewards": [0.0]},
    )

    internal = await worker._agent_loop_postprocess(output, validate=False, raw_prompt=[{"role": "user"}])
    batch = worker._postprocess([internal])

    assert batch.batch["response_mask"].tolist() == [[1, 1, 0, 0, 1, 0]]
    assert batch.non_tensor_batch["turn_scores"][0] == [0.1]
    assert batch.non_tensor_batch["tool_rewards"][0] == [0.0]


@pytest.mark.asyncio
async def test_agent_loop_postprocess_aligns_routed_experts_with_left_padded_prompt():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker = make_agent_loop_worker(prompt_length=5, response_length=3)
    routed = np.arange(5 * 2 * 1).reshape(5, 2, 1)
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 1, 1],
        routed_experts=routed,
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": []},
    )

    internal = await worker._agent_loop_postprocess(output, validate=False, raw_prompt=[{"role": "user"}])
    batch = worker._postprocess([internal])

    padded = batch.batch["routed_experts"]
    assert padded.shape == (1, 8, 2, 1)
    assert padded[0, :3].sum().item() == 0
    assert torch.equal(padded[0, 3:8], torch.tensor(routed))


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_numpy_routing():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker = make_agent_loop_worker(prompt_length=3, response_length=2)
    routed = np.arange(4).reshape(2, 2, 1)
    routed.setflags(write=False)
    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        routed_experts=routed,
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": []},
    )

    internal = await worker._agent_loop_postprocess(output, validate=False, raw_prompt=[{"role": "user"}])

    assert internal.routed_experts.shape == (1, 5, 2, 1)


@pytest.mark.asyncio
async def test_agent_loop_postprocess_keeps_extra_field_schema_stable_across_samples():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker = make_agent_loop_worker(prompt_length=3, response_length=2)
    first = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        metrics=AgentLoopMetrics(),
        extra_fields={"global_steps": 1, "turn_scores": [], "tool_rewards": []},
    )
    second = AgentLoopOutput(
        prompt_ids=[3],
        response_ids=[4],
        response_mask=[1],
        metrics=AgentLoopMetrics(),
        extra_fields={"min_global_steps": 1, "max_global_steps": 2, "turn_scores": [], "tool_rewards": []},
    )

    internals = [
        await worker._agent_loop_postprocess(first, validate=False, raw_prompt=[{"role": "user"}]),
        await worker._agent_loop_postprocess(second, validate=False, raw_prompt=[{"role": "user"}]),
    ]
    batch = worker._postprocess(internals)

    for key in ("global_steps", "min_global_steps", "max_global_steps", "turn_scores", "tool_rewards", "extras"):
        assert key in batch.non_tensor_batch
    assert batch.non_tensor_batch["global_steps"].tolist() == [1, None]
    assert batch.non_tensor_batch["max_global_steps"].tolist() == [None, 2]


@pytest.mark.asyncio
async def test_single_turn_agent_forwards_sampling_and_multimodal_payload(monkeypatch):
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.workers.rollout.replica import TokenOutput

    calls = []

    class FakeServerManager:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return TokenOutput(
                token_ids=[21, 22],
                log_probs=[-2.1, -2.2],
                stop_reason="completed",
                num_preempted=3,
                extra_fields={"global_steps": 5},
            )

    class FakeDataset:
        @staticmethod
        async def process_multi_modal_info(messages, image_patch_size=14, config=None):
            return ["image"], ["video"], ["audio"]

    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=22050),
        apply_chat_template=lambda *args, **kwargs: [0],
    )
    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
    SingleTurnAgentLoop.__init__(
        loop,
        trainer_config=DictConfigWrap(
            OmegaConf.create(
                {
                "actor_rollout_ref": {
                    "rollout": {"prompt_length": 8, "response_length": 4},
                }
            }
            )
        ),
        server_manager=FakeServerManager(),
        tokenizer=SimpleNamespace(),
        processor=processor,
        dataset_cls=FakeDataset,
        data_config=DictConfigWrap(OmegaConf.create({"mm_processor_kwargs": {}})),
    )

    async def fake_apply_chat_template(*args, **kwargs):
        return [1, 2, 3]

    monkeypatch.setattr(loop, "apply_chat_template", fake_apply_chat_template)

    output = await loop.run({"max_tokens": 2, "logprobs": True}, raw_prompt=[{"role": "user", "content": "x"}])

    assert output.response_ids == [21, 22]
    assert output.response_logprobs == [-2.1, -2.2]
    assert output.metrics.num_preempted == 3
    assert output.extra_fields["global_steps"] == 5
    assert calls[0]["prompt_ids"] == [1, 2, 3]
    assert calls[0]["image_data"] == ["image"]
    assert calls[0]["video_data"] == ["video"]
    assert calls[0]["audio_data"] == ["audio"]
    assert calls[0]["mm_processor_kwargs"] == {"sampling_rate": 22050}


def test_agent_loop_sampling_params_switch_to_validation_values():
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.rollout_config = SimpleNamespace(
        temperature=0.9,
        top_p=0.8,
        top_k=20,
        calculate_log_probs=True,
        val_kwargs=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=-1),
    )

    params = {
        "temperature": worker.rollout_config.temperature,
        "top_p": worker.rollout_config.top_p,
        "top_k": worker.rollout_config.top_k,
        "logprobs": worker.rollout_config.calculate_log_probs,
    }
    params["temperature"] = worker.rollout_config.val_kwargs.temperature
    params["top_p"] = worker.rollout_config.val_kwargs.top_p
    params["top_k"] = worker.rollout_config.val_kwargs.top_k

    assert params == {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "logprobs": True}
