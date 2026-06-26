import copy
import logging
import sys
import types
from typing import Any

import pytest
import ray
from omegaconf import OmegaConf


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level=logging.ERROR)
    yield
    ray.shutdown()


@ray.remote
class ScriptedRolloutServer:
    def __init__(self, outputs: list[dict[str, Any]] | None = None, label: str = "server"):
        self.outputs = list(outputs or [])
        self.label = label
        self.calls = []
        self.events = []

    async def generate(self, **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        self.calls.append(copy.deepcopy(kwargs))
        if self.outputs:
            payload = self.outputs.pop(0)
        else:
            payload = {
                "token_ids": [999],
                "log_probs": [-0.1],
                "stop_reason": "length",
                "extra_fields": {"global_steps": None},
            }
        return TokenOutput(**payload)

    def get_calls(self):
        return copy.deepcopy(self.calls)

    def get_events(self):
        return list(self.events)

    async def wake_up(self, *args, **kwargs):
        self.events.append(("wake_up", copy.deepcopy(kwargs)))

    async def sleep(self, *args, **kwargs):
        self.events.append(("sleep", copy.deepcopy(kwargs)))

    async def wait_for_requests_to_drain(self, *args, **kwargs):
        self.events.append(("wait_for_requests_to_drain", copy.deepcopy(kwargs)))

    async def abort_all_requests(self, *args, **kwargs):
        self.events.append(("abort_all_requests", copy.deepcopy(kwargs)))
        return {"aborted_count": 1, "request_ids": [f"{self.label}-request"]}

    async def resume_generation(self, *args, **kwargs):
        self.events.append(("resume_generation", copy.deepcopy(kwargs)))

    async def clear_kv_cache(self, *args, **kwargs):
        self.events.append(("clear_kv_cache", copy.deepcopy(kwargs)))

    async def release_kv_cache(self, *args, **kwargs):
        self.events.append(("release_kv_cache", copy.deepcopy(kwargs)))

    async def resume_kv_cache(self, *args, **kwargs):
        self.events.append(("resume_kv_cache", copy.deepcopy(kwargs)))

    async def start_profile(self, **kwargs):
        self.events.append(("start_profile", copy.deepcopy(kwargs)))

    async def stop_profile(self, *args, **kwargs):
        self.events.append(("stop_profile", copy.deepcopy(kwargs)))


def _make_config(*, partial_rollout: bool = False):
    rollout = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")
    OmegaConf.update(rollout, "name", "vllm", force_add=True)
    OmegaConf.update(rollout, "mode", "async", force_add=True)
    OmegaConf.update(rollout, "nnodes", 1, force_add=True)
    OmegaConf.update(rollout, "n_gpus_per_node", 1, force_add=True)
    OmegaConf.update(rollout, "tensor_model_parallel_size", 1, force_add=True)
    OmegaConf.update(rollout, "data_parallel_size", 1, force_add=True)
    OmegaConf.update(rollout, "pipeline_model_parallel_size", 1, force_add=True)
    OmegaConf.update(rollout, "prompt_length", 32, force_add=True)
    OmegaConf.update(rollout, "response_length", 16, force_add=True)
    OmegaConf.update(rollout, "max_num_seqs", 8, force_add=True)
    OmegaConf.update(
        rollout,
        "engine_kwargs.vllm",
        {"sentinel": "preserved", "max_num_batched_tokens": 64},
        force_add=True,
    )

    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": rollout,
                "model": {"_target_": "verl.workers.config.HFModelConfig", "path": "dummy-model"},
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 1,
            },
            "async_training": {
                "partial_rollout": partial_rollout,
            },
        }
    )


def _patch_vllm_standalone_launch(monkeypatch, outputs: list[dict[str, Any]]):
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    created = []

    async def fake_init_standalone(self):
        assert self.config.name == "vllm"
        assert self.config.engine_kwargs["vllm"]["sentinel"] == "preserved"
        server = ScriptedRolloutServer.remote(outputs)
        self.servers = [server]
        self._server_handle = server
        self._server_address = f"fake-vllm://replica-{self.replica_rank}"
        self.workers = []
        created.append(self)

    monkeypatch.setattr(replica_cls, "init_standalone", fake_init_standalone, raising=True)
    return created


def _install_fully_async_import_stubs(monkeypatch):
    """Keep this CPU test focused on rollout-client behavior, not trainer imports."""

    def stub_module(name: str):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    agent_loop = stub_module("verl.experimental.agent_loop.agent_loop")

    class AgentLoopManager:
        pass

    agent_loop.AgentLoopManager = AgentLoopManager

    detach_utils = stub_module("verl.experimental.fully_async_policy.detach_utils")

    class RolloutSample:
        pass

    detach_utils.RolloutSample = RolloutSample
    detach_utils.prepare_single_generation_data = lambda *args, **kwargs: None
    detach_utils.safe_create_task = lambda coro, *args, **kwargs: coro

    message_queue = stub_module("verl.experimental.fully_async_policy.message_queue")

    class MessageQueueClient:
        pass

    message_queue.MessageQueueClient = MessageQueueClient

    separation_trainer = stub_module("verl.experimental.separation.ray_trainer")

    class SeparateRayPPOTrainer:
        pass

    separation_trainer.SeparateRayPPOTrainer = SeparateRayPPOTrainer

    ppo_utils = stub_module("verl.trainer.ppo.utils")
    ppo_utils.need_reward_model = lambda *args, **kwargs: False

    checkpoint_manager = stub_module("verl.utils.checkpoint.checkpoint_manager")
    checkpoint_manager.find_latest_ckpt_path = lambda *args, **kwargs: None

    tracking = stub_module("verl.utils.tracking")

    class ValidationGenerationsLogger:
        pass

    tracking.ValidationGenerationsLogger = ValidationGenerationsLogger


@pytest.mark.asyncio
async def test_vllm_backend_runs_through_manager_and_client(monkeypatch):
    from verl.workers.rollout.llm_server import LLMServerManager

    outputs = [
        {
            "token_ids": [31, 32, 33],
            "log_probs": [-0.3, -0.2, -0.1],
            "stop_reason": "length",
            "extra_fields": {"global_steps": 7, "source": "fake-vllm"},
        }
    ]
    created = _patch_vllm_standalone_launch(monkeypatch, outputs)

    manager = await LLMServerManager.create(config=_make_config())
    assert manager.get_addresses() == ["fake-vllm://replica-0"]
    assert len(created) == 1

    result = await manager.get_client().generate(
        request_id="sticky-user-request",
        prompt_ids=[11, 12],
        sampling_params={"temperature": 0.0, "max_tokens": 3, "logprobs": True},
        image_data=["image-payload"],
        video_data=["video-payload"],
    )

    assert result.token_ids == [31, 32, 33]
    assert result.log_probs == [-0.3, -0.2, -0.1]
    assert result.stop_reason == "length"
    assert result.extra_fields["global_steps"] == 7

    calls = await manager.server_handles[0].get_calls.remote()
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt_ids"] == [11, 12]
    assert call["sampling_params"] == {"temperature": 0.0, "max_tokens": 3, "logprobs": True}
    assert call["image_data"] == ["image-payload"]
    assert call["video_data"] == ["video-payload"]
    assert call["request_id"] != "sticky-user-request"

    status = await manager.global_load_balancer.get_status.remote()
    assert status["total_inflight"] == 0
    assert status["active_servers"] == 1


@pytest.mark.asyncio
async def test_fully_async_client_resumes_after_aborted_generation(monkeypatch):
    _install_fully_async_import_stubs(monkeypatch)

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
            "stop_reason": "length",
            "num_preempted": 2,
            "extra_fields": {"global_steps": 4},
        },
    ]
    _patch_vllm_standalone_launch(monkeypatch, outputs)

    manager = await LLMServerManager.create(config=_make_config(partial_rollout=True))
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
    assert len(calls) == 2
    assert calls[0]["prompt_ids"] == [1, 2]
    assert calls[0]["sampling_params"]["max_tokens"] == 3
    assert calls[1]["prompt_ids"] == [1, 2, 101, 102]
    assert calls[1]["sampling_params"]["max_tokens"] == 1


@pytest.mark.asyncio
async def test_vllm_replica_lifecycle_forwards_to_rollout_servers():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(
        replica_rank=0,
        config=_make_config().actor_rollout_ref.rollout,
        model_config={"path": "dummy"},
    )

    server_a = ScriptedRolloutServer.remote([], label="a")
    server_b = ScriptedRolloutServer.remote([], label="b")
    replica.servers = [server_a, server_b]

    await replica.wake_up()
    await replica.sleep()
    abort_result = await replica.abort_all_requests()
    await replica.resume_generation()
    await replica.clear_kv_cache()
    await replica.release_kv_cache()
    await replica.resume_kv_cache()
    await replica.start_profile(trace_id="cpu-test")
    await replica.stop_profile()

    if abort_result is not None:
        assert abort_result["aborted_count"] == 2
        assert sorted(abort_result["request_ids"]) == ["a-request", "b-request"]

    for events in [await server_a.get_events.remote(), await server_b.get_events.remote()]:
        event_names = [name for name, _kwargs in events]
        for required in [
            "wake_up",
            "sleep",
            "abort_all_requests",
            "resume_generation",
            "clear_kv_cache",
            "release_kv_cache",
            "resume_kv_cache",
            "start_profile",
            "stop_profile",
        ]:
            assert required in event_names
        assert ("start_profile", {"trace_id": "cpu-test"}) in events


def test_rollout_config_exposes_vllm_engine_kwargs_without_dropping_existing_backends():
    cfg = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")

    assert "vllm" in cfg.engine_kwargs
    assert "sglang" in cfg.engine_kwargs
    assert "trtllm" in cfg.engine_kwargs

    rollout_cfg = _make_config().actor_rollout_ref.rollout
    assert rollout_cfg.name == "vllm"
    assert rollout_cfg.engine_kwargs["vllm"]["sentinel"] == "preserved"
