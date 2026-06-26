import copy
import sys
import types
from types import SimpleNamespace
from typing import Any

import ray
import torch
from omegaconf import OmegaConf


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
                "stop_reason": "completed",
                "extra_fields": {"global_steps": None},
            }
        return TokenOutput(**payload)

    def get_calls(self):
        return copy.deepcopy(self.calls)

    def get_events(self):
        return copy.deepcopy(self.events)

    async def wake_up(self, *args, **kwargs):
        self.events.append(("wake_up", copy.deepcopy(kwargs)))

    async def sleep(self, *args, **kwargs):
        self.events.append(("sleep", copy.deepcopy(kwargs)))

    async def wait_for_requests_to_drain(self, *args, **kwargs):
        self.events.append(("wait_for_requests_to_drain", copy.deepcopy(kwargs)))

    async def abort_all_requests(self, *args, **kwargs):
        self.events.append(("abort_all_requests", copy.deepcopy(kwargs)))
        return {"aborted_count": 1, "request_ids": [f"{self.label}-request"]}

    async def abort_request(self, request_id: str, *args, **kwargs):
        self.events.append(("abort_request", {"request_id": request_id, **copy.deepcopy(kwargs)}))
        return {"aborted": self.label == "a", "request_id": request_id}

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


def make_config(
    *,
    rollout_name: str = "vllm",
    partial_rollout: bool = False,
    n_gpus_per_node: int = 1,
    nnodes: int = 1,
    tp: int = 1,
    dp: int = 1,
    pp: int = 1,
    calculate_log_probs: bool = False,
):
    rollout = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")
    OmegaConf.update(rollout, "name", rollout_name, force_add=True)
    OmegaConf.update(rollout, "mode", "async", force_add=True)
    OmegaConf.update(rollout, "nnodes", nnodes, force_add=True)
    OmegaConf.update(rollout, "n_gpus_per_node", n_gpus_per_node, force_add=True)
    OmegaConf.update(rollout, "tensor_model_parallel_size", tp, force_add=True)
    OmegaConf.update(rollout, "data_parallel_size", dp, force_add=True)
    OmegaConf.update(rollout, "pipeline_model_parallel_size", pp, force_add=True)
    OmegaConf.update(rollout, "prompt_length", 8, force_add=True)
    OmegaConf.update(rollout, "response_length", 6, force_add=True)
    OmegaConf.update(rollout, "max_model_len", 14, force_add=True)
    OmegaConf.update(rollout, "max_num_seqs", 8, force_add=True)
    OmegaConf.update(rollout, "calculate_log_probs", calculate_log_probs, force_add=True)
    OmegaConf.update(rollout, "engine_kwargs.vllm", {"sentinel": "preserved"}, force_add=True)
    OmegaConf.update(rollout, "engine_kwargs.sglang", {"sentinel": "sglang"}, force_add=True)
    OmegaConf.update(rollout, "engine_kwargs.trtllm", {"sentinel": "trtllm"}, force_add=True)

    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": rollout,
                "model": {
                    "_target_": "verl.workers.config.HFModelConfig",
                    "path": "dummy-model",
                    "lora_rank": 0,
                    "lora": {"rank": 0, "merge": False},
                },
            },
            "reward": {
                "reward_model": {
                    "rollout": copy.deepcopy(rollout),
                    "enable": False,
                    "enable_resource_pool": False,
                }
            },
            "trainer": {"nnodes": 1, "n_gpus_per_node": n_gpus_per_node},
            "async_training": {"partial_rollout": partial_rollout, "use_trainer_do_validate": False},
            "data": {"mm_processor_kwargs": {}},
            "distillation": {"enable": False},
        }
    )


def patch_vllm_standalone_launch(monkeypatch, outputs: list[dict[str, Any]], *, labels: list[str] | None = None):
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    created = []
    labels = labels or ["server"]

    async def fake_init_standalone(self):
        assert self.config.name == "vllm"
        assert self.config.engine_kwargs["vllm"]["sentinel"] == "preserved"
        label = labels[len(created) % len(labels)]
        server = ScriptedRolloutServer.remote(outputs, label=label)
        self.servers = [server]
        self._server_handle = server
        self._server_address = f"fake-vllm://replica-{self.replica_rank}"
        self.workers = []
        created.append(self)

    monkeypatch.setattr(replica_cls, "init_standalone", fake_init_standalone, raising=True)
    return created


def install_fully_async_import_stubs(monkeypatch):
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


class FakeTokenLogprob:
    def __init__(self, logprob: float, rank: int = 1):
        self.logprob = logprob
        self.rank = rank


class FakeAsyncEngine:
    def __init__(self, final_outputs: list[Any], *, loras: set[int] | None = None):
        self.final_outputs = list(final_outputs)
        self.loras = set(loras or set())
        self.generate_calls = []
        self.lifecycle_calls = []
        self.output_processor = SimpleNamespace(request_states={})

    async def list_loras(self):
        return set(self.loras)

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        final_output = self.final_outputs.pop(0)

        async def gen():
            yield final_output

        return gen()

    async def wake_up(self, **kwargs):
        self.lifecycle_calls.append(("wake_up", kwargs))

    async def sleep(self, **kwargs):
        self.lifecycle_calls.append(("sleep", kwargs))

    async def reset_prefix_cache(self, **kwargs):
        self.lifecycle_calls.append(("reset_prefix_cache", kwargs))

    async def wait_for_requests_to_drain(self):
        self.lifecycle_calls.append(("wait_for_requests_to_drain", {}))

    async def pause_generation(self, **kwargs):
        self.lifecycle_calls.append(("pause_generation", kwargs))

    async def resume_generation(self):
        self.lifecycle_calls.append(("resume_generation", {}))


def make_completion_output(
    token_ids: list[int],
    *,
    logprobs: list[float] | None = None,
    finish_reason: str = "length",
    routed_experts: Any = None,
    num_preempted: int | None = None,
):
    if logprobs is None:
        logprob_payload = None
    else:
        logprob_payload = [
            {token_id: FakeTokenLogprob(logprob)}
            for token_id, logprob in zip(token_ids, logprobs, strict=True)
        ]
    return SimpleNamespace(
        token_ids=token_ids,
        logprobs=logprob_payload,
        finish_reason=finish_reason,
        routed_experts=routed_experts,
        num_preempted=num_preempted,
    )


def make_request_output(
    outputs: list[Any],
    *,
    prompt_logprobs: list[dict[str, FakeTokenLogprob] | None] | None = None,
    metrics: Any = None,
):
    return SimpleNamespace(outputs=outputs, prompt_logprobs=prompt_logprobs or [None], metrics=metrics)


def make_vllm_http_server(
    monkeypatch,
    final_outputs: list[Any],
    *,
    max_model_len: int = 14,
    response_length: int = 6,
    prompt_length: int = 8,
    enable_routing_replay: bool = False,
    mtp: Any = None,
    lora_rank: int = 0,
    lora_merge: bool = False,
    loaded_loras: set[int] | None = None,
):
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout import vllm_async_server as module

    monkeypatch.setattr(module, "qwen2_5_vl_dedup_image_tokens", lambda ids, processor: ids)
    server = module.vLLMHttpServer.__new__(module.vLLMHttpServer)
    server.config = OmegaConf.create(
        {
            "max_model_len": max_model_len,
            "response_length": response_length,
            "prompt_length": prompt_length,
            "repetition_penalty": 1.05,
            "enable_rollout_routing_replay": enable_routing_replay,
            "free_cache_engine": True,
            "mtp": mtp,
        }
    )
    server.model_config = OmegaConf.create(
        {
            "lora_rank": lora_rank,
            "lora": {"rank": lora_rank, "merge": lora_merge},
            "processor": None,
        }
    )
    server.global_steps = None
    server.node_rank = 0
    server.rollout_mode = RolloutMode.HYBRID
    server.engine = FakeAsyncEngine(final_outputs, loras=loaded_loras)
    return server, server.engine


def make_agent_loop_worker(*, prompt_length: int = 5, response_length: int = 6, calculate_log_probs: bool = True):
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

        def pad(
            self,
            encoded,
            *,
            padding,
            max_length,
            return_tensors,
            return_attention_mask,
        ):
            ids = list(encoded["input_ids"])
            pad_size = max_length - len(ids)
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_size + ids
                mask = [0] * pad_size + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * pad_size
                mask = [1] * len(ids) + [0] * pad_size
            result = {"input_ids": torch.tensor([padded], dtype=torch.long)}
            if return_attention_mask:
                result["attention_mask"] = torch.tensor([mask], dtype=torch.long)
            return result

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.rollout_config = OmegaConf.create(
        {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "calculate_log_probs": calculate_log_probs,
        }
    )
    worker.tokenizer = FakeTokenizer()
    worker.processor = None
    worker.reward_loop_worker_handles = None
    worker.distillation_enabled = False

    async def no_score(outputs, kwargs):
        return None

    async def no_teacher(output, prompt_ids, response_ids, validate, sample_kwargs=None):
        return None

    worker._compute_score = no_score
    worker._compute_teacher_logprobs = no_teacher
    worker._compute_multi_modal_inputs = lambda output, input_ids: None
    worker._compute_position_ids = lambda input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs=None: (
        torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    )
    worker._get_mm_processor_kwargs = lambda audio_data=None: {}
    return worker


class FakeRolloutForWeights:
    def __init__(self):
        self.events = []
        self.sleep_level = 2

    async def resume(self, tags):
        self.events.append(("resume", list(tags)))

    async def update_weights(self, weights, **kwargs):
        self.events.append(("update_weights", weights, copy.deepcopy(kwargs)))


class FakeActorEngine:
    def __init__(self, *, peft_config=None, params_true="adapter", params_false="base", offload=False):
        self.peft_config = peft_config
        self.params_true = params_true
        self.params_false = params_false
        self.is_param_offload_enabled = offload
        self.calls = []
        self.to_calls = []

    def get_per_tensor_param(self, *args, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if kwargs.get("base_sync_done", True):
            return self.params_true, self.peft_config
        return self.params_false, self.peft_config

    def to(self, device, **kwargs):
        self.to_calls.append((device, copy.deepcopy(kwargs)))
