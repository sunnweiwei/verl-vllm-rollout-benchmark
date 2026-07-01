import pytest
from omegaconf import OmegaConf

from vllm_eval_helpers import FakeActorEngine, FakeRolloutForWeights


def _make_worker(*, peft_config=None, peft_merge=False, base_sync_done=False, backend="naive", free_cache=True):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "checkpoint_engine": {"backend": backend},
                "free_cache_engine": free_cache,
            }
        }
    )
    worker.rollout = FakeRolloutForWeights()
    worker.actor = type("Actor", (), {"engine": FakeActorEngine(peft_config=peft_config)})()
    worker.layered_summon = False
    worker.peft_merge = peft_merge
    worker.base_sync_done = base_sync_done
    return worker


def _patch_gpu_side_effects(monkeypatch):
    import verl.workers.engine_workers as engine_workers

    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", lambda *_args, **_kwargs: None)


@pytest.mark.asyncio
async def test_adapter_mode_first_update_sends_base_before_adapter(monkeypatch):
    _patch_gpu_side_effects(monkeypatch)

    peft_cfg = {"r": 8}
    worker = _make_worker(peft_config=peft_cfg, peft_merge=False, base_sync_done=False)

    from verl.workers.engine_workers import ActorRolloutRefWorker

    await ActorRolloutRefWorker.update_weights(worker, global_steps=10, mode="naive")

    assert worker.actor.engine.calls == [
        {"layered_summon": False, "base_sync_done": True},
        {"layered_summon": False, "base_sync_done": False},
    ]
    assert worker.rollout.events[0] == ("resume", ["weights"])
    assert worker.rollout.events[1] == (
        "update_weights",
        "base",
        {"peft_config": peft_cfg, "base_sync_done": False, "global_steps": 10},
    )
    assert worker.rollout.events[2] == (
        "update_weights",
        "adapter",
        {"peft_config": peft_cfg, "base_sync_done": True, "global_steps": 10},
    )
    assert worker.rollout.events[-1] == ("resume", ["kv_cache"])
    assert worker.base_sync_done is True


@pytest.mark.asyncio
async def test_adapter_mode_subsequent_update_sends_adapter_only(monkeypatch):
    _patch_gpu_side_effects(monkeypatch)

    peft_cfg = {"r": 8}
    worker = _make_worker(peft_config=peft_cfg, peft_merge=False, base_sync_done=True)

    from verl.workers.engine_workers import ActorRolloutRefWorker

    await ActorRolloutRefWorker.update_weights(worker, global_steps=11, mode="naive")

    update_events = [event for event in worker.rollout.events if event[0] == "update_weights"]
    assert update_events == [
        (
            "update_weights",
            "adapter",
            {"peft_config": peft_cfg, "base_sync_done": True, "global_steps": 11},
        )
    ]


@pytest.mark.asyncio
async def test_merged_lora_or_plain_model_uses_single_standard_update(monkeypatch):
    _patch_gpu_side_effects(monkeypatch)

    worker = _make_worker(peft_config=None, peft_merge=True, base_sync_done=False)

    from verl.workers.engine_workers import ActorRolloutRefWorker

    await ActorRolloutRefWorker.update_weights(worker, global_steps=12, mode="naive")

    update_events = [event for event in worker.rollout.events if event[0] == "update_weights"]
    assert update_events == [
        (
            "update_weights",
            "adapter",
            {"peft_config": None, "base_sync_done": True, "global_steps": 12},
        )
    ]


@pytest.mark.asyncio
async def test_free_cache_engine_false_skips_resume_calls(monkeypatch):
    _patch_gpu_side_effects(monkeypatch)

    worker = _make_worker(peft_config=None, free_cache=False)

    from verl.workers.engine_workers import ActorRolloutRefWorker

    await ActorRolloutRefWorker.update_weights(worker, global_steps=13, mode="naive")

    assert all(event[0] != "resume" for event in worker.rollout.events)
