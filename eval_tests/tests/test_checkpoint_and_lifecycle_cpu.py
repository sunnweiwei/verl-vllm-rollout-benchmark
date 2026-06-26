import pytest
import ray

from vllm_eval_helpers import ScriptedRolloutServer, make_config


@pytest.mark.asyncio
async def test_vllm_replica_lifecycle_forwards_to_rollout_servers():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(
        replica_rank=0,
        config=make_config().actor_rollout_ref.rollout,
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


@pytest.mark.asyncio
async def test_vllm_replica_sleep_and_release_wait_for_requests_to_drain_first():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(replica_rank=0, config=make_config().actor_rollout_ref.rollout, model_config={"path": "x"})
    server = ScriptedRolloutServer.remote([], label="a")
    replica.servers = [server]

    await replica.sleep()
    await replica.release_kv_cache()

    events = [name for name, _ in await server.get_events.remote()]
    assert events.index("wait_for_requests_to_drain") < events.index("sleep")
    assert events.count("wait_for_requests_to_drain") == 2
    assert "release_kv_cache" in events


@pytest.mark.asyncio
async def test_vllm_replica_abort_request_tries_servers_until_one_aborts():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(replica_rank=0, config=make_config().actor_rollout_ref.rollout, model_config={"path": "x"})
    server_a = ScriptedRolloutServer.remote([], label="a")
    server_b = ScriptedRolloutServer.remote([], label="b")
    replica.servers = [server_a, server_b]

    result = await replica.abort_request("request-1")

    assert result["aborted"] is True
    assert result["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_checkpoint_manager_non_naive_update_orders_rollout_lifecycle(monkeypatch):
    from verl.checkpoint_engine import base as checkpoint_base
    from verl.checkpoint_engine.base import CheckpointEngineManager

    events = []

    class FakeReplica:
        workers = ["w0"]

        async def abort_all_requests(self):
            events.append("abort")

        async def release_kv_cache(self):
            events.append("release_kv")

        async def resume_kv_cache(self):
            events.append("resume_kv")

        async def resume_generation(self):
            events.append("resume_generation")

    class FakeTrainer:
        world_size = 1

        def update_weights(self, global_steps=None, mode="auto"):
            events.append(("trainer_update", global_steps, mode))
            return []

        def execute_checkpoint_engine(self, methods):
            events.append(("trainer_execute", methods))
            return []

    class FakeRolloutGroup:
        world_size = 1

        def __init__(self, *args, **kwargs):
            pass

        def update_weights(self, global_steps=None):
            events.append(("rollout_update", global_steps))
            return []

        def execute_checkpoint_engine(self, methods):
            events.append(("rollout_execute", methods))
            return []

    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", FakeRolloutGroup)
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.trainer = FakeTrainer()
    manager.replicas = [FakeReplica()]
    manager.build_process_group = lambda rollout: events.append("build_pg")

    await manager.update_weights(global_steps=12)

    assert events[:3] == ["abort", "release_kv", "build_pg"]
    assert ("trainer_update", 12, "nccl") in events
    assert ("rollout_update", 12) in events
    assert events[-2:] == ["resume_kv", "resume_generation"]


@pytest.mark.asyncio
async def test_checkpoint_manager_naive_update_delegates_to_trainer_only():
    from verl.checkpoint_engine.base import CheckpointEngineManager

    events = []

    class FakeTrainer:
        def update_weights(self, global_steps=None, mode="auto"):
            events.append(("trainer_update", global_steps, mode))
            return []

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "naive"
    manager.trainer = FakeTrainer()
    manager.replicas = []

    await manager.update_weights(global_steps=3)

    assert events == [("trainer_update", 3, "naive")]


@pytest.mark.asyncio
async def test_checkpoint_worker_passes_received_weights_and_global_steps_to_rollout():
    from verl.checkpoint_engine.base import CheckpointEngineWorker

    class FakeCheckpointEngine:
        def __init__(self):
            self.received_steps = []

        def receive_weights(self, global_steps=None):
            self.received_steps.append(global_steps)
            return [("layer.weight", 1)]

    class FakeServerAdapter:
        def __init__(self):
            self.calls = []

        async def update_weights(self, weights, global_steps=None):
            self.calls.append((weights, global_steps))

    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = FakeCheckpointEngine()
    worker.server_adapter = FakeServerAdapter()

    await worker.update_weights(global_steps=44)

    assert worker.checkpoint_engine.received_steps == [44]
    assert worker.server_adapter.calls == [([("layer.weight", 1)], 44)]


@pytest.mark.asyncio
async def test_manager_start_and_stop_profile_forwards_to_replicas():
    from verl.workers.rollout.llm_server import LLMServerManager

    class FakeReplica:
        def __init__(self):
            self.events = []

        async def start_profile(self, **kwargs):
            self.events.append(("start", kwargs))

        async def stop_profile(self):
            self.events.append(("stop", {}))

    replica = FakeReplica()
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_replicas = [replica]

    await manager.start_profile(trace_id="abc")
    await manager.stop_profile()

    assert replica.events == [("start", {"trace_id": "abc"}), ("stop", {})]
