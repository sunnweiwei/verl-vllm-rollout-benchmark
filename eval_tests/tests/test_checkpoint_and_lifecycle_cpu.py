from types import SimpleNamespace

import pytest
import ray

from vllm_eval_helpers import ScriptedRolloutServer, make_config


def _make_vllm_replica_with_servers():
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
    replica._server_handle = server_a
    replica._server_address = "fake-vllm://replica-0"
    return replica, server_a, server_b


async def _event_names(*servers):
    return [[name for name, _kwargs in await server.get_events.remote()] for server in servers]


@pytest.mark.asyncio
async def test_vllm_replica_wake_up_forwards_to_servers():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.wake_up()

    for names in await _event_names(server_a, server_b):
        assert "wake_up" in names


@pytest.mark.asyncio
async def test_vllm_replica_sleep_forwards_to_servers():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.sleep()

    for names in await _event_names(server_a, server_b):
        assert "sleep" in names


@pytest.mark.asyncio
async def test_vllm_replica_resume_generation_forwards_to_servers():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.resume_generation()

    for names in await _event_names(server_a, server_b):
        assert "resume_generation" in names


@pytest.mark.asyncio
async def test_vllm_replica_kv_cache_controls_forward_to_servers():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.clear_kv_cache()
    await replica.release_kv_cache()
    await replica.resume_kv_cache()

    for names in await _event_names(server_a, server_b):
        assert "clear_kv_cache" in names
        assert "release_kv_cache" in names
        assert "resume_kv_cache" in names


@pytest.mark.asyncio
async def test_vllm_replica_profile_forwards_to_servers():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.start_profile(trace_id="cpu-test")
    await replica.stop_profile()

    for events in [await server_a.get_events.remote(), await server_b.get_events.remote()]:
        names = [name for name, _kwargs in events]
        assert "start_profile" in names
        assert "stop_profile" in names
        assert ("start_profile", {"trace_id": "cpu-test"}) in events


@pytest.mark.asyncio
async def test_vllm_replica_abort_all_requests_calls_each_server():
    replica, server_a, server_b = _make_vllm_replica_with_servers()

    await replica.abort_all_requests()

    for names in await _event_names(server_a, server_b):
        assert "abort_all_requests" in names


@pytest.mark.asyncio
async def test_vllm_replica_abort_all_requests_aggregates_server_results():
    replica, _server_a, _server_b = _make_vllm_replica_with_servers()

    abort_result = await replica.abort_all_requests()

    assert abort_result["aborted_count"] == 2
    assert sorted(abort_result["request_ids"]) == ["a-request", "b-request"]


def _make_single_server_vllm_replica():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(replica_rank=0, config=make_config().actor_rollout_ref.rollout, model_config={"path": "x"})
    server = ScriptedRolloutServer.remote([], label="a")
    replica.servers = [server]
    replica._server_handle = server
    return replica, server


@pytest.mark.asyncio
async def test_vllm_replica_sleep_waits_for_requests_to_drain_first():
    replica, server = _make_single_server_vllm_replica()

    await replica.sleep()

    events = [name for name, _ in await server.get_events.remote()]
    assert events.index("wait_for_requests_to_drain") < events.index("sleep")
    assert events.count("wait_for_requests_to_drain") == 1


@pytest.mark.asyncio
async def test_vllm_replica_release_kv_cache_waits_for_requests_to_drain_first():
    replica, server = _make_single_server_vllm_replica()

    await replica.release_kv_cache()

    events = [name for name, _ in await server.get_events.remote()]
    assert events.index("wait_for_requests_to_drain") < events.index("release_kv_cache")
    assert events.count("wait_for_requests_to_drain") == 1
    assert "release_kv_cache" in events


@pytest.mark.asyncio
async def test_vllm_replica_abort_request_tries_servers_until_one_aborts():
    from verl.workers.rollout.replica import get_rollout_replica_class

    replica_cls = get_rollout_replica_class("vllm")
    replica = replica_cls(replica_rank=0, config=make_config().actor_rollout_ref.rollout, model_config={"path": "x"})
    server_a = ScriptedRolloutServer.remote([], label="a")
    server_b = ScriptedRolloutServer.remote([], label="b")
    replica.servers = [server_a, server_b]
    replica._server_handle = server_a

    result = await replica.abort_request("request-1")

    assert result["aborted"] is True
    assert result["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_vllm_async_rollout_adapter_resume_and_release_use_server_lifecycle():
    from verl.workers.rollout.base import get_rollout_class

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        async def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))

    class FakeServerHandle:
        wake_up = RemoteMethod("wake_up")
        sleep = RemoteMethod("sleep")

    rollout_cls = get_rollout_class("vllm", "async")
    rollout = rollout_cls.__new__(rollout_cls)
    rollout.config = SimpleNamespace(free_cache_engine=True)
    rollout.rollout_rank = 0
    rollout.server_handle = FakeServerHandle()

    await rollout.resume(tags=["weights"])
    await rollout.release()

    assert events == [
        ("wake_up", (), {"tags": ["weights"]}),
        ("sleep", (), {}),
    ]


@pytest.mark.asyncio
async def test_vllm_async_rollout_adapter_lazily_finds_server_actor(monkeypatch):
    import ray

    from verl.workers.rollout.base import get_rollout_class

    events = []
    actor_names = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        async def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))

    class FakeServerHandle:
        wake_up = RemoteMethod("wake_up")

    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    def fake_get_actor(name):
        actor_names.append(name)
        return FakeServerHandle()

    monkeypatch.setattr(ray, "get_actor", fake_get_actor)

    rollout_cls = get_rollout_class("vllm", "async")
    rollout = rollout_cls.__new__(rollout_cls)
    rollout.config = Config(name="vllm", free_cache_engine=True)
    rollout.rollout_rank = 0
    rollout.replica_rank = 2
    rollout.node_rank = 3
    rollout.server_handle = None

    await rollout.resume(tags=["kv_cache"])

    assert actor_names == ["vllm_server_2_3"]
    assert events == [("wake_up", (), {"tags": ["kv_cache"]})]
    assert rollout.server_handle is not None


@pytest.mark.asyncio
async def test_vllm_async_rollout_adapter_nonzero_rank_skips_server_lifecycle():
    from verl.workers.rollout.base import get_rollout_class

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        async def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))

    class FakeServerHandle:
        wake_up = RemoteMethod("wake_up")
        sleep = RemoteMethod("sleep")

    rollout_cls = get_rollout_class("vllm", "async")
    rollout = rollout_cls.__new__(rollout_cls)
    rollout.config = SimpleNamespace(free_cache_engine=True)
    rollout.rollout_rank = 1
    rollout.server_handle = FakeServerHandle()

    await rollout.resume(tags=["weights"])
    await rollout.release()

    assert events == []


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
