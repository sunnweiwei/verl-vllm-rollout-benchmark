import importlib.util
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_diagnose_module():
    diagnose_path = Path.cwd() / "scripts" / "diagnose.py"
    if not diagnose_path.exists():
        pytest.skip("This implementation does not ship the optional diagnose script")
    spec = importlib.util.spec_from_file_location("verl_eval_diagnose_script", diagnose_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnose_collects_runtime_metadata_without_touching_real_network_or_gpu(monkeypatch, capsys):
    diagnose = _load_diagnose_module()
    calls = []

    monkeypatch.setattr(diagnose.platform, "python_version", lambda: "3.11.test")
    monkeypatch.setattr(diagnose.platform, "python_compiler", lambda: "compiler")
    monkeypatch.setattr(diagnose.platform, "python_build", lambda: ("build", "date"))
    monkeypatch.setattr(diagnose.platform, "architecture", lambda: ("64bit", "ELF"))
    monkeypatch.setattr(diagnose.platform, "platform", lambda: "Linux-test")
    monkeypatch.setattr(diagnose.platform, "system", lambda: "Linux")
    monkeypatch.setattr(diagnose.platform, "node", lambda: "node-a")
    monkeypatch.setattr(diagnose.platform, "release", lambda: "kernel")
    monkeypatch.setattr(diagnose.platform, "version", lambda: "version")
    monkeypatch.setattr(diagnose.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diagnose.platform, "processor", lambda: "H100 host CPU")
    monkeypatch.setattr(diagnose.sys, "platform", "linux")

    fake_verl = types.ModuleType("verl")
    fake_verl.__version__ = "eval-version"
    fake_verl.__file__ = str(Path.cwd() / "verl" / "__init__.py")
    monkeypatch.setitem(sys.modules, "verl", fake_verl)

    fake_pip = types.ModuleType("pip")
    fake_pip.__version__ = "pip-version"
    fake_pip.__file__ = "/tmp/fake-pip/__init__.py"
    monkeypatch.setitem(sys.modules, "pip", fake_pip)

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(("run", tuple(cmd), capture_output, text, check))
        if cmd[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="abc123\n", stderr="", returncode=0)
        if cmd and cmd[0] == "nvidia-smi":
            return SimpleNamespace(stdout="NVIDIA H100, 81559\nNVIDIA H100, 81559\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected subprocess.run command: {cmd}")

    monkeypatch.setattr(diagnose.subprocess, "run", fake_run)
    monkeypatch.setattr(diagnose.subprocess, "call", lambda cmd: calls.append(("call", tuple(cmd))) or 0)
    monkeypatch.setattr(
        diagnose.subprocess,
        "check_output",
        lambda cmd: b"Cuda compilation tools, release 12.4, V12.4.0\n",
    )
    def fake_package_version(package):
        versions = {"vllm": "0.10.0", "ray": "2.48.0", "torch": "2.8.0"}
        if package not in versions:
            raise diagnose.importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(diagnose.importlib.metadata, "version", fake_package_version)
    monkeypatch.setattr(diagnose.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnose.torch.version, "cuda", "12.4", raising=False)
    monkeypatch.setattr(diagnose.psutil, "virtual_memory", lambda: SimpleNamespace(total=128 * 1024**3))
    monkeypatch.setattr(diagnose.socket, "gethostbyname", lambda host: "127.0.0.1")
    monkeypatch.setattr(diagnose, "urlopen", lambda url, timeout: SimpleNamespace(status=200))
    monkeypatch.setenv("VERL_DIAG_TEST", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    diagnose.check_python()
    diagnose.check_pip()
    assert diagnose._get_current_git_commit() == "abc123"
    diagnose.check_verl()
    diagnose.check_os()
    diagnose.check_hardware()
    diagnose.check_network(SimpleNamespace(timeout=3, region="cn,unknown-region"))
    diagnose.check_environment()
    diagnose.check_pip_package_versions()
    diagnose.check_cuda_versions()
    assert diagnose._get_cpu_memory() == 128
    gpu_count, gpu_info = diagnose._get_gpu_info()
    assert gpu_count == 2
    assert gpu_info[0]["type"] == "NVIDIA H100"
    assert gpu_info[0]["memory"] > 79
    assert diagnose._get_system_info()["gpu_count"] == 2
    diagnose.check_system_info()

    out = capsys.readouterr().out
    assert "Python Info" in out
    assert "verl Info" in out
    assert "CUDA Runtime : 12.4" in out
    assert "vllm" in out
    assert "GPU Count" in out
    assert any(call[0] == "call" and call[1] == ("lscpu",) for call in calls)


def test_diagnose_handles_missing_commands_cuda_and_arguments(monkeypatch, capsys):
    diagnose = _load_diagnose_module()

    def missing_run(cmd, capture_output=False, text=False, check=False):
        if cmd[:2] == ["git", "rev-parse"]:
            raise subprocess.CalledProcessError(128, cmd, stderr="not a git repo")
        if cmd and cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise AssertionError(f"unexpected subprocess.run command: {cmd}")

    monkeypatch.setattr(diagnose.subprocess, "run", missing_run)
    monkeypatch.setattr(diagnose.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diagnose.socket, "gethostbyname", lambda _host: (_ for _ in ()).throw(OSError("dns failed")))

    assert diagnose._get_current_git_commit() is None
    assert diagnose._get_gpu_info() == (0, [])
    diagnose.check_cuda_versions()
    diagnose.test_connection("PYPI", "https://pypi.python.org/pypi/pip", timeout=1)

    monkeypatch.setattr(
        diagnose.sys,
        "argv",
        [
            "diagnose.py",
            "--python",
            "0",
            "--pip",
            "0",
            "--network",
            "1",
            "--region",
            "cn",
            "--timeout",
            "0",
        ],
    )
    args = diagnose.parse_args()
    assert args.python == 0
    assert args.network == 1
    assert args.region == "cn"
    assert args.timeout == 0

    out = capsys.readouterr().out
    assert "Error running git command" in out
    assert "Failed to execute nvidia-smi command" in out
    assert "CUDA is not available" in out
    assert "Error resolving DNS" in out


def test_diagnose_cli_entrypoint_dispatches_safe_default_sections(monkeypatch, capsys):
    diagnose_path = Path.cwd() / "scripts" / "diagnose.py"
    if not diagnose_path.exists():
        pytest.skip("This implementation does not ship the optional diagnose script")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose.py",
            "--verl",
            "0",
            "--hardware",
            "0",
            "--network",
            "0",
        ],
    )
    runpy.run_path(str(diagnose_path), run_name="__main__")

    out = capsys.readouterr().out
    assert "Python Info" in out
    assert "Pip Info" in out
    assert "Platform Info" in out
    assert "Environment" in out
    assert "System Info" in out
