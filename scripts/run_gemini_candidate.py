#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "task"
DEFAULT_RUNS_ROOT = Path("/mnt/data/projects/verl-vllm-gemini-runs")
DEFAULT_RUNTIME = Path("/mnt/data/projects/deep-swe/third_party/gemini_cli_runtime")
DEFAULT_TASK_IMAGE_CANDIDATES = (
    "verl-vllm-rollout-benchmark:task-v0",
    "verl-vllm-capability-benchmark:task-v0",
)
DEFAULT_TEST_IMAGE = "verl-vllm-rollout-benchmark:cpu-tests"
ALLOWED_TOOLS = (
    "read_file",
    "read_many_files",
    "write_file",
    "grep_search",
    "glob",
    "list_directory",
    "run_shell_command",
    "replace",
)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout=None,
    stderr=None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        check=True,
    )


def image_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def pick_task_image(explicit: str | None) -> str:
    if explicit:
        if not image_exists(explicit):
            raise SystemExit(f"task image not found: {explicit}")
        return explicit
    for image in DEFAULT_TASK_IMAGE_CANDIDATES:
        if image_exists(image):
            return image
    raise SystemExit(
        "no task image found; build one with "
        "`docker build -f docker/Dockerfile.eval -t verl-vllm-rollout-benchmark:task-v0 .`"
    )


def copy_clean_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        return {name for name in names if name in ignored}

    shutil.copytree(src, dst, ignore=ignore)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-._")
    return value or "gemini"


def selected_api_key(env: dict[str, str], key_index: int) -> str | None:
    if env.get("GEMINI_API_KEY"):
        return env["GEMINI_API_KEY"]
    keys = [k.strip() for k in env.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        return env.get("GOOGLE_API_KEY")
    return keys[key_index % len(keys)]


def write_settings(home: Path, model: str, max_session_turns: int, shell_timeout_sec: int) -> None:
    settings = {
        "general": {
            "maxSessionTurns": max_session_turns,
            "topicUpdateNarration": False,
            "enableNotifications": False,
        },
        "model": {
            "name": model,
            "compressionThreshold": 0.95,
        },
        "security": {
            "auth": {
                "selectedType": "gemini-api-key",
            },
        },
        "ui": {
            "loadingPhrases": "off",
        },
        "telemetry": {
            "enabled": False,
            "logPrompts": False,
        },
        "tools": {
            "shell": {
                "inactivityTimeout": shell_timeout_sec,
            },
        },
    }
    settings_dir = home / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n"
    )


def prepare_workspace(run_dir: Path) -> tuple[Path, Path]:
    workspace = run_dir / "workspace"
    verl_dir = workspace / "verl"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASK_ROOT / "PROMPT.md", workspace / "PROMPT.md")
    copy_clean_tree(TASK_ROOT / "verl", verl_dir)
    copy_clean_tree(TASK_ROOT / "vllm-src", workspace / "vllm-src")

    run(["git", "init", "-b", "main"], cwd=verl_dir, stdout=subprocess.DEVNULL)
    run(["git", "config", "user.email", "benchmark@example.invalid"], cwd=verl_dir)
    run(["git", "config", "user.name", "Benchmark Baseline"], cwd=verl_dir)
    run(["git", "add", "-A"], cwd=verl_dir)
    run(
        ["git", "commit", "-m", "clean sanitized baseline"],
        cwd=verl_dir,
        stdout=subprocess.DEVNULL,
    )
    return workspace, verl_dir


def write_agent_prompt(workspace: Path) -> Path:
    prompt = (workspace / "PROMPT.md").read_text()
    full_prompt = f"""You are working in /workspace/verl.

Implement the feature request below in the VERL checkout. You may inspect
/workspace/vllm-src as the local vLLM source tree. Keep existing non-vLLM
rollout paths working. Leave your final code changes in /workspace/verl.

{prompt}
"""
    prompt_path = workspace / ".gemini_prompt.txt"
    prompt_path.write_text(full_prompt)
    return prompt_path


def run_gemini(
    *,
    run_dir: Path,
    workspace: Path,
    runtime: Path,
    task_image: str,
    model: str,
    max_session_turns: int,
    shell_timeout_sec: int,
    timeout_sec: int | None,
    key_index: int,
) -> None:
    node = runtime / "bin" / "node"
    gemini_js = runtime / "bundle" / "gemini.js"
    if not node.exists() or not gemini_js.exists():
        raise SystemExit(f"Gemini CLI runtime is incomplete: {runtime}")

    gemini_home = workspace / ".gemini_home"
    write_settings(gemini_home, model, max_session_turns, shell_timeout_sec)
    write_agent_prompt(workspace)

    env = os.environ.copy()
    key = selected_api_key(env, key_index)
    if not key:
        raise SystemExit("set GEMINI_API_KEY or GEMINI_API_KEYS before running Gemini")
    env["GEMINI_API_KEY"] = key
    env["GOOGLE_API_KEY"] = key

    uid = str(os.getuid())
    gid = str(os.getgid())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/workspace/.gemini_home",
        "-e",
        "GEMINI_API_KEY",
        "-e",
        "GOOGLE_API_KEY",
        "-e",
        f"GEMINI_CLI_MODEL={model}",
        "-e",
        f"GEMINI_CLI_ALLOWED_TOOLS={','.join(ALLOWED_TOOLS)}",
        "-e",
        "GEMINI_CLI_RETRY_FOREVER_ON_429=1",
        "-e",
        "GEMINI_DEFAULT_AUTH_TYPE=gemini-api-key",
        "-e",
        "GEMINI_TELEMETRY_ENABLED=false",
        "-e",
        "GEMINI_TELEMETRY_LOG_PROMPTS=false",
        "-e",
        "GEMINI_SANDBOX=false",
        "-e",
        "GEMINI_CLI_TRUST_WORKSPACE=true",
        "-e",
        "NO_COLOR=1",
        "-e",
        "TERM=xterm-256color",
        "-e",
        "COLORTERM=truecolor",
        "-e",
        "CI=true",
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{runtime}:/opt/gemini_cli_runtime:ro",
        "-w",
        "/workspace/verl",
        task_image,
        "bash",
        "-lc",
        (
            "/opt/gemini_cli_runtime/bin/node "
            "/opt/gemini_cli_runtime/bundle/gemini.js "
            '--model "$GEMINI_CLI_MODEL" '
            "--approval-mode yolo "
            '--allowed-tools "$GEMINI_CLI_ALLOWED_TOOLS" '
            "--include-directories /workspace "
            "--output-format stream-json "
            '--prompt "" '
            "< /workspace/.gemini_prompt.txt"
        ),
    ]

    stdout_path = run_dir / "gemini_cli.stream.jsonl"
    stderr_path = run_dir / "gemini_cli.stderr.txt"
    print(f"running Gemini CLI in {task_image}; logs: {run_dir}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        started = time.time()
        subprocess.run(
            cmd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=True,
            timeout=timeout_sec,
        )
        elapsed = int(time.time() - started)
    print(f"Gemini CLI finished in {elapsed}s", flush=True)


def export_patch(verl_dir: Path, patch_path: Path) -> bool:
    run(["git", "add", "-A"], cwd=verl_dir)
    result = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=str(verl_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    patch_path.write_text(result.stdout)
    return bool(result.stdout.strip())


def apply_patch_to_fresh_checkout(patch_path: Path, run_dir: Path) -> Path:
    patched_dir = run_dir / "patched" / "verl"
    copy_clean_tree(TASK_ROOT / "verl", patched_dir)
    run(["git", "init", "-b", "main"], cwd=patched_dir, stdout=subprocess.DEVNULL)
    run(["git", "apply", str(patch_path)], cwd=patched_dir)
    return patched_dir


def run_cpu_tests(source_dir: Path, run_dir: Path, test_image: str, timeout_sec: int | None) -> int:
    env = os.environ.copy()
    env["TEST_IMAGE"] = test_image
    stdout_path = run_dir / "cpu_tests.stdout.txt"
    stderr_path = run_dir / "cpu_tests.stderr.txt"
    cmd = [str(REPO_ROOT / "eval_tests" / "run_cpu_tests.sh"), str(source_dir)]
    print(f"running CPU eval tests against {source_dir}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_sec,
        )
    print(f"CPU eval tests exited with {result.returncode}", flush=True)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gemini CLI candidate on the sanitized VERL vLLM rollout task."
    )
    parser.add_argument("--run-name", default="gemini", help="name suffix for the run directory")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--gemini-runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--task-image", default=None)
    parser.add_argument("--test-image", default=DEFAULT_TEST_IMAGE)
    parser.add_argument("--model", default=os.environ.get("GEMINI_CLI_MODEL", "gemini-3.5-flash"))
    parser.add_argument("--key-index", type=int, default=0)
    parser.add_argument("--max-session-turns", type=int, default=200)
    parser.add_argument("--shell-timeout-sec", type=int, default=180)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=7200,
        help="Gemini timeout in seconds; use 0 for no timeout",
    )
    parser.add_argument(
        "--test-timeout-sec",
        type=int,
        default=3600,
        help="CPU test timeout in seconds; use 0 for no timeout",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_image = pick_task_image(args.task_image)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_root / f"{timestamp}-{slugify(args.run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    workspace, verl_dir = prepare_workspace(run_dir)
    print(f"prepared candidate workspace: {workspace}", flush=True)
    print(f"task image: {task_image}", flush=True)

    if args.prepare_only:
        print(f"prepare-only run directory: {run_dir}", flush=True)
        return 0

    timeout = None if args.timeout_sec == 0 else args.timeout_sec
    test_timeout = None if args.test_timeout_sec == 0 else args.test_timeout_sec

    run_gemini(
        run_dir=run_dir,
        workspace=workspace,
        runtime=args.gemini_runtime,
        task_image=task_image,
        model=args.model,
        max_session_turns=args.max_session_turns,
        shell_timeout_sec=args.shell_timeout_sec,
        timeout_sec=timeout,
        key_index=args.key_index,
    )

    patch_path = run_dir / "candidate.patch"
    has_patch = export_patch(verl_dir, patch_path)
    print(f"candidate patch: {patch_path}", flush=True)
    if not has_patch:
        print("Gemini produced no code changes", flush=True)
        return 1

    if args.skip_tests:
        return 0

    patched_dir = apply_patch_to_fresh_checkout(patch_path, run_dir)
    return run_cpu_tests(patched_dir, run_dir, args.test_image, test_timeout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
