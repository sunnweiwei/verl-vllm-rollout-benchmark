#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GEMINI_RUNNER_PATH = REPO_ROOT / "scripts" / "run_gemini_test_generator.py"
DEFAULT_CODEX_BIN = Path("/home/sunweiwei_google_com/.codex/packages/standalone/current/bin/codex")
DEFAULT_CODEX_HOME = Path("/home/sunweiwei_google_com/.codex")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("run_gemini_test_generator", GEMINI_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base runner from {GEMINI_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def codex_container_name(run_dir: Path) -> str:
    safe_name = base.re.sub(r"[^A-Za-z0-9_.-]+", "-", run_dir.name).strip("-._")
    return f"codex-testgen-{safe_name[:80]}-{uuid.uuid4().hex[:8]}"


def codex_docker_env_args() -> list[str]:
    return [
        "-e",
        "CODEX_HOME=/codex_home",
        "-e",
        "HOME=/workspace/.codex_runtime_home",
        "-e",
        "NO_COLOR=1",
        "-e",
        "TERM=xterm-256color",
        "-e",
        "CI=true",
    ]


def codex_mount_args(args: argparse.Namespace, workspace: Path) -> list[str]:
    return [
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{args.codex_bin.parent.parent}:/opt/codex:ro",
        "-v",
        f"{args.codex_home}:/codex_home",
    ]


def ensure_codex_runtime(args: argparse.Namespace) -> None:
    if not args.codex_bin.exists():
        raise SystemExit(f"Codex binary not found: {args.codex_bin}")
    if not args.codex_home.exists():
        raise SystemExit(f"Codex home not found: {args.codex_home}")
    if not base.image_exists(args.image):
        raise SystemExit(f"Docker image not found: {args.image}")


def extract_codex_thread_id(stdout_path: Path) -> str:
    thread_ids: list[str] = []
    for line in stdout_path.read_text(errors="ignore").splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            thread_ids.append(event["thread_id"])
    if not thread_ids:
        raise RuntimeError(f"could not find Codex thread_id in {stdout_path}")
    return thread_ids[-1]


def start_codex_container(args: argparse.Namespace, *, workspace: Path, container_name: str) -> None:
    ensure_codex_runtime(args)
    uid = str(os.getuid())
    gid = str(os.getgid())
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--user",
        f"{uid}:{gid}",
        *codex_docker_env_args(),
        *codex_mount_args(args, workspace),
        "-w",
        "/workspace",
        args.image,
        "bash",
        "-lc",
        "trap 'exit 0' TERM INT; while true; do sleep 3600 & wait $!; done",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


def stop_codex_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def exec_codex_in_container(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    container_name: str,
    prompt_file: str,
    log_prefix: str,
    resume_thread_id: str | None = None,
) -> str | None:
    prompt_path = Path("/workspace") / prompt_file
    last_message = Path("/workspace") / f".codex_last_{base.slugify(log_prefix)}.txt"
    if resume_thread_id is None:
        codex_command = (
            "/opt/codex/bin/codex "
            "exec "
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            "--ignore-rules "
            "--json "
            f"--model {args.model} "
            "--cd /workspace "
            f"--output-last-message {last_message} "
            f"- < {prompt_path}"
        )
    else:
        codex_command = (
            "/opt/codex/bin/codex "
            "exec resume "
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            "--ignore-rules "
            "--json "
            f"--model {args.model} "
            f"--output-last-message {last_message} "
            f"{resume_thread_id} "
            f"- < {prompt_path}"
        )

    cmd = [
        "docker",
        "exec",
        "-i",
        container_name,
        "bash",
        "-lc",
        codex_command,
    ]
    stdout_path = run_dir / f"{log_prefix}.jsonl"
    stderr_path = run_dir / f"{log_prefix}.stderr.txt"
    print(f"running Codex CLI in persistent container {container_name}; logs: {run_dir}", flush=True)
    started = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=True,
            timeout=None if args.timeout_sec == 0 else args.timeout_sec,
        )
    print(f"Codex CLI finished in {int(time.time() - started)}s", flush=True)
    if resume_thread_id is None:
        return extract_codex_thread_id(stdout_path)
    return None


def run_codex(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    *,
    prompt_file: str = ".gemini_prompt.txt",
    log_prefix: str = "codex_cli",
) -> None:
    ensure_codex_runtime(args)

    uid = str(os.getuid())
    gid = str(os.getgid())
    prompt_path = Path("/workspace") / prompt_file
    last_message = Path("/workspace") / f".codex_last_{base.slugify(log_prefix)}.txt"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-e",
        "CODEX_HOME=/codex_home",
        "-e",
        "HOME=/workspace/.codex_runtime_home",
        "-e",
        "NO_COLOR=1",
        "-e",
        "TERM=xterm-256color",
        "-e",
        "CI=true",
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{args.codex_bin.parent.parent}:/opt/codex:ro",
        "-v",
        f"{args.codex_home}:/codex_home",
        "-w",
        "/workspace",
        args.image,
        "bash",
        "-lc",
        (
            "/opt/codex/bin/codex "
            "--dangerously-bypass-approvals-and-sandbox "
            "exec "
            "--ephemeral "
            "--skip-git-repo-check "
            "--ignore-rules "
            f"--model {args.model} "
            "--cd /workspace "
            f"--output-last-message {last_message} "
            f"- < {prompt_path}"
        ),
    ]

    stdout_path = run_dir / f"{log_prefix}.stdout.txt"
    stderr_path = run_dir / f"{log_prefix}.stderr.txt"
    print(f"running Codex CLI in {args.image}; logs: {run_dir}", flush=True)
    started = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=True,
            timeout=None if args.timeout_sec == 0 else args.timeout_sec,
        )
    print(f"Codex CLI finished in {int(time.time() - started)}s", flush=True)


def run_staged_codex(args: argparse.Namespace, run_dir: Path, workspace: Path) -> None:
    for stage in (1, 2, 3, 4, 5):
        prompt_name = base.write_stage_prompt(workspace, stage)
        moved_reference = None
        if stage != 2:
            moved_reference = base.move_workspace_path(workspace, run_dir, "reference_project", f"codex_stage{stage}")
        try:
            run_codex(
                args,
                run_dir,
                workspace,
                prompt_file=prompt_name,
                log_prefix=f"codex_stage{stage}",
            )
        finally:
            base.restore_workspace_path(moved_reference)
        findings = base.stage_audit_findings(
            run_dir,
            workspace,
            stage=stage,
            filename=f"codex_stage_audit_stage{stage}.txt",
        )
        if findings:
            print(
                f"codex stage {stage} audit findings: " + " | ".join(findings[:8]),
                flush=True,
            )


def run_two_stage_codex(args: argparse.Namespace, run_dir: Path, workspace: Path) -> None:
    stage1_prompt_name, stage2_prompt_name = base.write_two_stage_prompt_files(args, workspace)
    container_name = codex_container_name(run_dir)
    moved_reference = None
    try:
        moved_reference = base.move_workspace_path(workspace, run_dir, "reference_project", "codex_two_stage1")
        start_codex_container(args, workspace=workspace, container_name=container_name)
        thread_id = exec_codex_in_container(
            args,
            run_dir,
            container_name=container_name,
            prompt_file=stage1_prompt_name,
            log_prefix="codex_twostage1",
        )
        if thread_id is None:
            raise RuntimeError("Codex stage 1 did not return a thread id")
        (run_dir / "codex_two_stage_thread_id.txt").write_text(thread_id + "\n")
        base.restore_workspace_path(moved_reference)
        moved_reference = None

        exec_codex_in_container(
            args,
            run_dir,
            container_name=container_name,
            prompt_file=stage2_prompt_name,
            log_prefix="codex_twostage2",
            resume_thread_id=thread_id,
        )
    finally:
        base.restore_workspace_path(moved_reference)
        stop_codex_container(container_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluator tests with Codex CLI using the same staged benchmark inputs."
    )
    parser.add_argument("--run-name", default="codex-testgen")
    parser.add_argument("--runs-root", type=Path, default=base.DEFAULT_RUNS_ROOT)
    parser.add_argument("--image", default=base.DEFAULT_IMAGE)
    parser.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.5"))
    parser.add_argument("--key-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--codex-bin", type=Path, default=DEFAULT_CODEX_BIN)
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--project-source", type=Path, default=REPO_ROOT / "task" / "verl")
    parser.add_argument("--agent-prompt", type=Path, default=REPO_ROOT / "task" / "PROMPT.md")
    parser.add_argument("--reference-project", type=Path, default=base.DEFAULT_REFERENCE_PROJECT)
    parser.add_argument("--generator-prompt", type=Path, default=base.DEFAULT_GENERATOR_PROMPT)
    parser.add_argument("--generator-prompt-stage1", type=Path, default=base.DEFAULT_GENERATOR_PROMPT_STAGE1)
    parser.add_argument("--generator-prompt-stage2", type=Path, default=base.DEFAULT_GENERATOR_PROMPT_STAGE2)
    parser.add_argument("--candidate-patch", type=Path, action="append", default=[])
    parser.add_argument("--candidate-runs-root", type=Path, default=base.DEFAULT_CANDIDATE_RUNS_ROOT)
    parser.add_argument(
        "--validation-target",
        action="append",
        default=[],
        metavar="LABEL=/path/to/project",
        help="additional known-good target expected to pass the generated evaluator",
    )
    parser.add_argument(
        "--no-default-validation-targets",
        action="store_true",
        help="do not auto-add repo-default known-good validation targets",
    )
    parser.add_argument("--max-candidate-patches", type=int, default=0)
    parser.add_argument("--score-manual-suite", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="Codex timeout per stage; 0 disables timeout")
    parser.add_argument("--test-timeout-sec", type=int, default=240)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--generation-mode", choices=("single", "staged", "two-stage"), default="single")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--existing-generated-tests", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")

    base.run_gemini = run_codex
    base.run_staged_gemini = run_staged_codex
    base.run_two_stage_gemini = run_two_stage_codex

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    slug = base.slugify(args.run_name)

    if args.num_runs == 1:
        run_dir = args.runs_root / f"testgen-{slug}-{timestamp}"
        summary = base.run_once(args, run_dir, 0)
        if args.prepare_only or args.skip_validation:
            return 0
        return (
            0
            if summary["reference_exit_code"] == 0
            and summary["baseline_exit_code"] != 0
            and not summary.get("quality_issues")
            else 1
        )

    batch_dir = args.runs_root / f"testgen-{slug}-{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.num_runs):
        run_dir = batch_dir / f"run-{index + 1:02d}"
        try:
            results.append(base.run_once(args, run_dir, index))
        except Exception as exc:
            result = {
                "run_dir": str(run_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"run {index + 1} failed: {result['error']}", flush=True)
            results.append(result)

    (batch_dir / "batch_summary.json").write_text(
        base.json.dumps({"batch_dir": str(batch_dir), "num_runs": args.num_runs, "results": results}, indent=2)
        + "\n"
    )
    if args.prepare_only or args.skip_validation:
        return 0
    acceptable = [
        result
        for result in results
        if result.get("reference_exit_code") == 0
        and result.get("baseline_exit_code") not in (0, None)
        and not result.get("quality_issues")
    ]
    return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
