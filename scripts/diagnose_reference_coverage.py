#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = Path("/mnt/data/projects/verl-vllm-testgen-runs")
DEFAULT_IMAGE = "verl-vllm-rollout-benchmark:cpu-tests"
DEFAULT_PROJECT_SOURCE = Path("/mnt/data/projects/verl-vllm-capability-benchmark/task/verl")
DEFAULT_REFERENCE_PROJECT = Path("/mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0")
DEFAULT_COUNTER_REFERENCE_PROJECT = Path(
    "/mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project"
)

EXCLUDED_PATH_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "docs",
    "examples",
}


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-._")
    return value or "target"


def copy_clean_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".gemini_home",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }
        return {name for name in names if name in ignored}

    shutil.copytree(src, dst, ignore=ignore)


def parse_reference_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--reference must be label=/path/to/project, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = slugify(label)
    path = Path(raw_path).expanduser()
    if not path.is_dir():
        raise SystemExit(f"reference target does not exist or is not a directory for {label}: {path}")
    return label, path


def default_references() -> list[tuple[str, Path]]:
    refs: list[tuple[str, Path]] = []
    if DEFAULT_REFERENCE_PROJECT.exists():
        refs.append(("official_reference", DEFAULT_REFERENCE_PROJECT))
    if DEFAULT_COUNTER_REFERENCE_PROJECT.exists():
        refs.append(("counter_reference", DEFAULT_COUNTER_REFERENCE_PROJECT))
    return refs


def path_matches_any(value: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def should_consider_python_file(
    path: Path,
    *,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
) -> bool:
    if path.suffix != ".py":
        return False
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return False
    rel_text = path.as_posix()
    if include_patterns and not path_matches_any(rel_text, include_patterns):
        return False
    if exclude_patterns and path_matches_any(rel_text, exclude_patterns):
        return False
    return True


def is_codeish_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def changed_lines_for_file(base_file: Path, target_file: Path) -> set[int]:
    base_lines = base_file.read_text(errors="ignore").splitlines() if base_file.exists() else []
    target_lines = target_file.read_text(errors="ignore").splitlines()
    matcher = difflib.SequenceMatcher(None, base_lines, target_lines, autojunk=False)
    changed: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in {"replace", "insert"}:
            continue
        for lineno in range(j1 + 1, j2 + 1):
            if lineno <= len(target_lines) and is_codeish_line(target_lines[lineno - 1]):
                changed.add(lineno)
    return changed


def build_changed_line_map(
    base_root: Path,
    target_root: Path,
    *,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    for target_file in sorted(target_root.rglob("*.py")):
        rel = target_file.relative_to(target_root)
        if not should_consider_python_file(rel, include_patterns=include_patterns, exclude_patterns=exclude_patterns):
            continue
        rel_text = rel.as_posix()
        line_set = changed_lines_for_file(base_root / rel, target_file)
        if line_set:
            changed[rel_text] = line_set
    return changed


def iter_node_body_lines(node: ast.AST) -> set[int]:
    lines: set[int] = set()
    body = getattr(node, "body", None)
    if not isinstance(body, list):
        return lines
    for child in body:
        start = getattr(child, "lineno", None)
        end = getattr(child, "end_lineno", None) or start
        if isinstance(start, int) and isinstance(end, int):
            lines.update(range(start, end + 1))
    return lines


def iter_class_activation_lines(node: ast.ClassDef) -> set[int]:
    lines: set[int] = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_lines = iter_node_body_lines(child)
            method_lines.discard(child.lineno)
            lines.update(method_lines)
            continue
        if isinstance(child, ast.ClassDef):
            lines.update(iter_class_activation_lines(child))
            continue
        start = getattr(child, "lineno", None)
        end = getattr(child, "end_lineno", None) or start
        if isinstance(start, int) and isinstance(end, int):
            lines.update(range(start, end + 1))
    lines.discard(node.lineno)
    return lines


def collect_changed_symbols(target_root: Path, changed: dict[str, set[int]]) -> dict[str, dict[str, list[dict[str, object]]]]:
    symbols: dict[str, dict[str, list[dict[str, object]]]] = {}
    for rel, changed_lines in changed.items():
        path = target_root / rel
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        file_symbols = {"functions": [], "classes": []}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or node.lineno
                if changed_lines.intersection(range(start, end + 1)):
                    body_lines = iter_node_body_lines(node)
                    file_symbols["functions"].append(
                        {"name": node.name, "start": start, "end": end, "body_lines": sorted(body_lines)}
                    )
            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or node.lineno
                if changed_lines.intersection(range(start, end + 1)):
                    body_lines = iter_class_activation_lines(node)
                    file_symbols["classes"].append(
                        {"name": node.name, "start": start, "end": end, "body_lines": sorted(body_lines)}
                    )
        symbols[rel] = file_symbols
    return symbols


TRACE_RUNNER = r"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

tests_dir = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
pytest_targets = sys.argv[3:] or [str(tests_dir)]
raw_dir = Path(os.environ["VERL_EVAL_COVERAGE_OUTPUT_DIR"]).resolve()


try:
    exit_code = pytest.main(["-q", *pytest_targets])
finally:
    module = sys.modules.get("sitecustomize")
    if module is not None and hasattr(module, "write_counts_now"):
        module.write_counts_now()

# Give Ray actors and spawned engine workers a short window to run their own
# atexit handlers after pytest has torn down managers and Ray contexts.
time.sleep(float(os.environ.get("VERL_EVAL_COVERAGE_AGGREGATE_DELAY_SEC", "2")))

counts = {}
for path in sorted(raw_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue
    for rel, rel_counts in payload.get("counts", {}).items():
        merged = counts.setdefault(rel, {})
        for line, value in rel_counts.items():
            merged[line] = merged.get(line, 0) + int(value)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps({"exit_code": exit_code, "counts": counts}, sort_keys=True) + "\n")
sys.exit(exit_code)
"""


SITE_CUSTOMIZE = r"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import signal
import sys
import threading
import uuid

target_root = os.environ.get("VERL_EVAL_COVERAGE_TARGET_ROOT")
output_dir = os.environ.get("VERL_EVAL_COVERAGE_OUTPUT_DIR")
changed_lines_file = os.environ.get("VERL_EVAL_COVERAGE_CHANGED_LINES_FILE")

_targets = None
if changed_lines_file:
    try:
        _raw_targets = json.loads(Path(changed_lines_file).read_text())
        _targets = {
            str(rel): {int(line) for line in lines}
            for rel, lines in _raw_targets.items()
            if isinstance(lines, list)
        }
    except Exception:
        _targets = {}

_enabled = bool(target_root and output_dir and (_targets is None or _targets))
_prefix = str(Path(target_root).resolve()) + os.sep if _enabled else ""
_counts = {}
_wrote = False
_file_tracers = {}


def _get_line_tracer(rel):
    tracer = _file_tracers.get(rel)
    if tracer is not None:
        return tracer
    target_lines = _targets.get(rel) if _targets is not None else None

    def _line_tracer(frame, event, arg):
        if event == "line" and (target_lines is None or frame.f_lineno in target_lines):
            rel_counts = _counts.setdefault(rel, {})
            key = str(frame.f_lineno)
            rel_counts[key] = rel_counts.get(key, 0) + 1
        return _line_tracer

    _file_tracers[rel] = _line_tracer
    return _line_tracer


def _global_tracer(frame, event, arg):
    if event != "call":
        return None
    filename = frame.f_code.co_filename
    if not filename.startswith(_prefix):
        return None
    rel = filename[len(_prefix):]
    if _targets is not None and rel not in _targets:
        return None
    return _get_line_tracer(rel)


def write_counts_now():
    global _wrote
    if not _enabled or _wrote:
        return
    _wrote = True
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "counts": _counts,
    }
    out_path = out_dir / f"{os.getpid()}-{uuid.uuid4().hex}.json"
    try:
        out_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def _handle_signal(signum, frame):
    write_counts_now()
    previous = _previous_handlers.get(signum)
    if callable(previous):
        previous(signum, frame)
    elif previous == signal.SIG_IGN:
        return
    else:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


_previous_handlers = {}
if _enabled:
    sys.settrace(_global_tracer)
    threading.settrace(_global_tracer)
    atexit.register(write_counts_now)
    for _signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if _signum is None:
            continue
        try:
            _previous_handlers[_signum] = signal.getsignal(_signum)
            signal.signal(_signum, _handle_signal)
        except Exception:
            pass
"""


def run_trace_for_target(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    label: str,
    rel_target: str,
    changed: dict[str, set[int]],
) -> dict[str, object]:
    uid = str(os.getuid())
    gid = str(os.getgid())
    target_path = f"/workspace/{rel_target}"
    out_rel = f"coverage/{label}/trace_counts.json"
    raw_rel = f"coverage/{label}/raw"
    changed_rel = f"coverage/{label}/changed_lines.json"
    changed_path = workspace / changed_rel
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_text(json.dumps({rel: sorted(lines) for rel, lines in changed.items()}, sort_keys=True) + "\n")
    pytest_targets = []
    for target in args.pytest_target:
        if target.startswith("/workspace/"):
            pytest_targets.append(target)
        elif target.startswith("generated_tests/"):
            pytest_targets.append(f"/workspace/{target}")
        else:
            pytest_targets.append(f"/workspace/generated_tests/{target}")
    stdout_path = run_dir / f"coverage_{label}.stdout.txt"
    stderr_path = run_dir / f"coverage_{label}.stderr.txt"
    cmd = [
        "docker",
        "run",
        "--rm",
    ]
    if args.docker_gpus:
        cmd += ["--gpus", args.docker_gpus]
    if args.docker_ipc:
        cmd += ["--ipc", args.docker_ipc]
    if args.docker_shm_size:
        cmd += ["--shm-size", args.docker_shm_size]
    for volume in args.docker_volume:
        cmd += ["-v", volume]
    cmd += [
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/workspace/.coverage_home",
        "-e",
        "USER=tester",
        "-e",
        "LOGNAME=tester",
        "-e",
        "PYTEST_ADDOPTS=-p no:cacheprovider",
        "-e",
        f"PROJECT_UNDER_TEST={target_path}",
        "-e",
        f"VERL_EVAL_COVERAGE_TARGET_ROOT={target_path}",
        "-e",
        f"VERL_EVAL_COVERAGE_OUTPUT_DIR=/workspace/{raw_rel}",
        "-e",
        f"VERL_EVAL_COVERAGE_CHANGED_LINES_FILE=/workspace/{changed_rel}",
        "-e",
        f"PYTHONPATH=/workspace/trace_site:{target_path}:/workspace/generated_tests/tests:/workspace/generated_tests",
    ]
    for env in args.docker_env:
        cmd += ["-e", env]
    cmd += [
        "-v",
        f"{workspace}:/workspace",
        "-w",
        target_path,
        args.image,
        "timeout",
        str(args.test_timeout_sec),
        "bash",
        "-lc",
        " ".join(
            [
                "python",
                "/workspace/trace_runner.py",
                "/workspace/generated_tests",
                f"/workspace/{out_rel}",
                *[shlex.quote(target) for target in pytest_targets],
            ]
        ),
    ]
    print(f"running coverage trace against {label}: {target_path}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)

    trace_path = workspace / out_rel
    if trace_path.exists():
        payload = json.loads(trace_path.read_text())
    else:
        payload = {"exit_code": result.returncode, "counts": {}}
    payload["docker_exit_code"] = result.returncode
    payload["stdout"] = str(stdout_path)
    payload["stderr"] = str(stderr_path)
    return payload


def summarize_coverage(
    target_root: Path,
    changed: dict[str, set[int]],
    symbols: dict[str, dict[str, list[dict[str, object]]]],
    trace_payload: dict[str, object],
) -> dict[str, object]:
    raw_counts = trace_payload.get("counts")
    counts: dict[str, dict[int, int]] = {}
    if isinstance(raw_counts, dict):
        for rel, rel_counts in raw_counts.items():
            if isinstance(rel_counts, dict):
                counts[rel] = {int(line): int(count) for line, count in rel_counts.items()}

    files: dict[str, dict[str, object]] = {}
    total_changed = 0
    total_covered = 0
    for rel, changed_lines in sorted(changed.items()):
        covered = changed_lines.intersection(counts.get(rel, {}).keys())
        total_changed += len(changed_lines)
        total_covered += len(covered)
        files[rel] = {
            "changed_lines": len(changed_lines),
            "covered_changed_lines": len(covered),
            "line_coverage": (len(covered) / len(changed_lines)) if changed_lines else None,
            "uncovered_lines": sorted(changed_lines - covered)[:100],
        }

    function_total = 0
    function_covered = 0
    class_total = 0
    class_covered = 0
    covered_functions: list[str] = []
    covered_classes: list[str] = []
    uncovered_functions: list[str] = []
    uncovered_classes: list[str] = []
    for rel, file_symbols in symbols.items():
        rel_counts = counts.get(rel, {})
        for item in file_symbols.get("functions", []):
            function_total += 1
            body_lines = {int(line) for line in item.get("body_lines", [])}
            body_lines.discard(int(item["start"]))
            activated = bool(body_lines.intersection(rel_counts.keys()))
            if activated:
                function_covered += 1
                covered_functions.append(f"{rel}:{item['start']}:{item['name']}")
            else:
                uncovered_functions.append(f"{rel}:{item['start']}:{item['name']}")
        for item in file_symbols.get("classes", []):
            class_total += 1
            body_lines = {int(line) for line in item.get("body_lines", [])}
            body_lines.discard(int(item["start"]))
            activated = bool(body_lines.intersection(rel_counts.keys()))
            if activated:
                class_covered += 1
                covered_classes.append(f"{rel}:{item['start']}:{item['name']}")
            else:
                uncovered_classes.append(f"{rel}:{item['start']}:{item['name']}")

    return {
        "exit_code": trace_payload.get("exit_code"),
        "docker_exit_code": trace_payload.get("docker_exit_code"),
        "changed_files": len(changed),
        "changed_lines": total_changed,
        "covered_changed_lines": total_covered,
        "line_coverage": (total_covered / total_changed) if total_changed else None,
        "changed_functions": function_total,
        "covered_changed_functions": function_covered,
        "function_coverage": (function_covered / function_total) if function_total else None,
        "changed_classes": class_total,
        "covered_changed_classes": class_covered,
        "class_coverage": (class_covered / class_total) if class_total else None,
        "files": files,
        "covered_functions": covered_functions,
        "covered_classes": covered_classes,
        "uncovered_functions": uncovered_functions,
        "uncovered_classes": uncovered_classes,
        "target_root": str(target_root),
    }


def pct(value: object) -> str:
    if not isinstance(value, float):
        return "-"
    return f"{value * 100:.1f}%"


def write_report(run_dir: Path, summaries: dict[str, dict[str, object]]) -> None:
    lines = ["# Reference Coverage Diagnostic", ""]
    lines.append(
        "This report measures coverage of implementation lines that differ from the task's starting "
        "codebase, not whole-repository coverage. It is a diagnostic for whether evaluator tests "
        "exercise the behavior added by each reference implementation."
    )
    lines.append("")
    lines.append("| reference | pytest exit | changed files | changed lines | covered lines | line cov | functions | function cov | classes | class cov |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, summary in summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(summary.get("exit_code")),
                    str(summary.get("changed_files")),
                    str(summary.get("changed_lines")),
                    str(summary.get("covered_changed_lines")),
                    pct(summary.get("line_coverage")),
                    f"{summary.get('covered_changed_functions')}/{summary.get('changed_functions')}",
                    pct(summary.get("function_coverage")),
                    f"{summary.get('covered_changed_classes')}/{summary.get('changed_classes')}",
                    pct(summary.get("class_coverage")),
                ]
            )
            + " |"
        )

    for label, summary in summaries.items():
        lines.extend(["", f"## {label}", ""])
        files = summary.get("files") if isinstance(summary.get("files"), dict) else {}
        rows = sorted(
            files.items(),
            key=lambda item: (
                item[1].get("line_coverage") if isinstance(item[1].get("line_coverage"), float) else -1,
                item[1].get("changed_lines", 0),
            ),
        )
        lines.append("| file | changed lines | covered | line cov | first uncovered changed lines |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for rel, item in rows[:40]:
            uncovered = item.get("uncovered_lines") or []
            uncovered_text = ", ".join(str(line) for line in uncovered[:12])
            if len(uncovered) > 12:
                uncovered_text += ", ..."
            lines.append(
                f"| `{rel}` | {item.get('changed_lines')} | {item.get('covered_changed_lines')} | "
                f"{pct(item.get('line_coverage'))} | {uncovered_text} |"
            )

        uncovered_functions = summary.get("uncovered_functions") or []
        uncovered_classes = summary.get("uncovered_classes") or []
        if uncovered_functions:
            lines.extend(["", "Uncovered changed functions, first 40:", ""])
            lines.extend(f"- `{name}`" for name in uncovered_functions[:40])
        if uncovered_classes:
            lines.extend(["", "Uncovered changed classes, first 40:", ""])
            lines.extend(f"- `{name}`" for name in uncovered_classes[:40])

    (run_dir / "coverage_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose changed-line coverage of generated tests on reference targets.")
    parser.add_argument("--generated-tests", type=Path, required=True)
    parser.add_argument("--run-name", default="coverage-generated-tests")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--project-source", type=Path, default=DEFAULT_PROJECT_SOURCE)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=/path/to/reference",
        help="reference target to diagnose; repeatable. Defaults to official_reference and counter_reference.",
    )
    parser.add_argument("--no-default-references", action="store_true")
    parser.add_argument(
        "--include-path-regex",
        action="append",
        default=[],
        help=(
            "only include changed Python files whose relative path matches this regex; repeatable. "
            "By default all implementation Python diffs are included except tests/docs/examples."
        ),
    )
    parser.add_argument(
        "--exclude-path-regex",
        action="append",
        default=[],
        help="exclude changed Python files whose relative path matches this regex; repeatable.",
    )
    parser.add_argument(
        "--pytest-target",
        action="append",
        default=[],
        help=(
            "optional pytest file or node id under /workspace/generated_tests; repeatable. "
            "Defaults to the whole generated_tests directory."
        ),
    )
    parser.add_argument("--test-timeout-sec", type=int, default=300)
    parser.add_argument("--docker-gpus", default="")
    parser.add_argument("--docker-ipc", default="")
    parser.add_argument("--docker-shm-size", default="")
    parser.add_argument("--docker-env", action="append", default=[])
    parser.add_argument("--docker-volume", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.generated_tests.is_dir():
        raise SystemExit(f"generated tests directory does not exist: {args.generated_tests}")
    if not args.project_source.is_dir():
        raise SystemExit(f"project source does not exist: {args.project_source}")

    references: list[tuple[str, Path]] = []
    if not args.no_default_references:
        references.extend(default_references())
    seen = {label for label, _path in references}
    for spec in args.reference:
        label, path = parse_reference_spec(spec)
        if label in seen:
            raise SystemExit(f"duplicate reference label: {label}")
        references.append((label, path))
        seen.add(label)
    if not references:
        raise SystemExit("no reference targets configured")

    include_patterns = [re.compile(pattern) for pattern in args.include_path_regex]
    exclude_patterns = [re.compile(pattern) for pattern in args.exclude_path_regex]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_root / f"coverage-{slugify(args.run_name)}-{timestamp}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)

    print("preparing coverage diagnostic workspace:", workspace, flush=True)
    copy_clean_tree(args.project_source, workspace / "project")
    copy_clean_tree(args.generated_tests, workspace / "generated_tests")
    (workspace / "trace_runner.py").write_text(TRACE_RUNNER)
    trace_site = workspace / "trace_site"
    trace_site.mkdir(parents=True, exist_ok=True)
    (trace_site / "sitecustomize.py").write_text(SITE_CUSTOMIZE)

    target_root = workspace / "reference_targets"
    summaries: dict[str, dict[str, object]] = {}
    for label, source_dir in references:
        rel_target = f"reference_targets/{label}"
        target_dir = target_root / label
        copy_clean_tree(source_dir, target_dir)
        changed = build_changed_line_map(
            workspace / "project",
            target_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        symbols = collect_changed_symbols(target_dir, changed)
        trace_payload = run_trace_for_target(args, run_dir, workspace, label, rel_target, changed)
        summaries[label] = summarize_coverage(target_dir, changed, symbols, trace_payload)
        (run_dir / f"changed_lines_{label}.json").write_text(
            json.dumps({rel: sorted(lines) for rel, lines in changed.items()}, indent=2, sort_keys=True) + "\n"
        )

    payload = {
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "generated_tests": str(args.generated_tests),
        "project_source": str(args.project_source),
        "references": {label: str(path) for label, path in references},
        "include_path_regex": args.include_path_regex,
        "exclude_path_regex": args.exclude_path_regex,
        "pytest_target": args.pytest_target,
        "docker_gpus": args.docker_gpus,
        "docker_ipc": args.docker_ipc,
        "docker_shm_size": args.docker_shm_size,
        "docker_env": args.docker_env,
        "docker_volume": args.docker_volume,
        "summaries": summaries,
    }
    (run_dir / "coverage_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(run_dir, summaries)
    print(f"coverage diagnostic written to {run_dir}", flush=True)
    print((run_dir / "coverage_report.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
