#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATOR_PROMPT = REPO_ROOT / "docs" / "prompts" / "unit-test-generator-prompt-v3.md"
DEFAULT_GENERATOR_PROMPT_STAGE1 = REPO_ROOT / "docs" / "prompts" / "unit-test-generator-prompt-v4-stage1.md"
DEFAULT_GENERATOR_PROMPT_STAGE2 = REPO_ROOT / "docs" / "prompts" / "unit-test-generator-prompt-v4-stage2.md"
AGENT_PROMPT_PLACEHOLDER = "{{AGENT_PROMPT}}"
DEFAULT_RUNS_ROOT = Path("/mnt/data/projects/verl-vllm-testgen-runs")
DEFAULT_RUNTIME = Path("/mnt/data/projects/deep-swe/third_party/gemini_cli_runtime")
DEFAULT_IMAGE = "verl-vllm-rollout-benchmark:cpu-tests"
DEFAULT_REFERENCE_PROJECT = Path("/mnt/data/projects/verl-vllm-capability-benchmark/verl-v0.8.0")
DEFAULT_COUNTER_REFERENCE_PROJECT = Path(
    "/mnt/data/projects/verl-vllm-counterreference-oracle/workspace/alt_reference_project"
)
DEFAULT_CANDIDATE_RUNS_ROOT = Path("/mnt/data/projects/verl-vllm-gemini-runs")
DEFAULT_KEY_FILE = Path("/mnt/data/projects/verl-vllm-secrets/gemini_api_keys.txt")
ALLOWED_TOOLS = (
    "read_file",
    "read_many_files",
    "write_file",
    "grep_search",
    "glob",
    "list_directory",
    "replace",
    "run_shell_command",
)
FORBIDDEN_AUDIT_PATTERNS = (
    "__name__",
    "__module__",
    "__mro__",
    "__bases__",
    "__ray_actor_class__",
    "_underlying_class",
    "issubclass",
    "isinstance",
    "hasattr(",
    "_registry",
    "_REGISTRY",
    "patch(",
    "patch.object",
    "monkeypatch.setattr",
    ".called",
    "assert_called",
    "assert_any_call",
    "call_args_list",
    "call_count",
    "dummy_path",
    "fake_model_path",
    "hf_hub_download(",
    "mock_model_path",
    "snapshot_download(",
    "/tmp/test_model_path",
    "???",
    "ray.init(",
    "@ray.remote",
    "pytest.skip",
    "requests.get(",
    "requests.post(",
    "reference_project",
    "sys.modules.update",
    "sys_modules_dict",
    "sys.meta_path",
    "MetaPathFinder",
)
TEXT_AUDIT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
CONTRACT_DOC_NAMES = {
    "BEHAVIOR_INVENTORY.md",
    "COVERAGE_SUFFICIENCY_AUDIT.md",
    "CONTRACT_DECISION_TABLE.md",
    "HARNESS_STRATEGY.md",
    "IMPORT_COMPATIBILITY_AUDIT.md",
    "INFERRED_CONTRACTS.md",
    "OPTIONAL_DEPENDENCY_STUBS.md",
    "PROJECT_CONTRACT_SCAN.md",
    "PUBLIC_BEHAVIOR_TARGETS.md",
    "TASK_ANALYSIS.md",
    "PUBLIC_CONTRACTS.md",
    "COVERAGE_MAP.md",
    "TEST_PLAN.md",
    "TEST_CONTRACT_AUDIT.md",
    "FAILURE_MODE_AUDIT.md",
}
ALL_GENERATED_DOC_NAMES = CONTRACT_DOC_NAMES | {
    "README.md",
    "SELF_AUDIT.md",
}


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def image_exists(name: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


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


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-._")
    return value or "gemini-testgen"


def selected_api_key(env: dict[str, str], key_index: int, key_file: Path | None) -> str | None:
    if env.get("GEMINI_API_KEY"):
        return env["GEMINI_API_KEY"]
    keys = [k.strip() for k in env.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if keys:
        return keys[key_index % len(keys)]
    if key_file and key_file.exists():
        keys = [line.strip() for line in key_file.read_text().splitlines() if line.strip()]
        if keys:
            return keys[key_index % len(keys)]
    return env.get("GOOGLE_API_KEY")


def write_settings(
    home: Path,
    model: str,
    max_session_turns: int,
    shell_timeout_sec: int,
    gemini_thinking_level: str,
) -> str:
    active_model = model
    model_configs: dict[str, object] | None = None
    if gemini_thinking_level != "default":
        alias = f"evalgen-{slugify(model)}-thinking-{gemini_thinking_level.lower()}"
        active_model = alias
        model_configs = {
            "customAliases": {
                alias: {
                    "extends": "chat-base-3",
                    "modelConfig": {
                        "model": model,
                        "generateContentConfig": {
                            "thinkingConfig": {
                                "includeThoughts": True,
                                "thinkingLevel": gemini_thinking_level,
                            }
                        },
                    },
                }
            }
        }
    settings = {
        "general": {
            "maxSessionTurns": max_session_turns,
            "topicUpdateNarration": False,
            "enableNotifications": False,
        },
        "model": {
            "name": active_model,
            "compressionThreshold": 0.95,
        },
        "security": {
            "auth": {
                "selectedType": "gemini-api-key",
            },
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
        "ui": {
            "loadingPhrases": "off",
        },
    }
    if model_configs is not None:
        settings["modelConfigs"] = model_configs
    settings_dir = home / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return active_model


def render_generator_prompt(prompt_path: Path, agent_prompt_path: Path) -> str:
    generator_prompt = prompt_path.read_text()
    if AGENT_PROMPT_PLACEHOLDER in generator_prompt:
        generator_prompt = generator_prompt.replace(AGENT_PROMPT_PLACEHOLDER, agent_prompt_path.read_text())
    return generator_prompt


def write_direct_prompt_file(workspace: Path, prompt_name: str, content: str, saved_path: str | None = None) -> str:
    note = ""
    if saved_path is not None:
        note = f"\n\nFor reference, this exact prompt is also saved at `{saved_path}` inside the workspace.\n"
    (workspace / prompt_name).write_text(content.rstrip() + note)
    return prompt_name


def prepare_workspace(args: argparse.Namespace, run_dir: Path) -> Path:
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    copy_clean_tree(args.project_source, workspace / "project")
    copy_clean_tree(args.reference_project, workspace / "reference_project")
    shutil.copy2(args.agent_prompt, workspace / "AGENT_PROMPT.md")
    generator_prompt = render_generator_prompt(args.generator_prompt, args.agent_prompt)
    (workspace / "EVALUATOR_GENERATOR_PROMPT.md").write_text(generator_prompt)
    write_direct_prompt_file(
        workspace,
        ".gemini_prompt.txt",
        generator_prompt,
        saved_path="/workspace/EVALUATOR_GENERATOR_PROMPT.md",
    )
    return workspace


def write_stage_prompt(workspace: Path, stage: int) -> str:
    if stage == 1:
        content = """Follow the evaluator-generation prompt above and its Project-First Rule.

Stage 1 intentionally does not provide /workspace/reference_project. Use only:
- /workspace/project
- /workspace/AGENT_PROMPT.md
- /workspace/EVALUATOR_GENERATOR_PROMPT.md

Stage 1 only: create or revise these files under /workspace/generated_tests:
- PROJECT_CONTRACT_SCAN.md
- TASK_ANALYSIS.md
- PUBLIC_CONTRACTS.md
- INFERRED_CONTRACTS.md

PROJECT_CONTRACT_SCAN.md must list project/prompt evidence paths for every public or inferred contract.
Do not mention reference tests, reference implementation modules, or reference-only class/helper names in this stage.
Do not write REFERENCE_TEST_TRIAGE.md, NON_CONTRACTS.md, BEHAVIOR_INVENTORY.md, HARNESS_STRATEGY.md, COVERAGE_MAP.md, TEST_PLAN.md, pytest files, README.md, SELF_AUDIT.md, TEST_CONTRACT_AUDIT.md, FAILURE_MODE_AUDIT.md, or COVERAGE_SUFFICIENCY_AUDIT.md in this stage. Do not write a custom test runner.
"""
    elif stage == 2:
        content = """Follow the evaluator-generation prompt above and inspect the existing files under /workspace/generated_tests.

Stage 2 only: create or revise:
- REFERENCE_TEST_TRIAGE.md
- NON_CONTRACTS.md
- CONTRACT_DECISION_TABLE.md
- PUBLIC_BEHAVIOR_TARGETS.md
- OPTIONAL_DEPENDENCY_STUBS.md
- IMPORT_COMPATIBILITY_AUDIT.md

Use PROJECT_CONTRACT_SCAN.md, PUBLIC_CONTRACTS.md, and INFERRED_CONTRACTS.md as the contract boundary.
Inspect official tests, examples, configs, and public workflows in /workspace/reference_project.
For useful reference tests/examples, classify each as keep/adapt/discard and explain the public behavior intent.
If reference code introduces a name or module path not supported by the project-first scan, record it in NON_CONTRACTS.md unless you can add current-project evidence to INFERRED_CONTRACTS.md.
Names in NON_CONTRACTS.md are not future test targets. They must not become behavior IDs, pytest filenames, test names, imports, monkeypatch targets, or assertion subjects.
CONTRACT_DECISION_TABLE.md must classify every candidate item as BEHAVIOR_TARGET, PUBLIC_ENTRY_POINT, SUPPORTING_EVIDENCE, or NON_CONTRACT. A class, method, helper, module path, registry, or inheritance fact is not a BEHAVIOR_TARGET by itself; classify it as PUBLIC_ENTRY_POINT, SUPPORTING_EVIDENCE, or NON_CONTRACT.
PUBLIC_BEHAVIOR_TARGETS.md is the only list of direct test targets. Each row must describe an observable workflow with trigger/input, allowed public entry points, real project logic exercised, fake external boundary, observable output/state/error, reference evidence, expected incomplete-project failure, and CPU coverage priority or gap.
OPTIONAL_DEPENDENCY_STUBS.md must list optional external dependency root symbols, submodules, type-annotation attributes, and minimal behavior needed for tested paths.
IMPORT_COMPATIBILITY_AUDIT.md must list every external dependency root imported, stubbed, or monkeypatched while importing selected public project entry points, including absent optional packages and installed packages with optional/version-sensitive submodules. Include import statements on tested paths, transitive top-level imports of project modules imported by pytest, required root-level symbols, required submodules, deterministic behavior symbols, inert placeholder symbols, and how fake modules remain compatible with importlib.util.find_spec.
Use search tools to find import statements for each dependency root on the tested public entry-point import graph. Include submodule imports as well as root-level symbols; a root __getattr__ fallback does not replace registering required submodules in sys.modules.
Do not write BEHAVIOR_INVENTORY.md, HARNESS_STRATEGY.md, COVERAGE_MAP.md, TEST_PLAN.md, pytest files, README.md, SELF_AUDIT.md, TEST_CONTRACT_AUDIT.md, FAILURE_MODE_AUDIT.md, or COVERAGE_SUFFICIENCY_AUDIT.md in this stage. Do not write a custom test runner.
"""
    elif stage == 3:
        content = """Follow the evaluator-generation prompt above and inspect the existing files under /workspace/generated_tests.

Stage 3 only: create or revise:
- BEHAVIOR_INVENTORY.md
- HARNESS_STRATEGY.md
- COVERAGE_MAP.md

Stage 3 intentionally does not provide /workspace/reference_project. Use REFERENCE_TEST_TRIAGE.md for reference-derived behavior evidence.
Use PROJECT_CONTRACT_SCAN.md, REFERENCE_TEST_TRIAGE.md, INFERRED_CONTRACTS.md, CONTRACT_DECISION_TABLE.md, PUBLIC_BEHAVIOR_TARGETS.md, OPTIONAL_DEPENDENCY_STUBS.md, and IMPORT_COMPATIBILITY_AUDIT.md as input. For each important public behavior target, choose a public CPU harness or mark a justified gap.
Do not write TEST_PLAN.md, pytest files, README.md, SELF_AUDIT.md, TEST_CONTRACT_AUDIT.md, FAILURE_MODE_AUDIT.md, or COVERAGE_SUFFICIENCY_AUDIT.md in this stage. Do not write a custom test runner.
Focus on broad behavior inventory, harness strategy, and coverage mapping. Do not stop at selector/config smoke behavior when the task has downstream behavior.
Every BEHAVIOR_INVENTORY.md and COVERAGE_MAP.md row must describe an observable public workflow: trigger/input, real project logic exercised, external fake boundary, observable output/state/error, reference oracle, and expected baseline failure.
Do not create behavior rows whose main target is class inheritance, method presence, module-name matching, helper-function existence, or a NON_CONTRACTS.md item. Put such notes in TEST_CONTRACT_AUDIT.md later if needed.
Do not create behavior rows for PUBLIC_ENTRY_POINT, SUPPORTING_EVIDENCE, or NON_CONTRACT rows from CONTRACT_DECISION_TABLE.md unless they are attached to a behavior row in PUBLIC_BEHAVIOR_TARGETS.md.
If a row uses a class name, interface, selector, wrapper, or monkeypatch, state why that is a public boundary or external-boundary adapter for this task. Otherwise prefer downstream observable behavior.
Before marking a downstream behavior as a CPU gap, consider public-boundary CPU harnesses: fake server/client handles, synthetic public output objects, bounded local CPU runtime fixtures, downstream consumers, or monkeypatches at heavyweight external launch boundaries.
"""
    elif stage == 4:
        content = """Follow the evaluator-generation prompt above and inspect the existing files under /workspace/generated_tests.

Stage 4 only: revise the analysis files if needed, then create:
- TEST_PLAN.md
- COVERAGE_SUFFICIENCY_AUDIT.md
- TEST_CONTRACT_AUDIT.md
- FAILURE_MODE_AUDIT.md

Stage 4 intentionally does not provide /workspace/reference_project. Use REFERENCE_TEST_TRIAGE.md, CONTRACT_DECISION_TABLE.md, PUBLIC_BEHAVIOR_TARGETS.md, OPTIONAL_DEPENDENCY_STUBS.md, IMPORT_COMPATIBILITY_AUDIT.md, and the existing analysis artifacts, not reference implementation files.
Do not write pytest files, README.md, or SELF_AUDIT.md in this stage. Do not write a custom test runner.
Every planned test must trace to PUBLIC_BEHAVIOR_TARGETS.md plus REFERENCE_TEST_TRIAGE.md when applicable, BEHAVIOR_INVENTORY.md, HARNESS_STRATEGY.md, and COVERAGE_MAP.md. Selector tests do not count as coverage for downstream generation/update/cache/lifecycle behavior.
If a planned test uses a name that is not directly visible in /workspace/project or /workspace/AGENT_PROMPT.md, it must trace to INFERRED_CONTRACTS.md and that row must cite current-project evidence, not only reference-project evidence.
Do not plan tests whose primary target is listed in NON_CONTRACTS.md. Do not plan tests whose primary oracle is class inheritance, method presence, module-name matching, helper-function existence, or "raises NotImplementedError" unless a real project caller observes that outcome in a public workflow.
Do not plan tests for PUBLIC_ENTRY_POINT, SUPPORTING_EVIDENCE, or NON_CONTRACT rows from CONTRACT_DECISION_TABLE.md. PUBLIC_ENTRY_POINT rows may be used only to reach a behavior target from PUBLIC_BEHAVIOR_TARGETS.md.
For any planned test whose oracle is structural-looking, import-based, construction-based, mocked-call-based, or exact-error-text-based, write the public-contract reason in TEST_CONTRACT_AUDIT.md. If there is no such reason, rewrite the test toward returned data, public state, emitted records, or downstream consumer behavior.
Do not plan a test whose only oracle is issubclass, isinstance, hasattr, class __name__, import success, or module-path equality. If one of these is needed at the boundary, pair it with a downstream observable behavior in the same test.
Before planning direct calls to project constructors, dataclasses, config objects, or functions, inspect their actual signatures and current-project call sites. Do not guess keyword arguments from reference code, nearby backends, or task wording. If the signature is complex, use an existing project factory/example or a public workflow that constructs it.
Do not plan pytest imports from modules that exist only in /workspace/reference_project unless the exact dotted path appears in /workspace/project or /workspace/AGENT_PROMPT.md as an import, registry target, config target, or public extension point. This is invalid even if an official reference test imports that helper; adapt through a public project caller or mark a gap.
If the plan only tests configuration or activation while the inventory has downstream behaviors that can be exercised with public-boundary CPU fakes, expand the plan.
"""
    elif stage == 5:
        content = """Follow the evaluator-generation prompt above and inspect all existing files under /workspace/generated_tests.

Stage 5 only: implement the evaluator from the accepted plan:
- one or more pytest files
- README.md
- SELF_AUDIT.md

Stage 5 intentionally does not provide /workspace/reference_project. Implement from TEST_PLAN.md, HARNESS_STRATEGY.md, REFERENCE_TEST_TRIAGE.md, PUBLIC_BEHAVIOR_TARGETS.md, CONTRACT_DECISION_TABLE.md, OPTIONAL_DEPENDENCY_STUBS.md, and IMPORT_COMPATIBILITY_AUDIT.md.
Do not write a custom test runner or shell entrypoint. The benchmark harness will run your pytest files directly with the target project and generated test directory on PYTHONPATH.
Do not add new public contracts or new reference-only requirements at this stage. If a planned test cannot be implemented without quarantined names, heavyweight model/tokenizer loading, heavyweight/GPU runtime, or mock-call-count oracles, remove that test and mark the behavior as a CPU/GPU/integration gap in the audits.
Bounded local CPU runtime fixtures and narrow optional-dependency stubs are allowed when they exercise real public project controllers or consumers.
Ensure tests can collect under both /workspace/project and /workspace/reference_project. If you stub optional packages, provide import-compatible submodules and attributes used by project imports and type annotations.
Before writing pytest stubs, copy every required root symbol, submodule, type-annotation attribute, and minimal behavior listed in OPTIONAL_DEPENDENCY_STUBS.md and IMPORT_COMPATIBILITY_AUDIT.md. For example, if the tested path imports `from dependency import Symbol`, the fake root module must define `Symbol`. If creating a fake module in sys.modules, set __spec__; if it is a package, set __path__. If the import audit is incomplete, add a module-level __getattr__ fallback for inert import-time symbols while keeping assertion-relevant behavior explicit and deterministic.
Register every required submodule and parent package from IMPORT_COMPATIBILITY_AUDIT.md in sys.modules. A root __getattr__ fallback does not satisfy `import dependency.submodule` or `from dependency.submodule import Symbol`.
Avoid top-level imports of missing capability implementation modules unless the exact module path is supported by PROJECT_CONTRACT_SCAN.md or INFERRED_CONTRACTS.md.
Do not import from a reference-only module in pytest code unless the exact dotted path appears in /workspace/project or /workspace/AGENT_PROMPT.md as an import, registry target, config target, or public extension point. This is a fatal evaluator design error even when the reference passes. Adapt official reference helper tests through public project callers instead, or remove the test and mark the behavior as a gap.
Do not call project constructors or functions with guessed keyword arguments. Use signatures and current-project call sites already inspected in TEST_CONTRACT_AUDIT.md, or rewrite the test through an existing public workflow.
Do not use placeholder model paths such as /tmp/test_model_path, dummy_path, mock_model_path, fake_model_path, or ??? unless the test creates the required local files first. Prefer minimal plain objects for model config metadata when real HF loading is irrelevant.
Do not implement pytest tests for NON_CONTRACTS.md items. If a planned test depends on one, remove that test or rewrite it through a public caller behavior.
Before final output, scan pytest code for structural-looking checks, reference names, fake class names, dynamic module lookups, and monkeypatch targets. Keep only the ones justified by PUBLIC_CONTRACTS.md, INFERRED_CONTRACTS.md, HARNESS_STRATEGY.md, or TEST_CONTRACT_AUDIT.md, and never leave a structural check as the only oracle in a test.
"""
    else:
        raise ValueError(f"unknown generation stage: {stage}")
    base_prompt = (workspace / "EVALUATOR_GENERATOR_PROMPT.md").read_text()
    content = (
        base_prompt.rstrip()
        + "\n\n---\n\n"
        + f"Stage {stage} instructions for this run:\n\n"
        + content.rstrip()
    )
    prompt_name = f".gemini_stage_{stage}_prompt.txt"
    return write_direct_prompt_file(
        workspace,
        prompt_name,
        content,
        saved_path="/workspace/EVALUATOR_GENERATOR_PROMPT.md",
    )


def move_workspace_path(workspace: Path, run_dir: Path, name: str, label: str) -> tuple[Path, Path] | None:
    src = workspace / name
    if not src.exists():
        return None
    dst = run_dir / f".hidden_{label}_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return src, dst


def restore_workspace_path(moved: tuple[Path, Path] | None) -> None:
    if moved is None:
        return
    src, dst = moved
    if src.exists():
        shutil.rmtree(src)
    shutil.move(str(dst), str(src))


def run_staged_gemini(args: argparse.Namespace, run_dir: Path, workspace: Path) -> None:
    for stage in (1, 2, 3, 4, 5):
        prompt_name = write_stage_prompt(workspace, stage)
        moved_reference = None
        if stage != 2:
            moved_reference = move_workspace_path(workspace, run_dir, "reference_project", f"stage{stage}")
        try:
            run_gemini(
                args,
                run_dir,
                workspace,
                prompt_file=prompt_name,
                log_prefix=f"gemini_stage{stage}",
            )
        finally:
            restore_workspace_path(moved_reference)
        findings = stage_audit_findings(
            run_dir,
            workspace,
            stage=stage,
            filename=f"stage_audit_stage{stage}.txt",
        )
        if findings:
            print(
                f"stage {stage} audit findings: " + " | ".join(findings[:8]),
                flush=True,
            )


def write_two_stage_prompt_files(args: argparse.Namespace, workspace: Path) -> tuple[str, str]:
    stage1_prompt = render_generator_prompt(args.generator_prompt_stage1, args.agent_prompt)
    stage2_prompt = render_generator_prompt(args.generator_prompt_stage2, args.agent_prompt)
    (workspace / "EVALUATOR_GENERATOR_PROMPT_STAGE1.md").write_text(stage1_prompt)
    (workspace / "EVALUATOR_GENERATOR_PROMPT_STAGE2.md").write_text(stage2_prompt)

    stage1_prompt_name = ".gemini_two_stage1_prompt.txt"
    stage2_prompt_name = ".gemini_two_stage2_prompt.txt"
    write_direct_prompt_file(
        workspace,
        stage1_prompt_name,
        stage1_prompt,
        saved_path="/workspace/EVALUATOR_GENERATOR_PROMPT_STAGE1.md",
    )
    write_direct_prompt_file(
        workspace,
        stage2_prompt_name,
        stage2_prompt,
        saved_path="/workspace/EVALUATOR_GENERATOR_PROMPT_STAGE2.md",
    )
    return stage1_prompt_name, stage2_prompt_name


def run_two_stage_gemini(args: argparse.Namespace, run_dir: Path, workspace: Path) -> None:
    stage1_prompt_name, stage2_prompt_name = write_two_stage_prompt_files(args, workspace)
    two_stage_session_id = str(uuid.uuid4())
    (run_dir / "two_stage_session_id.txt").write_text(two_stage_session_id + "\n")
    active_model, policy_path, env, key = prepare_gemini_runtime(args, workspace)
    container_name = persistent_container_name(run_dir)
    moved_reference = None
    try:
        moved_reference = move_workspace_path(workspace, run_dir, "reference_project", "two_stage1")
        start_gemini_container(
            args,
            workspace=workspace,
            env=env,
            active_model=active_model,
            container_name=container_name,
        )
        exec_gemini_in_container(
            args,
            run_dir,
            workspace,
            container_name=container_name,
            policy_path=policy_path,
            prompt_file=stage1_prompt_name,
            log_prefix="gemini_twostage1",
            session_id=two_stage_session_id,
        )
        restore_workspace_path(moved_reference)
        moved_reference = None

        exec_gemini_in_container(
            args,
            run_dir,
            workspace,
            container_name=container_name,
            policy_path=policy_path,
            prompt_file=stage2_prompt_name,
            log_prefix="gemini_twostage2",
            resume_session=two_stage_session_id,
        )
    finally:
        restore_workspace_path(moved_reference)
        stop_gemini_container(container_name)
        cleanup_sensitive_state(run_dir, workspace, key)


PLAN_AUDIT_DOCS = {
    "BEHAVIOR_INVENTORY.md",
    "CONTRACT_DECISION_TABLE.md",
    "COVERAGE_MAP.md",
    "TEST_PLAN.md",
    "COVERAGE_SUFFICIENCY_AUDIT.md",
    "HARNESS_STRATEGY.md",
    "IMPORT_COMPATIBILITY_AUDIT.md",
    "OPTIONAL_DEPENDENCY_STUBS.md",
    "PROJECT_CONTRACT_SCAN.md",
    "PUBLIC_BEHAVIOR_TARGETS.md",
    "TEST_CONTRACT_AUDIT.md",
    "FAILURE_MODE_AUDIT.md",
}

PLAN_SEMANTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "subclass/interface oracle",
        re.compile(r"\b(subclass(?:es|ing)?|inherits?\s+from|inheriting\s+from)\b", re.IGNORECASE),
    ),
    (
        "interface-conformance oracle",
        re.compile(r"\b(conforms?\s+to|interface compliance|base class compliance)\b", re.IGNORECASE),
    ),
    (
        "mock-call oracle",
        re.compile(
            r"\b(assert|verify|checking|check)\b[^\n]*(\bcalled\b|call_count|call_args|assert_called)",
            re.IGNORECASE,
        ),
    ),
    (
        "implementation-specific exact error text",
        re.compile(r"\b(exact|assert)[^\n]*(error-message|error message|exception text)\b", re.IGNORECASE),
    ),
)


def plan_semantic_audit(
    run_dir: Path,
    workspace: Path,
    *,
    filename: str = "plan_semantic_audit.txt",
) -> dict[str, list[str]]:
    generated_tests = workspace / "generated_tests"
    hits: dict[str, list[str]] = {}

    for rel_name in sorted(PLAN_AUDIT_DOCS):
        path = generated_tests / rel_name
        if not path.exists() or not path.is_file():
            continue
        rel_hits: list[str] = []
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if line_is_non_contract_context(line):
                continue
            for label, pattern in PLAN_SEMANTIC_PATTERNS:
                if pattern.search(line):
                    rel_hits.append(f"line {lineno}: {label}: {line.strip()[:220]}")
        if rel_hits:
            hits[rel_name] = rel_hits

    lines: list[str] = []
    if hits:
        for rel, rel_hits in sorted(hits.items()):
            lines.append(f"{rel}:")
            lines.extend(f"  - {hit}" for hit in rel_hits)
    else:
        lines.append("none")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return hits


def stage_audit_findings(
    run_dir: Path,
    workspace: Path,
    *,
    stage: int,
    filename: str,
) -> list[str]:
    findings: list[str] = []
    tool_hits = tool_usage_audit(run_dir, filename=f"tool_usage_{filename}")
    plan_hits = plan_semantic_audit(run_dir, workspace, filename=f"plan_semantic_{filename}")
    reference_doc_hits = reference_only_contract_doc_audit(
        run_dir,
        workspace,
        filename=f"reference_only_contract_docs_{filename}",
    )
    reference_symbol_hits = reference_only_symbol_audit(
        run_dir,
        workspace,
        filename=f"reference_only_symbols_{filename}",
    )
    non_contract_hits = non_contract_reuse_audit(
        run_dir,
        workspace,
        filename=f"non_contract_reuse_{filename}",
    )
    static_hits = static_audit(run_dir, workspace, filename=f"static_{filename}")

    if tool_hits:
        for rel, hits in sorted(tool_hits.items()):
            findings.append(f"{rel}: forbidden tool use " + "; ".join(hits[:2]))
    for rel, hits in sorted(plan_hits.items()):
        findings.append(f"{rel}: " + "; ".join(hits[:3]))
    for rel, hits in sorted(reference_doc_hits.items()):
        findings.append(f"{rel}: " + "; ".join(hits[:3]))
    for rel, hits in sorted(reference_symbol_hits.items()):
        findings.append(f"{rel}: " + "; ".join(hits[:3]))
    for rel, hits in sorted(non_contract_hits.items()):
        findings.append(f"{rel}: " + "; ".join(hits[:3]))

    if stage == 5:
        stage3_patterns = {
            "__name__",
            "__module__",
            "__mro__",
            "__bases__",
            "_underlying_class",
            "issubclass",
            "isinstance",
            "hasattr(",
            "pytest.skip",
            ".called",
            "assert_called",
            "assert_any_call",
            "call_args_list",
            "call_count",
            "patch.object",
            "sys.modules.update",
            "sys_modules_dict",
            "sys.meta_path",
            "MetaPathFinder",
        }
        for pattern in sorted(stage3_patterns):
            paths = static_hits.get(pattern, [])
            flagged = [path for path in paths if Path(path).name.startswith("test_") and path.endswith(".py")]
            if flagged:
                findings.append(f"{pattern}: {', '.join(flagged)}")

    (run_dir / filename).write_text("\n".join(findings or ["none"]) + "\n")
    return findings


def write_gemini_policy(workspace: Path) -> Path:
    allowed_tools = ", ".join(f'"{tool}"' for tool in ALLOWED_TOOLS)
    policy = f"""# Generated by run_gemini_test_generator.py.
# The evaluator generator may inspect project files, write generated_tests, and
# run local shell commands in the prepared Docker workspace. It must run its
# generated tests against both /workspace/reference_project and /workspace/project
# before finishing.

[[rule]]
toolName = [{allowed_tools}]
decision = "allow"
priority = 900

[[rule]]
toolName = "*"
decision = "deny"
priority = 100
denyMessage = "Evaluator generation is limited to local read/search/write and shell tools in the prepared workspace. Web, package installs, model downloads, and unknown tools are not available."
"""
    path = workspace / ".gemini_policy.toml"
    path.write_text(policy)
    return path


def gemini_session_args(session_id: str | None, resume_session: str | None) -> str:
    if session_id is not None and resume_session is not None:
        raise ValueError("session_id and resume_session are mutually exclusive")
    args = ""
    if session_id is not None:
        args += f"--session-id {session_id} "
    if resume_session is not None:
        args += f"--resume {resume_session} "
    return args


def gemini_shell_command(
    *,
    policy_path: Path,
    workspace: Path,
    prompt_file: str,
    session_id: str | None = None,
    resume_session: str | None = None,
) -> str:
    return (
        "/opt/gemini_cli_runtime/bin/node "
        "/opt/gemini_cli_runtime/bundle/gemini.js "
        '--model "$GEMINI_CLI_MODEL" '
        "--approval-mode yolo "
        f"--admin-policy {policy_path.relative_to(workspace)} "
        "--include-directories /workspace "
        "--output-format stream-json "
        + gemini_session_args(session_id, resume_session)
        + '--prompt "" '
        f"< /workspace/{prompt_file}"
    )


def gemini_docker_env_args(active_model: str) -> list[str]:
    return [
        "-e",
        "HOME=/workspace/.gemini_home",
        "-e",
        "GEMINI_API_KEY",
        "-e",
        "GOOGLE_API_KEY",
        "-e",
        f"GEMINI_CLI_MODEL={active_model}",
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
    ]


def prepare_gemini_runtime(args: argparse.Namespace, workspace: Path) -> tuple[str, Path, dict[str, str], str]:
    node = args.gemini_runtime / "bin" / "node"
    gemini_js = args.gemini_runtime / "bundle" / "gemini.js"
    if not node.exists() or not gemini_js.exists():
        raise SystemExit(f"Gemini CLI runtime is incomplete: {args.gemini_runtime}")
    if not image_exists(args.image):
        raise SystemExit(f"Docker image not found: {args.image}")

    gemini_home = workspace / ".gemini_home"
    active_model = write_settings(
        gemini_home,
        args.model,
        args.max_session_turns,
        args.shell_timeout_sec,
        args.gemini_thinking_level,
    )
    policy_path = write_gemini_policy(workspace)

    env = os.environ.copy()
    key = selected_api_key(env, args.key_index, args.key_file)
    if not key:
        raise SystemExit("set GEMINI_API_KEY, GEMINI_API_KEYS, or GOOGLE_API_KEY before running Gemini")
    env["GEMINI_API_KEY"] = key
    env["GOOGLE_API_KEY"] = key
    return active_model, policy_path, env, key


def persistent_container_name(run_dir: Path) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_dir.name).strip("-._")
    return f"gemini-testgen-{safe_name[:80]}-{uuid.uuid4().hex[:8]}"


def start_gemini_container(
    args: argparse.Namespace,
    *,
    workspace: Path,
    env: dict[str, str],
    active_model: str,
    container_name: str,
) -> None:
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
        *gemini_docker_env_args(active_model),
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{args.gemini_runtime}:/opt/gemini_cli_runtime:ro",
        "-w",
        "/workspace",
        args.image,
        "bash",
        "-lc",
        "trap 'exit 0' TERM INT; while true; do sleep 3600 & wait $!; done",
    ]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


def stop_gemini_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def exec_gemini_in_container(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    *,
    container_name: str,
    policy_path: Path,
    prompt_file: str,
    log_prefix: str,
    session_id: str | None = None,
    resume_session: str | None = None,
) -> None:
    cmd = [
        "docker",
        "exec",
        "-i",
        container_name,
        "bash",
        "-lc",
        gemini_shell_command(
            policy_path=policy_path,
            workspace=workspace,
            prompt_file=prompt_file,
            session_id=session_id,
            resume_session=resume_session,
        ),
    ]

    stdout_path = run_dir / f"{log_prefix}.stream.jsonl"
    stderr_path = run_dir / f"{log_prefix}.stderr.txt"
    print(f"running Gemini CLI in persistent container {container_name}; logs: {run_dir}", flush=True)
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
    assert_gemini_stream_success(stdout_path)
    print(f"Gemini CLI finished in {int(time.time() - started)}s", flush=True)


def run_gemini(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    *,
    prompt_file: str = ".gemini_prompt.txt",
    log_prefix: str = "gemini_cli",
    session_id: str | None = None,
    resume_session: str | None = None,
    cleanup_after: bool = True,
) -> None:
    active_model, policy_path, env, key = prepare_gemini_runtime(args, workspace)
    uid = str(os.getuid())
    gid = str(os.getgid())

    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        *gemini_docker_env_args(active_model),
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{args.gemini_runtime}:/opt/gemini_cli_runtime:ro",
        "-w",
        "/workspace",
        args.image,
        "bash",
        "-lc",
        gemini_shell_command(
            policy_path=policy_path,
            workspace=workspace,
            prompt_file=prompt_file,
            session_id=session_id,
            resume_session=resume_session,
        ),
    ]

    stdout_path = run_dir / f"{log_prefix}.stream.jsonl"
    stderr_path = run_dir / f"{log_prefix}.stderr.txt"
    print(f"running Gemini CLI in {args.image}; logs: {run_dir}", flush=True)
    started = time.time()
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            subprocess.run(
                cmd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=True,
                timeout=None if args.timeout_sec == 0 else args.timeout_sec,
            )
        assert_gemini_stream_success(stdout_path)
        print(f"Gemini CLI finished in {int(time.time() - started)}s", flush=True)
    finally:
        if cleanup_after:
            cleanup_sensitive_state(run_dir, workspace, key)
        else:
            redact_sensitive_state(run_dir, key)


def assert_gemini_stream_success(stdout_path: Path) -> None:
    errors: list[str] = []
    final_status: str | None = None
    if not stdout_path.exists():
        return
    for line in stdout_path.read_text(errors="ignore").splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "error":
            message = event.get("message") or event.get("error") or "unknown Gemini stream error"
            errors.append(str(message))
        elif event_type == "result":
            status = event.get("status")
            if isinstance(status, str):
                final_status = status
    if errors or final_status == "error":
        detail = "; ".join(errors[-3:]) if errors else f"final status: {final_status}"
        raise RuntimeError(f"Gemini stream reported failure in {stdout_path}: {detail}")


def redact_sensitive_state(run_dir: Path, key: str) -> None:
    replacements = {
        key: "[REDACTED_GEMINI_API_KEY]",
    }
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        redacted = text
        for secret, marker in replacements.items():
            if secret:
                redacted = redacted.replace(secret, marker)
        if redacted != text:
            path.write_text(redacted)


def cleanup_sensitive_state(run_dir: Path, workspace: Path, key: str) -> None:
    gemini_home = workspace / ".gemini_home"
    if gemini_home.exists():
        shutil.rmtree(gemini_home)
    redact_sensitive_state(run_dir, key)


def parse_pytest_summary(stdout_path: Path) -> dict[str, int]:
    if not stdout_path.exists():
        return {}
    text = stdout_path.read_text(errors="ignore")
    summary: dict[str, int] = {}

    for line in reversed(text.splitlines()):
        if " in " not in line:
            continue
        if not re.search(r"\b(passed|failed|errors?|skipped|xfailed|xpassed)\b", line):
            continue
        for count, key in re.findall(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b", line):
            normalized = "errors" if key in {"error", "errors"} else key
            summary[normalized] = summary.get(normalized, 0) + int(count)
        if summary:
            return summary

    for key in ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed"):
        total = 0
        for match in re.finditer(rf"(\d+)\s+{key}\b", text):
            total += int(match.group(1))
        if total:
            normalized = "errors" if key == "error" else key
            summary[normalized] = summary.get(normalized, 0) + total
    return summary


def count_generated_tests(workspace: Path) -> int:
    tests_dir = workspace / "generated_tests"
    if not tests_dir.exists():
        return 0
    count = 0
    for path in tests_dir.rglob("test_*.py"):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        text = path.read_text(errors="ignore")
        count += len(re.findall(r"^\s*(?:async\s+def|def)\s+test_", text, flags=re.MULTILINE))
    return count


def run_generated_tests(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    target_label: str,
    project_under_test: str,
    *,
    log_prefix: str = "generated",
) -> dict[str, object]:
    generated_tests = workspace / "generated_tests"
    if not generated_tests.exists():
        print("generated_tests directory was not produced; skipping validation", flush=True)
        return {"exit_code": 125, "summary": {}}

    uid = str(os.getuid())
    gid = str(os.getgid())
    target_path = f"/workspace/{project_under_test}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/workspace/.test_home",
        "-e",
        "USER=tester",
        "-e",
        "LOGNAME=tester",
        "-e",
        "PYTEST_ADDOPTS=-p no:cacheprovider",
        "-e",
        f"PROJECT_UNDER_TEST={target_path}",
        "-e",
        f"PYTHONPATH={target_path}:/workspace/generated_tests",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        args.image,
        "timeout",
        str(args.test_timeout_sec),
        "bash",
        "-lc",
        "python -m pytest -q /workspace/generated_tests",
    ]

    safe_label = slugify(target_label)
    stdout_path = run_dir / f"{log_prefix}_{safe_label}.stdout.txt"
    stderr_path = run_dir / f"{log_prefix}_{safe_label}.stderr.txt"
    print(f"running generated pytest tests against {target_path}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
    print(f"{target_label} generated tests exited with {result.returncode}", flush=True)
    return {
        "exit_code": result.returncode,
        "summary": parse_pytest_summary(stdout_path),
    }


def run_manual_suite(
    args: argparse.Namespace,
    run_dir: Path,
    target_label: str,
    source_dir: Path,
) -> dict[str, object]:
    safe_label = slugify(target_label)
    stdout_path = run_dir / f"manual_{safe_label}.stdout.txt"
    stderr_path = run_dir / f"manual_{safe_label}.stderr.txt"
    env = os.environ.copy()
    env["TEST_IMAGE"] = args.image
    cmd = [str(REPO_ROOT / "eval_tests" / "run_cpu_tests.sh"), str(source_dir)]
    print(f"running hand-written evaluator against {source_dir}", flush=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=args.test_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "summary": parse_pytest_summary(stdout_path),
                "error": "timeout",
            }
    print(f"{target_label} hand-written tests exited with {result.returncode}", flush=True)
    return {
        "exit_code": result.returncode,
        "summary": parse_pytest_summary(stdout_path),
    }


def iter_auditable_generated_files(tests_dir: Path):
    for path in tests_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix == ".pyc" or path.suffix not in TEXT_AUDIT_SUFFIXES:
            continue
        if path.stat().st_size > 5_000_000:
            continue
        yield path


def iter_generated_test_python_files(generated_tests: Path):
    if not generated_tests.exists():
        return
    for path in generated_tests.rglob("test_*.py"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.stat().st_size > 5_000_000:
            continue
        yield path


def audit_pattern_present(pattern: str, text: str) -> bool:
    if pattern == "_registry":
        return re.search(r"(?<![A-Za-z0-9])_registry\b", text) is not None
    if pattern == "_REGISTRY":
        return re.search(r"(?<![A-Za-z0-9])_REGISTRY\b", text) is not None
    return pattern in text


def static_audit(run_dir: Path, workspace: Path, *, filename: str = "static_audit.txt") -> dict[str, list[str]]:
    tests_dir = workspace / "generated_tests"
    lines: list[str] = []
    audit: dict[str, list[str]] = {}
    for pattern in FORBIDDEN_AUDIT_PATTERNS:
        hits: list[str] = []
        for path in iter_auditable_generated_files(tests_dir):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if audit_pattern_present(pattern, text):
                rel = path.relative_to(tests_dir)
                hits.append(str(rel))
        unique_hits = sorted(set(hits))
        audit[pattern] = unique_hits
        lines.append(f"{pattern}: {', '.join(unique_hits) if unique_hits else 'none'}")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return audit


def module_path_exists(root: Path, module: str) -> bool:
    if not module:
        return False
    rel = Path(*module.split("."))
    return (root / rel.with_suffix(".py")).is_file() or (root / rel / "__init__.py").is_file()


def find_known_module_prefix(project_root: Path, reference_root: Path, dotted: str) -> tuple[str, bool, bool] | None:
    cleaned = dotted.strip(".")
    if not cleaned or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", cleaned):
        return None
    parts = cleaned.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        in_project = module_path_exists(project_root, prefix)
        in_reference = module_path_exists(reference_root, prefix)
        if in_project or in_reference:
            return prefix, in_project, in_reference
    return None


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def dotted_candidates(value: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", value)


def extract_test_dotted_references(path: Path, text: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    refs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(("import", alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                refs.append(("from-import", node.module))
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name.endswith("patch") or name.endswith("patch.object") or name.endswith("setattr"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    for candidate in dotted_candidates(node.args[0].value):
                        refs.append(("string-patch", candidate))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for candidate in dotted_candidates(node.value):
                refs.append(("string", candidate))
    return refs


def reference_only_dotted_path_audit(
    run_dir: Path,
    workspace: Path,
    *,
    filename: str = "reference_only_dotted_paths.txt",
) -> dict[str, list[str]]:
    generated_tests = workspace / "generated_tests"
    project_root = workspace / "project"
    reference_root = workspace / "reference_project"
    hits: dict[str, list[str]] = {}

    for path in iter_generated_test_python_files(generated_tests):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(generated_tests)
        rel_hits: list[str] = []
        for kind, dotted in extract_test_dotted_references(path, text):
            prefix_info = find_known_module_prefix(project_root, reference_root, dotted)
            if prefix_info is None:
                continue
            prefix, in_project, in_reference = prefix_info
            if in_reference and not in_project:
                rel_hits.append(f"{kind} `{dotted}` resolves through reference-only module `{prefix}`")
        if rel_hits:
            hits[str(rel)] = sorted(set(rel_hits))

    lines: list[str] = []
    if hits:
        for rel, rel_hits in sorted(hits.items()):
            lines.append(f"{rel}:")
            lines.extend(f"  - {hit}" for hit in rel_hits)
    else:
        lines.append("none")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return hits


def reference_only_contract_doc_audit(
    run_dir: Path,
    workspace: Path,
    *,
    filename: str = "reference_only_contract_docs.txt",
) -> dict[str, list[str]]:
    generated_tests = workspace / "generated_tests"
    project_root = workspace / "project"
    reference_root = workspace / "reference_project"
    audited_names = {
        "TASK_ANALYSIS.md",
        "CONTRACT_DECISION_TABLE.md",
        "IMPORT_COMPATIBILITY_AUDIT.md",
        "OPTIONAL_DEPENDENCY_STUBS.md",
        "PROJECT_CONTRACT_SCAN.md",
        "PUBLIC_BEHAVIOR_TARGETS.md",
        "PUBLIC_CONTRACTS.md",
        "INFERRED_CONTRACTS.md",
        "HARNESS_STRATEGY.md",
        "COVERAGE_MAP.md",
        "TEST_PLAN.md",
        "TEST_CONTRACT_AUDIT.md",
        "FAILURE_MODE_AUDIT.md",
        "README.md",
        "SELF_AUDIT.md",
    }
    hits: dict[str, list[str]] = {}

    for rel_name in sorted(audited_names):
        path = generated_tests / rel_name
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        rel_hits: list[str] = []
        for dotted in dotted_candidates(text):
            prefix_info = find_known_module_prefix(project_root, reference_root, dotted)
            if prefix_info is None:
                continue
            prefix, in_project, in_reference = prefix_info
            if in_reference and not in_project:
                rel_hits.append(f"`{dotted}` resolves through reference-only module `{prefix}`")
        if rel_hits:
            hits[rel_name] = sorted(set(rel_hits))

    lines: list[str] = []
    if hits:
        for rel, rel_hits in sorted(hits.items()):
            lines.append(f"{rel}:")
            lines.extend(f"  - {hit}" for hit in rel_hits)
    else:
        lines.append("none")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return hits


def whole_word_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")


def is_likely_symbol(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z_]\w*", value) is not None and len(value) >= 3


def normalize_symbol_reference(raw: str) -> str | None:
    value = raw.strip()
    if not value or "/" in value or "://" in value:
        return None
    value = value.split("(", 1)[0].strip()
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    value = value.strip("()[]{}:,")
    if not is_likely_symbol(value):
        return None
    if value.islower() and "_" not in value and not value.startswith("_"):
        return None
    return value


def extract_python_defined_symbols(root: Path) -> set[str]:
    symbols: set[str] = set()
    if not root.exists():
        return symbols
    for path in root.rglob("*.py"):
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if is_likely_symbol(node.name):
                    symbols.add(node.name)
    return symbols


def extract_inferred_contract_symbols(workspace: Path) -> dict[str, str]:
    generated_tests = workspace / "generated_tests"
    path = generated_tests / "INFERRED_CONTRACTS.md"
    if not path.exists() or not path.is_file():
        return {}

    symbols: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        if line_is_non_contract_context(line):
            continue
        has_current_project_evidence = (
            "/workspace/project" in line or "/workspace/AGENT_PROMPT.md" in line
        )
        if not has_current_project_evidence:
            continue
        for raw in re.findall(r"`([^`]+)`", line):
            value = normalize_symbol_reference(raw)
            if value is None:
                continue
            symbols.setdefault(value, f"inferred contract in INFERRED_CONTRACTS.md line {lineno}")
    return symbols


def line_is_non_contract_context(line: str) -> bool:
    lowered = line.lower()
    allowed_terms = (
        "non-contract",
        "non contract",
        "reference-only",
        "reference only",
        "not tested",
        "not directly",
        "must not",
        "should not",
        "do not",
        "outside",
        "out of scope",
        "cpu gap",
        "evaluator gap",
        "forbidden",
        "invalid",
    )
    return any(term in lowered for term in allowed_terms)


def scan_symbol_reuse(
    generated_tests: Path,
    symbols: dict[str, str],
    *,
    filename: str,
    run_dir: Path,
) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    compiled = {symbol: whole_word_pattern(symbol) for symbol in symbols}

    for rel_name in sorted(CONTRACT_DOC_NAMES):
        path = generated_tests / rel_name
        if not path.exists() or not path.is_file():
            continue
        rel_hits: list[str] = []
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if line_is_non_contract_context(line):
                continue
            for symbol, pattern in compiled.items():
                if pattern.search(line):
                    rel_hits.append(f"line {lineno}: `{symbol}` ({symbols[symbol]})")
        if rel_hits:
            hits[rel_name] = sorted(set(rel_hits))

    for path in iter_generated_test_python_files(generated_tests):
        rel = str(path.relative_to(generated_tests))
        rel_hits: list[str] = []
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for symbol, pattern in compiled.items():
                if pattern.search(line):
                    rel_hits.append(f"line {lineno}: `{symbol}` ({symbols[symbol]})")
        if rel_hits:
            hits[rel] = sorted(set(rel_hits))

    lines: list[str] = []
    if hits:
        for rel, rel_hits in sorted(hits.items()):
            lines.append(f"{rel}:")
            lines.extend(f"  - {hit}" for hit in rel_hits)
    else:
        lines.append("none")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return hits


def extract_reference_only_python_symbols(workspace: Path) -> dict[str, str]:
    project_root = workspace / "project"
    reference_root = workspace / "reference_project"
    symbols: dict[str, str] = {}
    project_symbols = extract_python_defined_symbols(project_root)
    skip_names = {
        "main",
        "run",
        "test",
        "setup",
        "config",
        "logger",
        "initialize",
        "worker",
        "request",
        "response",
        "result",
        "state",
        "data",
        "model",
    }

    if not reference_root.exists():
        return symbols

    for ref_path in reference_root.rglob("*.py"):
        if any(part in {"__pycache__", ".git"} for part in ref_path.parts):
            continue
        rel = ref_path.relative_to(reference_root)
        if (project_root / rel).exists():
            continue
        try:
            tree = ast.parse(ref_path.read_text(errors="ignore"), filename=str(ref_path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if not is_likely_symbol(name) or name.lower() in skip_names:
                continue
            if name in project_symbols:
                continue
            symbols.setdefault(name, f"defined only in reference `{rel}`")
    return symbols


def reference_only_symbol_audit(
    run_dir: Path,
    workspace: Path,
    *,
    filename: str = "reference_only_symbols.txt",
) -> dict[str, list[str]]:
    generated_tests = workspace / "generated_tests"
    inferred_symbols = extract_inferred_contract_symbols(workspace)
    symbols = {
        symbol: source
        for symbol, source in extract_reference_only_python_symbols(workspace).items()
        if symbol not in inferred_symbols
    }
    return scan_symbol_reuse(generated_tests, symbols, filename=filename, run_dir=run_dir)


def extract_non_contract_symbols(workspace: Path) -> dict[str, str]:
    generated_tests = workspace / "generated_tests"
    non_contracts = generated_tests / "NON_CONTRACTS.md"
    if not non_contracts.exists():
        return {}

    symbols: dict[str, str] = {}
    text = non_contracts.read_text(errors="ignore")
    for line in text.splitlines():
        source = line
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) < 3 or set(cells[1].replace(" ", "")) <= {"-"}:
                continue
            source = cells[1]
        else:
            stripped = line.strip()
            heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
            leading_item = re.match(r"^[-*]\s+(`[^`]+`)(?:\s|$)", stripped)
            if heading:
                source = heading.group(1)
            elif leading_item:
                source = leading_item.group(1)
            else:
                continue
        for raw in re.findall(r"`([^`]+)`", source):
            value = raw.strip()
            if "/" in value or "://" in value:
                continue
            if "." in value:
                value = value.rsplit(".", 1)[-1]
            if not is_likely_symbol(value):
                continue
            if value.islower() and "_" not in value and not value.startswith("_"):
                continue
            symbols.setdefault(value, "listed in NON_CONTRACTS.md")
    return symbols


def non_contract_reuse_audit(
    run_dir: Path,
    workspace: Path,
    *,
    filename: str = "non_contract_reuse.txt",
) -> dict[str, list[str]]:
    generated_tests = workspace / "generated_tests"
    symbols = extract_non_contract_symbols(workspace)
    return scan_symbol_reuse(generated_tests, symbols, filename=filename, run_dir=run_dir)


def tool_usage_audit(run_dir: Path, *, filename: str = "tool_usage_audit.txt") -> dict[str, list[str]]:
    allowed = set(ALLOWED_TOOLS)
    hits: dict[str, list[str]] = {}
    for stream_path in sorted(run_dir.glob("*.stream.jsonl")):
        rel_hits: list[str] = []
        try:
            lines = stream_path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "tool_use":
                continue
            tool_name = str(event.get("tool_name") or "")
            if tool_name not in allowed:
                timestamp = event.get("timestamp") or f"line {index}"
                rel_hits.append(f"{timestamp}: `{tool_name}`")
        if rel_hits:
            hits[stream_path.name] = rel_hits

    lines: list[str] = []
    if hits:
        for rel, rel_hits in sorted(hits.items()):
            lines.append(f"{rel}:")
            lines.extend(f"  - {hit}" for hit in rel_hits)
    else:
        lines.append("none")
    (run_dir / filename).write_text("\n".join(lines) + "\n")
    return hits


def discover_candidate_patches(args: argparse.Namespace) -> list[Path]:
    patches = [path for path in args.candidate_patch if path.exists()]
    root = args.candidate_runs_root
    if root and root.exists() and args.max_candidate_patches != 0:
        discovered = sorted(root.rglob("candidate.patch"))
        if args.max_candidate_patches > 0:
            discovered = discovered[-args.max_candidate_patches :]
        patches.extend(discovered)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for patch in patches:
        resolved = patch.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def parse_validation_target(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--validation-target must be label=/path/to/project, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = slugify(label.strip())
    if not label:
        raise SystemExit(f"--validation-target label is empty: {spec}")
    if label in {"baseline", "reference"}:
        raise SystemExit(f"--validation-target label is reserved: {label}")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise SystemExit(f"--validation-target path does not exist for {label}: {path}")
    if not path.is_dir():
        raise SystemExit(f"--validation-target path is not a directory for {label}: {path}")
    return label, path


def parse_validation_targets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    if not args.no_default_validation_targets and DEFAULT_COUNTER_REFERENCE_PROJECT.exists():
        parsed.append(("counter_reference", DEFAULT_COUNTER_REFERENCE_PROJECT))
        seen.add("counter_reference")
    for spec in args.validation_target:
        label, path = parse_validation_target(spec)
        if label in seen:
            raise SystemExit(
                f"duplicate --validation-target label: {label}; "
                "use --no-default-validation-targets to replace a default target"
            )
        parsed.append((label, path))
        seen.add(label)
    return parsed


def materialize_candidate_targets(
    args: argparse.Namespace,
    workspace: Path,
    patches: list[Path],
) -> tuple[list[tuple[str, str, Path]], dict[str, str]]:
    targets: list[tuple[str, str, Path]] = [
        ("baseline", "project", workspace / "project"),
        ("reference", "reference_project", workspace / "reference_project"),
    ]
    patch_errors: dict[str, str] = {}

    validation_root = workspace / "validation_targets"
    for label, source_dir in parse_validation_targets(args):
        target_dir = validation_root / label
        rel_target = f"validation_targets/{label}"
        copy_clean_tree(source_dir, target_dir)
        targets.append((label, rel_target, target_dir))

    candidates_root = workspace / "candidate_targets"
    candidates_root.mkdir(exist_ok=True)

    for patch in patches:
        label = slugify(patch.parent.name)
        target_dir = candidates_root / label
        rel_target = f"candidate_targets/{label}"
        copy_clean_tree(args.project_source, target_dir)
        run_checked(["git", "init", "-b", "main"], cwd=target_dir)
        result = subprocess.run(
            ["git", "apply", str(patch)],
            cwd=str(target_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            patch_errors[label] = result.stderr.strip() or result.stdout.strip() or "git apply failed"
            shutil.rmtree(target_dir, ignore_errors=True)
            continue
        targets.append((label, rel_target, target_dir))
    return targets, patch_errors


def count_handwritten_tests() -> int:
    tests_dir = REPO_ROOT / "eval_tests" / "tests"
    count = 0
    for path in tests_dir.rglob("test_*.py"):
        text = path.read_text(errors="ignore")
        count += len(re.findall(r"^\s*(?:async\s+def|def)\s+test_", text, flags=re.MULTILINE))
    return count


def format_score(result: dict[str, object] | None) -> str:
    if result is None:
        return "-"
    if "error" in result:
        return f"error: {result['error']}"
    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    passed = int(summary.get("passed", 0))
    errors = int(summary.get("errors", 0))
    total = sum(int(value) for value in summary.values())
    exit_code = result.get("exit_code", "?")
    if errors and not passed and not summary.get("failed"):
        return f"pytest errors: {errors} (exit {exit_code})"
    if total == 0:
        return f"0/0 (exit {exit_code})"
    return f"{passed}/{total} (exit {exit_code})"


def write_score_table(
    run_dir: Path,
    target_labels: list[str],
    generated_test_count: int,
    generated_results: dict[str, dict[str, object]],
    manual_results: dict[str, dict[str, object]] | None,
    patch_errors: dict[str, str],
) -> None:
    rows: list[tuple[str, int, dict[str, dict[str, object]]]] = [
        ("generated", generated_test_count, generated_results)
    ]
    if manual_results is not None:
        rows.append(("hand_written", count_handwritten_tests(), manual_results))

    header = ["evaluator", "unit_tests", *target_labels]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for evaluator, test_count, results in rows:
        values = [evaluator, str(test_count)]
        for label in target_labels:
            values.append(format_score(results.get(label)))
        lines.append("| " + " | ".join(values) + " |")

    if patch_errors:
        lines.append("")
        lines.append("Patch application errors:")
        for label, error in sorted(patch_errors.items()):
            lines.append(f"- `{label}`: {error}")

    (run_dir / "score_table.md").write_text("\n".join(lines) + "\n")


def pass_count(result: dict[str, object]) -> int:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return 0
    return int(summary.get("passed", 0))


def generated_quality_issues(
    *,
    workspace: Path,
    generated_test_count: int,
    generated_results: dict[str, dict[str, object]],
    tool_usage_audit_result: dict[str, list[str]],
    static_audit_result: dict[str, list[str]],
    reference_only_audit_result: dict[str, list[str]],
    reference_only_contract_doc_audit_result: dict[str, list[str]],
    reference_only_symbol_audit_result: dict[str, list[str]],
    non_contract_reuse_audit_result: dict[str, list[str]],
    target_labels: list[str],
    validation_target_labels: list[str],
) -> list[str]:
    issues: list[str] = []
    reference = generated_results.get("reference")
    baseline = generated_results.get("baseline")

    if reference is None or reference.get("exit_code") != 0:
        issues.append("Reference project did not pass the generated evaluator with zero failures/errors.")
    for label in validation_target_labels:
        result = generated_results.get(label)
        if result is None or result.get("exit_code") != 0:
            issues.append(f"Validation target `{label}` did not pass the generated evaluator with zero failures/errors.")
    if baseline is None or baseline.get("exit_code") == 0:
        issues.append("Incomplete project did not fail the generated evaluator for the missing capability.")
    if isinstance(baseline, dict):
        baseline_summary = baseline.get("summary")
        if isinstance(baseline_summary, dict) and int(baseline_summary.get("errors", 0)) > 0:
            issues.append(
                "Incomplete project produced pytest errors instead of ordinary failing tests; "
                "the evaluator should collect under the incomplete project and fail during test execution."
            )
        if baseline.get("exit_code") == 2:
            issues.append(
                "Incomplete project exited with pytest code 2, usually collection/interruption; "
                "this is an evaluator bug, not an acceptable missing-capability failure."
            )

    if (workspace / "generated_tests" / "run_tests.sh").exists():
        issues.append(
            "Generated evaluator included generated_tests/run_tests.sh; custom runners are ignored. "
            "The generator should output pytest tests only."
        )

    if generated_test_count == 0:
        issues.append("No pytest test functions were generated.")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluator tests with Gemini CLI from exactly three benchmark inputs."
    )
    parser.add_argument("--run-name", default="gemini-testgen")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--gemini-runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--model", default=os.environ.get("GEMINI_CLI_MODEL", "gemini-3.5-flash"))
    parser.add_argument(
        "--gemini-thinking-level",
        choices=("default", "LOW", "HIGH"),
        default="HIGH",
        help="Gemini thinkingConfig override for Gemini 3 models; default uses an explicit HIGH alias.",
    )
    parser.add_argument("--key-index", type=int, default=0)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--project-source", type=Path, default=REPO_ROOT / "task" / "verl")
    parser.add_argument("--agent-prompt", type=Path, default=REPO_ROOT / "task" / "PROMPT.md")
    parser.add_argument("--reference-project", type=Path, default=DEFAULT_REFERENCE_PROJECT)
    parser.add_argument("--generator-prompt", type=Path, default=DEFAULT_GENERATOR_PROMPT)
    parser.add_argument("--generator-prompt-stage1", type=Path, default=DEFAULT_GENERATOR_PROMPT_STAGE1)
    parser.add_argument("--generator-prompt-stage2", type=Path, default=DEFAULT_GENERATOR_PROMPT_STAGE2)
    parser.add_argument(
        "--existing-generated-tests",
        type=Path,
        default=None,
        help="score an existing generated_tests directory instead of invoking Gemini",
    )
    parser.add_argument("--candidate-patch", type=Path, action="append", default=[])
    parser.add_argument("--candidate-runs-root", type=Path, default=DEFAULT_CANDIDATE_RUNS_ROOT)
    parser.add_argument(
        "--validation-target",
        action="append",
        default=[],
        metavar="LABEL=/path/to/project",
        help=(
            "additional known-good target expected to pass the generated evaluator; "
            "can be repeated, e.g. counter_reference=/path/to/alt_reference_project"
        ),
    )
    parser.add_argument(
        "--no-default-validation-targets",
        action="store_true",
        help="do not auto-add repo-default known-good validation targets such as the VERL counter reference",
    )
    parser.add_argument(
        "--max-candidate-patches",
        type=int,
        default=0,
        help=(
            "number of candidate.patch files to discover under --candidate-runs-root; "
            "0 disables discovery, negative means all"
        ),
    )
    parser.add_argument("--score-manual-suite", action="store_true")
    parser.add_argument("--max-session-turns", type=int, default=220)
    parser.add_argument("--shell-timeout-sec", type=int, default=180)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Gemini timeout; 0 disables timeout and lets the patched CLI retry transient API failures indefinitely",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=1,
        help=(
            "retry failed Gemini generation attempts this many times for stream/subprocess/timeout failures; "
            "quality failures after successful generation are not retried"
        ),
    )
    parser.add_argument("--test-timeout-sec", type=int, default=240)
    parser.add_argument("--num-runs", type=int, default=1, help="number of independent Gemini generations")
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=0,
        help="deprecated no-op; generated evaluators are no longer repaired from validation feedback",
    )
    parser.add_argument(
        "--stage-repair-rounds",
        type=int,
        default=0,
        help="deprecated; staged generation now records audit findings without rule-based repair",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("single", "staged", "two-stage"),
        default="single",
        help=(
            "single prompt generation, five-stage project-first generation, or two-stage "
            "project-first pytest generation followed by reference-oracle refinement"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def is_retryable_generation_error(exc: BaseException) -> bool:
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        return True
    if isinstance(exc, RuntimeError) and "Gemini stream reported failure" in str(exc):
        return True
    return False


def retry_run_dir(base_run_dir: Path, attempt: int) -> Path:
    if attempt == 0:
        return base_run_dir
    return base_run_dir.with_name(f"{base_run_dir.name}-retry{attempt}")


def run_once_with_generation_retries(args: argparse.Namespace, base_run_dir: Path, run_index: int) -> dict[str, object]:
    max_attempts = args.generation_retries + 1
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        run_dir = retry_run_dir(base_run_dir, attempt)
        try:
            return run_once(args, run_dir, run_index + attempt)
        except Exception as exc:
            last_exc = exc
            if not is_retryable_generation_error(exc) or attempt + 1 >= max_attempts:
                raise
            error_record = {
                "status": "retrying",
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "generation_retry.json").write_text(json.dumps(error_record, indent=2, sort_keys=True) + "\n")
            print(
                f"generation attempt {attempt + 1}/{max_attempts} failed with retryable error: "
                f"{error_record['error']}; retrying",
                flush=True,
            )
    assert last_exc is not None
    raise last_exc


def run_once(args: argparse.Namespace, run_dir: Path, run_index: int) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = prepare_workspace(args, run_dir)
    print(f"prepared test-generation workspace: {workspace}", flush=True)

    if args.prepare_only:
        return {
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "status": "prepared",
        }

    if args.existing_generated_tests is not None:
        copy_clean_tree(args.existing_generated_tests, workspace / "generated_tests")
    else:
        run_args = argparse.Namespace(**vars(args))
        run_args.key_index = args.key_index + run_index
        if args.generation_mode == "staged":
            run_staged_gemini(run_args, run_dir, workspace)
        elif args.generation_mode == "two-stage":
            run_two_stage_gemini(run_args, run_dir, workspace)
        else:
            run_gemini(run_args, run_dir, workspace)

    if args.skip_validation:
        tool_audit = tool_usage_audit(run_dir)
        audit = static_audit(run_dir, workspace)
        generated_test_count = count_generated_tests(workspace)
        return {
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "status": "generated",
            "generated_test_count": generated_test_count,
            "tool_usage_audit": tool_audit,
            "static_audit": audit,
        }

    candidate_patches = discover_candidate_patches(args)
    targets, patch_errors = materialize_candidate_targets(args, workspace, candidate_patches)
    target_labels = [label for label, _rel_target, _target_dir in targets]
    validation_target_labels = [label for label, _path in parse_validation_targets(args)]
    candidate_target_labels = [
        label
        for label in target_labels
        if label not in {"baseline", "reference"} and label not in set(validation_target_labels)
    ]
    target_groups = {
        "initial": ["baseline"],
        "official_reference": ["reference"],
        "counter_references": validation_target_labels,
        "candidate_variants": candidate_target_labels,
    }

    final_audit: dict[str, list[str]] = {}
    final_tool_usage_audit: dict[str, list[str]] = {}
    final_reference_only_audit: dict[str, list[str]] = {}
    final_reference_only_contract_doc_audit: dict[str, list[str]] = {}
    final_reference_only_symbol_audit: dict[str, list[str]] = {}
    final_non_contract_reuse_audit: dict[str, list[str]] = {}
    final_generated_test_count = 0
    final_generated_results: dict[str, dict[str, object]] = {}
    final_issues: list[str] = []

    attempt_label = "attempt0"
    final_tool_usage_audit = tool_usage_audit(run_dir, filename=f"tool_usage_audit_{attempt_label}.txt")
    final_audit = static_audit(run_dir, workspace, filename=f"static_audit_{attempt_label}.txt")
    final_reference_only_audit = reference_only_dotted_path_audit(
        run_dir,
        workspace,
        filename=f"reference_only_dotted_paths_{attempt_label}.txt",
    )
    final_reference_only_contract_doc_audit = reference_only_contract_doc_audit(
        run_dir,
        workspace,
        filename=f"reference_only_contract_docs_{attempt_label}.txt",
    )
    final_reference_only_symbol_audit = reference_only_symbol_audit(
        run_dir,
        workspace,
        filename=f"reference_only_symbols_{attempt_label}.txt",
    )
    final_non_contract_reuse_audit = non_contract_reuse_audit(
        run_dir,
        workspace,
        filename=f"non_contract_reuse_{attempt_label}.txt",
    )
    final_generated_test_count = count_generated_tests(workspace)
    final_generated_results = {}
    for label, rel_target, _target_dir in targets:
        final_generated_results[label] = run_generated_tests(
            args,
            run_dir,
            workspace,
            label,
            rel_target,
            log_prefix=f"generated_{attempt_label}",
        )

    final_issues = generated_quality_issues(
        workspace=workspace,
        generated_test_count=final_generated_test_count,
        generated_results=final_generated_results,
        tool_usage_audit_result=final_tool_usage_audit,
        static_audit_result=final_audit,
        reference_only_audit_result=final_reference_only_audit,
        reference_only_contract_doc_audit_result=final_reference_only_contract_doc_audit,
        reference_only_symbol_audit_result=final_reference_only_symbol_audit,
        non_contract_reuse_audit_result=final_non_contract_reuse_audit,
        target_labels=target_labels,
        validation_target_labels=validation_target_labels,
    )

    attempt_summary = {
        "attempt": 0,
        "generated_test_count": final_generated_test_count,
        "tool_usage_audit": final_tool_usage_audit,
        "static_audit": final_audit,
        "reference_only_dotted_paths": final_reference_only_audit,
        "reference_only_contract_docs": final_reference_only_contract_doc_audit,
        "reference_only_symbols": final_reference_only_symbol_audit,
        "non_contract_reuse": final_non_contract_reuse_audit,
        "target_labels": target_labels,
        "validation_target_labels": validation_target_labels,
        "target_groups": target_groups,
        "patch_errors": patch_errors,
        "generated_results": final_generated_results,
        "quality_issues": final_issues,
    }
    (run_dir / f"validation_summary_{attempt_label}.json").write_text(
        json.dumps(attempt_summary, indent=2, sort_keys=True) + "\n"
    )

    manual_results: dict[str, dict[str, object]] | None = None
    if args.score_manual_suite:
        manual_results = {}
        for label, _rel_target, target_dir in targets:
            manual_results[label] = run_manual_suite(args, run_dir, label, target_dir)

    write_score_table(
        run_dir,
        target_labels,
        final_generated_test_count,
        final_generated_results,
        manual_results,
        patch_errors,
    )

    reference_result = final_generated_results["reference"]
    baseline_result = final_generated_results["baseline"]
    summary = {
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "generated_test_count": final_generated_test_count,
        "tool_usage_audit": final_tool_usage_audit,
        "static_audit": final_audit,
        "reference_only_dotted_paths": final_reference_only_audit,
        "reference_only_contract_docs": final_reference_only_contract_doc_audit,
        "reference_only_symbols": final_reference_only_symbol_audit,
        "non_contract_reuse": final_non_contract_reuse_audit,
        "target_labels": target_labels,
        "validation_target_labels": validation_target_labels,
        "target_groups": target_groups,
        "patch_errors": patch_errors,
        "generated_results": final_generated_results,
        "manual_results": manual_results,
        "quality_issues": final_issues,
        "reference": reference_result,
        "baseline": baseline_result,
        "reference_exit_code": reference_result["exit_code"],
        "baseline_exit_code": baseline_result["exit_code"],
    }
    (run_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    if args.generation_retries < 0:
        raise SystemExit("--generation-retries must be >= 0")
    if args.stage_repair_rounds < 0:
        raise SystemExit("--stage-repair-rounds must be >= 0")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    slug = slugify(args.run_name)

    if args.num_runs == 1:
        run_dir = args.runs_root / f"testgen-{slug}-{timestamp}"
        summary = run_once_with_generation_retries(args, run_dir, 0)
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
    batch_results: list[dict[str, object]] = []
    for index in range(args.num_runs):
        run_dir = batch_dir / f"run-{index + 1:02d}"
        try:
            result = run_once_with_generation_retries(args, run_dir, index)
        except Exception as exc:
            result = {
                "run_dir": str(run_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"run {index + 1} failed: {result['error']}", flush=True)
        batch_results.append(result)

    batch_summary = {
        "batch_dir": str(batch_dir),
        "num_runs": args.num_runs,
        "results": batch_results,
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, sort_keys=True) + "\n")

    if args.prepare_only or args.skip_validation:
        return 0

    acceptable = [
        result
        for result in batch_results
        if result.get("reference_exit_code") == 0
        and result.get("baseline_exit_code") not in (0, None)
        and not result.get("quality_issues")
    ]
    return 0 if acceptable else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
