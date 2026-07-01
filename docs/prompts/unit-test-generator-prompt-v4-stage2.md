# Evaluator-Generation Agent - Stage 2

You are resuming the evaluator-generation task from Stage 1.

`/workspace/reference_project` is now available. Use it only as an executable behavior oracle to refine the pytest evaluator already written under `/workspace/generated_tests`.

## Inputs
- `/workspace/project` - the incomplete project seen by candidate agents.
- `/workspace/AGENT_PROMPT.md` - the candidate-facing task prompt.
- `/workspace/generated_tests` - the Stage 1 evaluator.
- `/workspace/reference_project` - the complete reference solution.

If your session context is unavailable, first read the Stage 1 prompt and existing generated tests. Do not restart from scratch unless the existing evaluator is unusable.

## Reference Rule
The reference project is evidence, not a hidden specification.

Use it to:
- run reference tests and small probes;
- calibrate expected behavior for public or prompt-implied surfaces;
- find missing behavior that is already justified by `/workspace/AGENT_PROMPT.md`, `/workspace/project`, or neighboring public workflows.

Do not add tests for module paths, class names, helper names, actor names, private registries, file layouts, or edge behavior visible only in `/workspace/reference_project`. A correct alternative implementation with different internals must be able to pass.

## Work
1. Review the Stage 1 tests and identify shallow, structural, or missing coverage.
2. Probe `/workspace/reference_project` to verify concrete expected behavior for valid public surfaces.
3. Refine `/workspace/generated_tests` so each important candidate-visible behavior has a pytest-level check when it is CPU-testable.
4. Record important non-tested gaps only when they cannot be fairly tested without private internals or heavyweight runtime.
5. Validate the final tests:

```bash
PYTHONPATH=/workspace/reference_project:/workspace/generated_tests python -m pytest /workspace/generated_tests
PYTHONPATH=/workspace/project:/workspace/generated_tests python -m pytest /workspace/generated_tests
```

The reference run must pass completely. The incomplete-project run should collect successfully and fail at test runtime because the missing capability is absent. Collection errors from direct imports of missing candidate code mean the evaluator is over-specified.

Do a coverage sufficiency check before finishing. If the task is a broad integration capability, the final evaluator should usually contain many independent pytest test nodes across the distinct public behavior targets discovered in Stage 1 and Stage 2. Do not stop after a handful of selector/config/lifecycle tests when additional CPU-testable downstream behavior is available.

## Output
- Only pytest tests and optional notes/coverage artifacts under `/workspace/generated_tests`.
- No custom test runner, shell entrypoint, package install script, or edits outside `/workspace/generated_tests`.
- CPU-only, deterministic, hermetic tests. No GPU, network, model downloads, external services, or real distributed cluster.

Now continue the task: refine the existing evaluator using the reference project as an oracle, then validate it against both projects.
