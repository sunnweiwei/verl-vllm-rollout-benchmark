# Prompt Versions

Prompt files in this directory are append-only experiment versions.

- `unit-test-generator-prompt-v1.md`: user-edited prompt committed as `08710f0`.
- `unit-test-generator-prompt-v2.md`: v1 copied forward with minimal edits: generated output is limited to pytest tests plus optional helpers/notes, direct-testing language is softened through the naming exception, coverage/audit notes are optional, and the task prompt is inlined through `{{AGENT_PROMPT}}`. The benchmark harness, not the generator, owns test execution.
- `unit-test-generator-prompt-v3.md`: v2 plus a narrow alternative-implementation litmus test and final import/name self-check, aimed at preventing tests that pass only because the official reference uses particular internal module paths, class names, helper names, or file layout.

The current runners default to `unit-test-generator-prompt-v3.md`. Add future prompt revisions as new versioned files instead of overwriting prior versions.
