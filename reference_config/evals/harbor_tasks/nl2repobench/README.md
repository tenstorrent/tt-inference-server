# NL2Repo-Bench as Harbor tasks (Claude Code harness)

Generates one Harbor task per upstream NL2Repo-Bench project
(github.com/multimodal-art-projection/NL2RepoBench @781a1da, `test_files/`, 104 tasks).

    git clone https://github.com/multimodal-art-projection/NL2RepoBench /path/NL2RepoBench
    python gen_tasks.py /path/NL2RepoBench "$TT_NL2REPO_TASKS_DIR" [task ...]

Faithful to the official OpenHands pipeline except for the agent:

- Agent container: `python:3.12-bookworm` (official runtime is OpenHands Python 3.12)
  with only `/workspace/start.md`; the official prompt as `instruction.md`.
- `/workspace` is a directory artifact, graded in a *separate* verifier container
  built from `ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0`, whose own
  `/workspace` (upstream tests + packaging) the agent never sees. The upstream
  Harbor adapter (harbor-framework/harbor#482) runs the agent inside that image,
  which exposes the hidden tests.
- `task_files/grade.py` reproduces `openhands/post_processor.py`: delete packaging and
  test files from the agent workspace, copy it over the official `/workspace`, run
  `test_commands.json`, reward = min(sum of "N passed" / test_case_count, 1).
  The benchmark score is the mean reward.
- `task_files/guard.py` is a Claude Code PreToolUse hook (managed settings) that blocks
  `pip download`, `git clone`, and installs/fetches naming the target project. The
  Qwen3.8 card applies such a restriction but does not publish its rules, so ours
  are an approximation.
- Timeouts: agent 4 h, verifier 1 h (upstream sets none).
