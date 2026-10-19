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
- Reward-hack fixes over the official grader (cf. PrimeIntellect-ai/prime-envs#929),
  each a deviation from upstream scoring; `nl2repo_result.json` also records the
  upstream count (`upstream_passed` / `upstream_reward`) of the same run:
  - agent `conftest.py` / `sitecustomize.py` / `usercustomize.py` are stripped, as are
    top-level agent modules shadowing anything importable in the verifier (`pytest`,
    stdlib); the reference's test files and conftests win over agent files.
  - Only pytest's final summary line counts (upstream sums every "N passed" in the log),
    and a clean summary with a non-zero pytest exit counts as fabricated (0 passed).
  - The six bare `pytest` commands name the hidden test paths (`EXPLICIT_TEST_PATHS`);
    upstream they also collect agent-written tests (1.0 with no implementation).
  - Nine official images ship the real project in site-packages (databases, gitingest,
    justext, phe, pytorch-grad-cam, pytz, scikit-learn, stamina, wifiphisher): when the
    agent's package fails to import, tests pass against the real library; more-Itertools'
    test commands install it as a dependency (0.98 with an empty workspace). Copies
    outside `/workspace` are deleted at image build and before every pytest command
    (`task_files/drop_preinstalled.py`).
  - The verifier runs as root (pyautogui's image sets a non-root user).
  - `start.md` is stored in Git LFS: generation fails on a pointer file (`git lfs pull`).
- `task_files/guard.py` is a Claude Code PreToolUse hook (managed settings) that blocks
  `pip download`, `git clone`, and installs/fetches naming the target project. The
  Qwen3.8 card applies such a restriction but does not publish its rules, so ours
  are an approximation.
- Timeouts: agent 4 h, verifier 1 h (upstream sets none). Claude Code runs with
  `max_turns=900`, the turn cap of the Qwen3.6-27B card's NL2Repo footnote.
