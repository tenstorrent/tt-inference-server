# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module.agentic import local_datasets, swebench_pro
from llm_module.agentic.local_datasets import prepare_local_dataset
from llm_module.agentic.swebench_pro import (
    SWEBENCH_PRO_COMMIT,
    SWEBENCH_PRO_DATASET,
    sanitize_task,
)

# Trimmed from the registry task layout (harbor-datasets datasets/swebenchpro).
_DOCKERFILE = """\
FROM jefzda/sweap-images:example
WORKDIR /app
RUN set -e && git reset --hard {base} && git clean -fd && git checkout {base}

# Install uv for running Python scripts
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh || true
"""
_TEST_SH = """\
#!/bin/bash
if [ -n "$GOLD_CHECKOUT" ]; then
    echo "Running gold tests checkout: $GOLD_CHECKOUT"
    eval "$GOLD_CHECKOUT" || { echo "ERROR: Gold tests checkout failed"; exit 1; }
fi
"""
_TASK_TOML = """\
[verifier]
timeout_sec = 3000

[agent]
timeout_sec = 3000
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository whose fix commit edits one test and adds another."""
    repo = tmp_path / "app"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "tests").mkdir()
    (repo / "src.py").write_text("x = 1\n")
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "src.py").write_text("x = 2\n")
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
    (repo / "tests" / "test_b.py").write_text("def test_b():\n    pass\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "the fix")
    fix = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v2")
    test_patch = _git(repo, "diff", base, fix, "--", "tests") + "\n"
    _git(repo, "checkout", "-q", base)
    _git(repo, "checkout", "-q", "-B", "main")
    return repo, base, fix, test_patch


def _task(tmp_path, base, fix, test_patch):
    task = tmp_path / "instance_example"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE.format(base=base))
    (task / "tests" / "test.sh").write_text(_TEST_SH)
    (task / "task.toml").write_text(_TASK_TOML)
    checkout = f"git checkout {fix} -- tests/test_a.py tests/test_b.py"
    (task / "tests" / "config.json").write_text(
        json.dumps(
            {
                "repo": "example/app",
                "base_commit": base,
                "test_patch": test_patch,
                "before_repo_set_cmd": f"git reset --hard {base}\n{checkout}",
            }
        )
    )
    return task


def _strip_history_script(dockerfile: Path) -> str:
    """The history-stripping RUN step of a sanitized Dockerfile, as shell."""
    text = dockerfile.read_text()
    run = re.search(r"^RUN set -e; d=/app;.*?(?<!\\)\n", text, re.M | re.S).group(0)
    return run[len("RUN ") :].replace("\\\n", "\n")


def test_swebench_pro_is_a_local_dataset():
    gen = local_datasets._GENERATORS[SWEBENCH_PRO_DATASET]
    assert gen is swebench_pro.generate_swebench_pro


def test_sanitized_image_has_no_fix_and_tests_still_restore(tmp_path, repo):
    repo, base, fix, test_patch = repo
    task = _task(tmp_path, base, fix, test_patch)

    sanitize_task(task)

    # The Dockerfile step leaves only the base commit.
    script = _strip_history_script(task / "environment" / "Dockerfile")
    subprocess.run(
        [
            "bash",
            "-c",
            script.replace("d=/app; [ -d /app/.git ] || d=/testbed;", f"d={repo};", 1),
        ],
        check=True,
        capture_output=True,
    )
    assert _git(repo, "rev-list", "--all").split() == [base]
    assert _git(repo, "for-each-ref", "--format=%(refname)") == "refs/heads/main"
    assert (
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", fix]).returncode != 0
    )

    # The agent edits a gold test file and creates the other one...
    (repo / "tests" / "test_a.py").write_text("agent edit\n")
    (repo / "tests" / "test_b.py").write_text("agent file\n")
    # ...and the verifier restores both to the fix's content without history.
    tests = task / "tests"
    restore = (tests / "restore_gold_tests.sh").read_text()
    restore = restore.replace("/tests/", f"{tests}/")
    subprocess.run(["bash", "-c", restore], cwd=repo, check=True, capture_output=True)
    assert (repo / "tests" / "test_a.py").read_text() == "def test_a():\n    assert 1\n"
    assert (repo / "tests" / "test_b.py").read_text() == "def test_b():\n    pass\n"
    assert (repo / "src.py").read_text() == "x = 1\n"


def test_sanitize_rewrites_verifier_and_blocks_agent_network(tmp_path, repo):
    _, base, fix, test_patch = repo
    task = _task(tmp_path, base, fix, test_patch)

    sanitize_task(task)

    test_sh = (task / "tests" / "test.sh").read_text()
    assert 'eval "$GOLD_CHECKOUT"' not in test_sh
    assert "bash /tests/restore_gold_tests.sh" in test_sh
    assert (task / "tests" / "gold_tests.patch").read_text() == test_patch
    toml = (task / "task.toml").read_text()
    assert '[agent]\nnetwork_mode = "no-network"\n' in toml
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert f"! git cat-file -e {fix}^{{commit}}" in dockerfile
    assert dockerfile.index("git cat-file") < dockerfile.index("# Install uv")


def test_sanitize_refuses_an_unexpected_task(tmp_path, repo):
    _, base, fix, test_patch = repo
    task = _task(tmp_path, base, fix, test_patch)
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")

    with pytest.raises(ValueError, match="expected exactly one"):
        sanitize_task(task)


def test_generates_once_from_the_pinned_commit_then_reuses_it(tmp_path, repo):
    _, base, fix, test_patch = repo
    template = _task(tmp_path / "template", base, fix, test_patch)
    python = tmp_path / "venv" / "bin" / "python"
    calls = []

    def fake_git(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            (Path(cmd[-1]) / ".git").mkdir(parents=True)
        elif "checkout" in cmd:
            src = Path(cmd[cmd.index("-C") + 1]) / "datasets" / "swebenchpro"
            shutil.copytree(template, src / template.name)

    with patch.object(swebench_pro.subprocess, "run", side_effect=fake_git):
        first = prepare_local_dataset(SWEBENCH_PRO_DATASET, python)
        n_calls = len(calls)
        second = prepare_local_dataset(SWEBENCH_PRO_DATASET, python)

    assert first == second
    assert first.name == SWEBENCH_PRO_DATASET
    assert first.parent.name.startswith(f"swebenchpro-{SWEBENCH_PRO_COMMIT[:12]}-")
    assert SWEBENCH_PRO_COMMIT in calls[1]
    assert len(calls) == n_calls
    assert (first / template.name / "tests" / "restore_gold_tests.sh").is_file()
    assert not list(first.parent.parent.glob("*.tmp-*"))
