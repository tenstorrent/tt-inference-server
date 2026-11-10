# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from llm_module.agentic.leak_scan import (
    LEAK_SCAN_FILE,
    scan_trajectory,
    write_leak_scan,
)
from llm_module.parsers.agentic import AgenticEvalParser


def _step(step_id, command, output, *, call_id=None):
    """An ATIF step with one Bash call; mini-swe-agent results carry no id."""
    result = {"content": output}
    if call_id:
        result["source_call_id"] = call_id
    return {
        "step_id": step_id,
        "tool_calls": [
            {
                "tool_call_id": call_id or f"c{step_id}",
                "function_name": "Bash" if call_id else "bash",
                "arguments": {"command": command},
            }
        ],
        "observation": {"results": [result]},
    }


def _commands(hits):
    return [(h["kind"], h["command"]) for h in hits]


def test_flags_code_host_fetches_and_whether_they_got_through():
    trajectory = {
        "steps": [
            # Seen in a leaked SWE-bench Pro trial: the upstream fix, fetched.
            _step(
                1,
                'curl -s "https://api.github.com/repos/o/app/commits/18c03daa86"',
                '{"sha": "' + "a" * 40 + '", "files": []}',
            ),
            # The same attempt with egress blocked.
            _step(
                2,
                "git clone https://github.com/o/app /tmp/up",
                "Cloning into '/tmp/up'...\nfatal: unable to access "
                "'https://github.com/o/app/': Could not resolve host: github.com",
                call_id="toolu_2",
            ),
        ]
    }

    hits = scan_trajectory(trajectory)

    assert [(h["kind"], h["fetched"]) for h in hits] == [
        ("network", True),
        ("network", False),
    ]


def test_flags_history_lookups_beyond_the_base_commit():
    trajectory = {
        "steps": [
            _step(1, "cd /app && git log --all --oneline | head -20", "..."),
            _step(2, "git show a6e671db25 -- lib/aix.py", "..."),
            _step(3, "git branch -a --contains 407407d", "..."),
            _step(4, "git reflog", "..."),
        ]
    }

    assert [h["kind"] for h in scan_trajectory(trajectory)] == ["history"] * 4


def test_ordinary_work_is_not_flagged():
    trajectory = {
        "steps": [
            _step(1, "cd /app && git status && git diff", "..."),
            _step(2, "git log --oneline -5", "..."),
            _step(3, "grep -rn 'github.com/o/dep' go.mod", "..."),
            _step(4, "pip install -e . && pytest tests/test_a.py", "..."),
            _step(5, "curl -s http://172.17.0.1:8000/v1/models", "..."),
        ]
    }

    assert scan_trajectory(trajectory) == []


def test_flags_a_package_fetch_of_the_project_itself():
    trajectory = {
        "steps": [
            _step(1, "pip download ansible-core==2.15 --no-deps", "..."),
            _step(2, "pip download ansible==9.0 --no-deps", "Saved ./ansible.whl"),
        ]
    }

    hits = scan_trajectory(trajectory, project="ansible/ansible")

    assert _commands(hits) == [("network", "pip download ansible==9.0 --no-deps")]


def test_write_leak_scan_reports_into_the_job_result(tmp_path):
    job = tmp_path / "swe_bench_pro"
    for trial, command, reward in (
        ("t_leak", "curl -s https://github.com/o/app/commit/abc1234.diff", 1.0),
        ("t_clean", "pytest -q", 0.0),
    ):
        (job / trial / "agent").mkdir(parents=True)
        (job / trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": f"instance_{trial}",
                    "verifier_result": {"rewards": {"reward": reward}},
                }
            )
        )
        (job / trial / "agent" / "trajectory.json").write_text(
            json.dumps({"steps": [_step(1, command, "HTTP/2 200\n" + "x" * 300)]})
        )
    result_file = job / "result.json"
    result_file.write_text(json.dumps({"stats": {"evals": {}}}))

    summary = write_leak_scan(job, result_file)

    assert summary == {
        "n_trials_scanned": 2,
        "n_flagged": 1,
        "network_attempts": ["t_leak"],
        "network_fetched": ["t_leak"],
        "history_lookups": [],
    }
    details = json.loads((job / LEAK_SCAN_FILE).read_text())
    assert details["flagged"][0]["reward"] == 1.0
    raw = json.loads(result_file.read_text())
    assert raw["leak_scan"] == summary

    block = AgenticEvalParser(task_name="swe_bench_pro", score=None).parse(raw)
    assert block.targets["leak_scan"] == summary
