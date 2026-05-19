# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

import json
from pathlib import Path

from evals import deepseek_external_runner as runner


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_lm_eval_score_extracts_percent(tmp_path):
    result_path = (
        tmp_path
        / "math500"
        / "deepseek-ai__DeepSeek-R1-0528"
        / "results_2026-05-13T00-00-00.json"
    )
    _write_json(
        result_path,
        {"results": {"r1_math500": {"exact_match,none": 0.875}}},
    )

    score = runner._read_lm_eval_score(
        tmp_path / "math500",
        runner.BENCHMARKS["math500"],
    )

    assert score == 87.5


def test_pass1_summary_supports_majority_metric(tmp_path):
    _write_json(
        tmp_path / "aime24_pass1" / "r1_aime24_pass1_summary.json",
        {
            "num_samples": 4,
            "num_problems": 2,
            "pass_at_1": 0.75,
            "pass_at_1_percent": 75.0,
            "majority_at_1": 0.5,
            "majority_at_1_percent": 50.0,
        },
    )

    score = runner._read_pass1_score(
        tmp_path / "aime24_pass1",
        "r1_aime24",
        "majority_at_1_percent",
    )

    assert score == 50.0


def test_combined_aime_scores_are_weighted_by_samples_and_problems(tmp_path):
    _write_json(
        tmp_path / "aime24_pass1" / "r1_aime24_pass1_summary.json",
        {
            "num_samples": 4,
            "num_problems": 2,
            "pass_at_1": 0.75,
            "pass_at_1_percent": 75.0,
            "majority_at_1": 0.5,
            "majority_at_1_percent": 50.0,
        },
    )
    _write_json(
        tmp_path / "aime25_pass1" / "r1_aime25_pass1_summary.json",
        {
            "num_samples": 2,
            "num_problems": 1,
            "pass_at_1": 0.5,
            "pass_at_1_percent": 50.0,
            "majority_at_1": 1.0,
            "majority_at_1_percent": 100.0,
        },
    )

    assert runner._combined_score(tmp_path, "pass1") == 100.0 * 4 / 6
    assert runner._combined_score(tmp_path, "majority") == 100.0 * 2 / 3


def test_summary_table_is_compact():
    markdown = runner._summary_markdown(
        "smoke",
        Path("/tmp/out"),
        [
            runner.BenchmarkResult(
                benchmark="MATH-500 pass@1",
                measured=86.0,
                reference=85.95,
                output_dir="/tmp/out/math500",
                status="complete",
            )
        ],
    )

    assert "| Benchmark | Measured | Reference |" in markdown
    assert "Delta vs FP8" not in markdown


def test_resolve_benchmarks_adds_dependencies_in_order():
    class Args:
        benchmarks = "aime24_25_pass1,math500"

    resolved = runner._resolve_benchmarks(Args(), runner.MODE_CONFIGS["smoke"])

    assert resolved == (
        "aime24_pass1",
        "aime25_pass1",
        "aime24_25_pass1",
        "math500",
    )


def test_list_benchmarks_does_not_require_mode(capsys):
    try:
        runner.parse_args(["--list-benchmarks"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")

    assert "math500" in capsys.readouterr().out


def test_sanity_mode_is_gentle():
    config = runner.MODE_CONFIGS["sanity"]

    assert config.benchmarks == ("aime24_short_sanity",)
    assert config.max_concurrent == 1
    assert config.max_gen_toks == 32768


def test_apply_mode_env_overrides_sanity_limits():
    env = runner._apply_mode_env(
        {"MAX_CONCURRENT": "30", "MAX_GEN_TOKS": "65535"},
        runner.MODE_CONFIGS["sanity"],
    )

    assert env["MAX_CONCURRENT"] == "1"
    assert env["MAX_GEN_TOKS"] == "32768"


def test_sanity_exit_code_passes_only_for_full_score(tmp_path, capsys):
    _write_json(
        tmp_path
        / "aime24_short_sanity"
        / "deepseek-ai__DeepSeek-R1-0528"
        / "results_2026-05-13T00-00-00.json",
        {
            "n-samples": {"r1_aime24_short": {"original": 5, "effective": 5}},
            "results": {"r1_aime24_short": {"exact_match,none": 1.0}},
        },
    )
    result = runner._collect_result(
        tmp_path,
        runner.BENCHMARKS["aime24_short_sanity"],
        "complete",
    )

    assert runner._sanity_exit_code(tmp_path, [result]) == 0
    assert "Sanity passed: 5/5" in capsys.readouterr().out


def test_sanity_exit_code_fails_for_partial_score(tmp_path, capsys):
    _write_json(
        tmp_path
        / "aime24_short_sanity"
        / "deepseek-ai__DeepSeek-R1-0528"
        / "results_2026-05-13T00-00-00.json",
        {
            "n-samples": {"r1_aime24_short": {"original": 5, "effective": 5}},
            "results": {"r1_aime24_short": {"exact_match,none": 0.8}},
        },
    )
    result = runner._collect_result(
        tmp_path,
        runner.BENCHMARKS["aime24_short_sanity"],
        "complete",
    )

    assert runner._sanity_exit_code(tmp_path, [result]) == 1
    assert "got 4/5" in capsys.readouterr().err
