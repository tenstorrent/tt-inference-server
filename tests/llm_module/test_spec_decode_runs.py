# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec-decode sweep presets and the ISL / concurrency env overrides."""

import pytest

from llm_module.spec_decode import (
    build_runs,
    parse_concurrencies,
    parse_isls,
)


_THROUGHPUT_PREFIX = "speed_bench_throughput_"


@pytest.fixture(autouse=True)
def _no_override_env(monkeypatch):
    monkeypatch.delenv("SPEC_DECODE_ISLS", raising=False)
    monkeypatch.delenv("SPEC_DECODE_CONCURRENCIES", raising=False)


def _throughput(runs):
    return [
        (r.public_dataset[len(_THROUGHPUT_PREFIX) :], r.max_concurrency)
        for r in runs
        if r.public_dataset.startswith(_THROUGHPUT_PREFIX)
    ]


def _qualitative(runs):
    return [
        r.public_dataset
        for r in runs
        if not r.public_dataset.startswith(_THROUGHPUT_PREFIX)
    ]


def test_full_preset_grid():
    runs = build_runs("full")
    assert len(_qualitative(runs)) == 11
    assert _throughput(runs) == [
        (isl, c) for isl in ("1k", "2k", "8k", "16k", "32k") for c in (1, 8, 16, 32, 64)
    ]


def test_ci_preset_keeps_three_concurrencies():
    runs = build_runs("ci")
    assert _qualitative(runs) == ["speed_bench_coding"]
    assert _throughput(runs) == [("32k", 1), ("32k", 16), ("32k", 64)]


def test_throughput_preset_has_no_qualitative_runs():
    runs = build_runs("throughput")
    assert _qualitative(runs) == []
    assert len(_throughput(runs)) == 25


def test_num_prompts_scale_with_concurrency():
    runs = build_runs("throughput", isls="1k")
    assert [r.num_prompts for r in runs] == [32, 32, 64, 128, 256]


def test_overrides_replace_preset_grid():
    runs = build_runs("throughput", isls="8k, 1k", concurrencies="32,8")
    assert _throughput(runs) == [("1k", 8), ("1k", 32), ("8k", 8), ("8k", 32)]


def test_overrides_leave_qualitative_runs_alone():
    runs = build_runs("ci", isls="1k", concurrencies="8")
    assert _qualitative(runs) == ["speed_bench_coding"]
    assert _throughput(runs) == [("1k", 8)]


def test_env_vars_replace_preset_grid(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_ISLS", "8k,1k")
    monkeypatch.setenv("SPEC_DECODE_CONCURRENCIES", "32,8")
    runs = build_runs("ci")
    assert _qualitative(runs) == ["speed_bench_coding"]
    assert _throughput(runs) == [("1k", 8), ("1k", 32), ("8k", 8), ("8k", 32)]


def test_env_isls_keep_preset_concurrencies(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_ISLS", "2k")
    assert _throughput(build_runs("ci")) == [("2k", 1), ("2k", 16), ("2k", 64)]


def test_empty_env_vars_keep_preset(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_ISLS", "")
    monkeypatch.setenv("SPEC_DECODE_CONCURRENCIES", "")
    assert _throughput(build_runs("ci")) == [("32k", 1), ("32k", 16), ("32k", 64)]


def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_ISLS", "32k")
    monkeypatch.setenv("SPEC_DECODE_CONCURRENCIES", "64")
    runs = build_runs("throughput", isls="1k", concurrencies="8")
    assert _throughput(runs) == [("1k", 8)]


def test_invalid_env_var_rejected(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_ISLS", "4k")
    with pytest.raises(ValueError, match="ISL bucket"):
        build_runs("full")


def test_env_concurrencies_ignore_max_concurrency(monkeypatch):
    monkeypatch.setenv("SPEC_DECODE_CONCURRENCIES", "16,64")
    runs = build_runs("ci", max_concurrency=8)
    assert _throughput(runs) == [("32k", 16), ("32k", 64)]


def test_max_concurrency_caps_preset_sweep():
    assert _throughput(build_runs("ci", max_concurrency=8)) == [("32k", 1), ("32k", 8)]
    assert _throughput(build_runs("ci", max_concurrency=128)) == [
        ("32k", 1),
        ("32k", 16),
        ("32k", 64),
    ]


def test_explicit_concurrencies_ignore_max_concurrency():
    runs = build_runs("ci", concurrencies="16,64", max_concurrency=8)
    assert _throughput(runs) == [("32k", 16), ("32k", 64)]


def test_parse_unset_returns_none():
    assert parse_isls(None) is None
    assert parse_isls("") is None
    assert parse_concurrencies(None) is None


def test_parse_isls_normalizes_and_dedupes():
    assert parse_isls("32K,1k,1k") == ("1k", "32k")


@pytest.mark.parametrize("value", ["4k", "1k,64k", ","])
def test_parse_isls_rejects_unknown_buckets(value):
    with pytest.raises(ValueError, match="ISL bucket"):
        parse_isls(value)


@pytest.mark.parametrize("value", ["0", "-1", "a,8", "1.5", ","])
def test_parse_concurrencies_rejects_invalid(value):
    with pytest.raises(ValueError, match="concurrencies"):
        parse_concurrencies(value)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="Unknown spec-decode preset"):
        build_runs("bogus")
