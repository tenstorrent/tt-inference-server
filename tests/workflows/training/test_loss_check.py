# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest

from workflows.training.loss_check import (
    LossCheckConfig,
    LossExpectation,
    evaluate,
    index_metrics,
    parse_config,
)


def _metrics(*pairs):
    """pairs: (step, metric_name, value)."""
    return [
        {"global_step": s, "metric_name": m, "value": v, "timestamp": 1.0}
        for (s, m, v) in pairs
    ]


class TestParseConfig:
    def test_full_config(self):
        raw = {
            "tolerances": {"rtol": 0.3, "atol": 0.2},
            "checks": {"require_decreasing": False, "final_train_loss_max": 4.0},
            "request": {"batch_size": 8, "max_steps": 15},
            "expected_losses": [
                {"global_step": 5, "metric_name": "train_loss", "value": 5.7},
                {"global_step": 10, "value": 4.6},  # metric_name defaults to train_loss
            ],
        }
        cfg = parse_config(raw)
        assert cfg.rtol == 0.3
        assert cfg.atol == 0.2
        assert cfg.require_decreasing is False
        assert cfg.final_train_loss_max == 4.0
        assert cfg.request == {"batch_size": 8, "max_steps": 15}
        assert cfg.expectations[0] == LossExpectation(5, "train_loss", 5.7)
        assert cfg.expectations[1] == LossExpectation(10, "train_loss", 4.6)

    def test_defaults(self):
        cfg = parse_config({})
        assert cfg.rtol == 0.5
        assert cfg.atol == 0.1
        assert cfg.require_decreasing is True
        assert cfg.final_train_loss_max is None
        assert cfg.expectations == []

    def test_missing_value_raises(self):
        with pytest.raises(ValueError):
            parse_config({"expected_losses": [{"global_step": 5}]})

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError):
            parse_config([1, 2, 3])


class TestIndexMetrics:
    def test_flattens_and_last_wins(self):
        indexed = index_metrics(
            _metrics((5, "train_loss", 5.7), (5, "train_loss", 5.5))
        )
        assert indexed[(5, "train_loss")] == 5.5

    def test_skips_malformed(self):
        indexed = index_metrics(
            [
                {"global_step": 5, "metric_name": "train_loss", "value": 5.7},
                {"metric_name": "train_loss", "value": 1.0},  # no step
                {"global_step": 10, "metric_name": "train_loss"},  # no value
            ]
        )
        assert indexed == {(5, "train_loss"): 5.7}


class TestEvaluate:
    def _cfg(self, **kw):
        base = dict(
            rtol=0.5,
            atol=0.1,
            require_decreasing=True,
            final_train_loss_max=None,
            expectations=[
                LossExpectation(5, "train_loss", 5.7),
                LossExpectation(10, "train_loss", 4.6),
            ],
        )
        base.update(kw)
        return LossCheckConfig(**base)

    def test_all_pass_within_tolerance(self):
        metrics = _metrics((5, "train_loss", 5.6), (10, "train_loss", 4.7))
        result = evaluate(metrics, self._cfg(), model="m", device="p150")
        assert result.passed is True
        assert all(r["status"] == "pass" for r in result.records)

    def test_out_of_tolerance_fails(self):
        metrics = _metrics((5, "train_loss", 20.0), (10, "train_loss", 4.7))
        result = evaluate(metrics, self._cfg(), model="m", device="p150")
        assert result.passed is False
        step5 = next(r for r in result.records if r["test_name"] == "train_loss@step5")
        assert step5["status"] == "fail"

    def test_missing_expected_metric_fails(self):
        metrics = _metrics((5, "train_loss", 5.6))  # step10 missing
        result = evaluate(metrics, self._cfg(), model="m", device="p150")
        step10 = next(r for r in result.records if r["test_name"] == "train_loss@step10")
        assert step10["status"] == "fail"
        assert "missing" in step10["description"]

    def test_no_metrics_fails(self):
        result = evaluate([], self._cfg(), model="m", device="p150")
        assert result.passed is False
        assert result.records[0]["test_name"] == "training_produced_metrics"

    def test_not_decreasing_fails(self):
        # loss goes UP from step5 to step10
        metrics = _metrics((5, "train_loss", 4.0), (10, "train_loss", 9.0))
        cfg = self._cfg(expectations=[])  # isolate the decreasing check
        result = evaluate(metrics, cfg, model="m", device="p150")
        dec = next(r for r in result.records if r["test_name"] == "train_loss_decreasing")
        assert dec["status"] == "fail"

    def test_decreasing_passes(self):
        metrics = _metrics((5, "train_loss", 9.0), (10, "train_loss", 4.0))
        cfg = self._cfg(expectations=[])
        result = evaluate(metrics, cfg, model="m", device="p150")
        dec = next(r for r in result.records if r["test_name"] == "train_loss_decreasing")
        assert dec["status"] == "pass"

    def test_final_threshold(self):
        metrics = _metrics((5, "train_loss", 9.0), (10, "train_loss", 6.0))
        cfg = self._cfg(expectations=[], final_train_loss_max=5.0)
        result = evaluate(metrics, cfg, model="m", device="p150")
        thr = next(
            r for r in result.records if r["test_name"] == "final_train_loss_threshold"
        )
        assert thr["status"] == "fail"

    def test_records_are_spec_test_shaped(self):
        metrics = _metrics((5, "train_loss", 5.6), (10, "train_loss", 4.7))
        result = evaluate(metrics, self._cfg(), model="mymodel", device="p150")
        for record in result.records:
            assert record["kind"] == "spec_tests"
            assert record["model"] == "mymodel"
            assert record["device"] == "p150"
            assert "test_name" in record and "attempts" in record
