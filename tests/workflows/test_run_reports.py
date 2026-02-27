#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from workflows.run_reports import add_target_checks_audio, add_target_checks_video


class TestAddTargetChecksAudio:
    def test_with_full_metrics(self):
        """Returns correct values when all keys are present."""
        metrics = {
            "functional_ttft": 4000,
            "functional_ttft_ratio": 0.5,
            "functional_ttft_check": 2,
            "complete_ttft": 800,
            "complete_ttft_ratio": 0.5,
            "complete_ttft_check": 2,
            "target_ttft": 400,
            "target_ttft_ratio": 0.5,
            "target_ttft_check": 2,
        }
        result = add_target_checks_audio(metrics)
        assert result["functional"]["ttft"] == 4000
        assert result["functional"]["ttft_ratio"] == 0.5
        assert result["functional"]["ttft_check"] == 2
        assert result["functional"]["tput_check"] == 1
        assert result["target"]["ttft"] == 400
        assert result["target"]["ttft_check"] == 2

    def test_with_empty_metrics_does_not_raise(self):
        """Must not raise KeyError when targets were not configured (e.g. missing device in reference JSON)."""
        result = add_target_checks_audio({})
        assert result["functional"]["ttft"] is None
        assert result["functional"]["ttft_ratio"] == "Undefined"
        assert result["functional"]["ttft_check"] == "Undefined"
        assert result["functional"]["tput_check"] == 1
        assert result["complete"]["ttft"] is None
        assert result["complete"]["ttft_ratio"] == "Undefined"
        assert result["target"]["ttft"] is None
        assert result["target"]["ttft_ratio"] == "Undefined"
        assert result["target"]["tput_check"] == 1


class TestAddTargetChecksVideo:
    def test_with_empty_metrics_does_not_raise(self):
        """Must not raise KeyError when metrics dict is empty."""
        result = add_target_checks_video({})
        assert result["functional"]["concurrency"] is None
        assert result["functional"]["concurrency_ratio"] == "Undefined"
        assert result["functional"]["concurrency_check"] == "Undefined"
        assert result["complete"]["concurrency"] is None
        assert result["target"]["concurrency"] is None
