#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import logging
from pathlib import Path
from types import SimpleNamespace

import utils.prompt_client as prompt_client_module
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


class _StalledCacheMonitor:
    def get_cache_generation_status(self):
        return SimpleNamespace(
            is_generating=True,
            is_first_run=False,
            is_stalled=True,
            no_progress_duration=181.0,
            cache_dir=Path("/tmp/tensor-cache"),
            file_count=1,
            total_size_bytes=1024,
        )

    def get_effective_timeout(self, default_timeout, cache_status=None):
        return 3600.0


class _GeneratingCacheMonitor:
    def __init__(self):
        self.marked_completed = False

    def get_cache_generation_status(self):
        return SimpleNamespace(
            is_generating=True,
            is_first_run=False,
            is_stalled=False,
            no_progress_duration=0.0,
            cache_dir=Path("/tmp/tensor-cache"),
            file_count=2,
            total_size_bytes=2048,
        )

    def get_effective_timeout(self, default_timeout, cache_status=None):
        return 3600.0

    def mark_cache_completed(self):
        self.marked_completed = True
        return True


class _ExistingCacheMonitor:
    def get_cache_generation_status(self):
        return SimpleNamespace(
            is_generating=False,
            is_first_run=False,
            has_existing_cache=True,
            is_stalled=False,
            no_progress_duration=0.0,
            cache_dir=Path("/tmp/tensor-cache"),
            file_count=4,
            total_size_bytes=8192,
        )

    def get_effective_timeout(self, default_timeout, cache_status=None):
        return float(default_timeout)


def test_wait_for_healthy_aborts_when_tensor_cache_stalls(monkeypatch):
    prompt_client = PromptClient(EnvironmentConfig())
    prompt_client.cache_monitor = _StalledCacheMonitor()

    def fail_health_check(*args, **kwargs):
        raise AssertionError(
            "Health check should not run after a stalled cache is detected"
        )

    monkeypatch.setattr(prompt_client_module.requests, "get", fail_health_check)

    assert prompt_client.wait_for_healthy(timeout=1200.0, interval=1) is False


def test_wait_for_healthy_marks_cache_complete_after_success(monkeypatch):
    prompt_client = PromptClient(EnvironmentConfig())
    prompt_client.cache_monitor = _GeneratingCacheMonitor()

    monkeypatch.setattr(
        prompt_client_module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=200),
    )

    assert prompt_client.wait_for_healthy(timeout=1200.0, interval=1) is True
    assert prompt_client.cache_monitor.marked_completed is True


def test_wait_for_healthy_logs_existing_tensor_cache_details(monkeypatch, caplog):
    prompt_client = PromptClient(EnvironmentConfig())
    prompt_client.cache_monitor = _ExistingCacheMonitor()

    monkeypatch.setattr(
        prompt_client_module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=200),
    )

    with caplog.at_level(logging.INFO):
        assert prompt_client.wait_for_healthy(timeout=1200.0, interval=1) is True

    assert (
        "Existing tensor cache detected - tracking 4 file(s), 8.0 KB; "
        "using standard timeout:=1200.0s" in caplog.text
    )
