#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from types import SimpleNamespace

import utils.prompt_client as prompt_client_module
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


class _StalledCacheMonitor:
    def get_cache_generation_status(self):
        return SimpleNamespace(
            is_generating=True,
            is_stalled=True,
            no_progress_duration=181.0,
            cache_dir=Path("/tmp/tensor-cache"),
            file_count=1,
            total_size_bytes=1024,
        )

    def get_effective_timeout(self, default_timeout, cache_status=None):
        return 3600.0


def test_wait_for_healthy_aborts_when_tensor_cache_stalls(monkeypatch):
    prompt_client = PromptClient(EnvironmentConfig())
    prompt_client.cache_monitor = _StalledCacheMonitor()

    def fail_health_check(*args, **kwargs):
        raise AssertionError(
            "Health check should not run after a stalled cache is detected"
        )

    monkeypatch.setattr(prompt_client_module.requests, "get", fail_health_check)

    assert prompt_client.wait_for_healthy(timeout=1200.0, interval=1) is False
