# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import importlib.util
import pathlib

from config.constants import DeviceIds, DeviceTypes, ModelConfigs, ModelRunners


def _real_settings_module():
    """Load config/settings.py straight from disk.

    Several test modules replace ``sys.modules["config.settings"]`` with a Mock
    at import time and never restore it, so a plain import hands back a Mock
    depending on collection order and every assertion below would pass
    vacuously.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.py"
    spec = importlib.util.spec_from_file_location("_real_config_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_settings_module = _real_settings_module()
audio_chunk_duration_for_worker_count = (
    _settings_module.audio_chunk_duration_for_worker_count
)
Settings = _settings_module.Settings


def test_chunk_duration_scales_with_worker_count():
    assert audio_chunk_duration_for_worker_count(1) == 30
    assert audio_chunk_duration_for_worker_count(3) == 30
    assert audio_chunk_duration_for_worker_count(4) == 15
    assert audio_chunk_duration_for_worker_count(7) == 15
    assert audio_chunk_duration_for_worker_count(8) == 10
    assert audio_chunk_duration_for_worker_count(32) == 10


def test_device_ids_32_counts_as_32_workers():
    worker_count = len(DeviceIds.DEVICE_IDS_32.value.replace(" ", "").split("),("))
    assert worker_count == 32
    assert audio_chunk_duration_for_worker_count(worker_count) == 10


def test_qwen3_asr_galaxy_keeps_throttle_zero_and_32_workers():
    cfg = ModelConfigs[(ModelRunners.TT_QWEN3_ASR, DeviceTypes.GALAXY)]
    assert cfg.get("default_throttle_level") == 0
    assert cfg["device_ids"] == DeviceIds.DEVICE_IDS_32.value


def test_audio_min_split_duration_default_is_single_runner_window():
    """Clips at/under this length stay whole (measured: 3s fan-out chunking
    costs +2.4 WER on short librispeech clips vs keeping them whole)."""
    assert Settings().audio_min_split_duration_seconds == 30


def test_qwen3_asr_duration_cap_is_one_full_galaxy_wave():
    """Cap = 32 runners * 10s chunk = 320s, so a single request can fill all 32
    runners in one pass before spilling into extra waves."""
    s = Settings()
    workers = 32
    assert s.max_audio_duration_qwen3_asr_seconds == (
        workers * audio_chunk_duration_for_worker_count(workers)
    )
    assert s.max_audio_duration_qwen3_asr_seconds == 320
