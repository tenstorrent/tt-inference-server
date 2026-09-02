# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from config.constants import DeviceIds, DeviceTypes, ModelConfigs, ModelRunners
from config.settings import audio_chunk_duration_for_worker_count


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
    from config.settings import Settings

    assert Settings().audio_min_split_duration_seconds == 30


def test_qwen3_asr_duration_cap_is_one_full_galaxy_wave():
    """Cap = 32 runners * 10s chunk = 320s, so a single request can fill all 32
    runners in one pass before spilling into extra waves."""
    from config.settings import Settings

    s = Settings()
    workers = 32
    assert s.max_audio_duration_qwen3_asr_seconds == (
        workers * audio_chunk_duration_for_worker_count(workers)
    )
    assert s.max_audio_duration_qwen3_asr_seconds == 320
