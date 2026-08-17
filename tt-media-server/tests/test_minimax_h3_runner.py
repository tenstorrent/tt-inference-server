# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the MiniMax-H3 video+audio runner and its audio-mux export path.

Split by dependency so the audio/config tests run anywhere (no ttnn / no
hardware), while the runner-registration test that imports the tt_dit pipeline
stack runs inside the media-server container.
"""

import shutil
import subprocess

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config wiring (import-light: config.constants pulls no ttnn)
# ---------------------------------------------------------------------------


def test_minimax_h3_registered_in_service_and_name_maps():
    from config.constants import (
        INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        MODEL_SERVICE_RUNNER_MAP,
        ModelNames,
        ModelRunners,
        ModelServices,
    )

    assert ModelRunners.TT_MINIMAX_H3 in MODEL_SERVICE_RUNNER_MAP[ModelServices.VIDEO]
    assert INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP[ModelRunners.TT_MINIMAX_H3] == {
        ModelNames.MINIMAX_H3
    }


@pytest.mark.parametrize(
    "device_type_name, expected_mesh",
    [("P150X4", (1, 4)), ("P300X2", (2, 2)), ("GALAXY", (4, 8))],
)
def test_minimax_h3_model_configs_cover_1_to_many_blackhole(
    device_type_name, expected_mesh
):
    from config.constants import DeviceTypes, ModelConfigs, ModelRunners

    key = (ModelRunners.TT_MINIMAX_H3, DeviceTypes[device_type_name])
    assert key in ModelConfigs, f"missing ModelConfigs row for {device_type_name}"
    assert ModelConfigs[key]["device_mesh_shape"] == expected_mesh
    assert ModelConfigs[key]["max_batch_size"] == 1


def test_minimax_h3_target_resolution_scales_with_mesh():
    from config.constants import minimax_h3_target_resolution

    small = minimax_h3_target_resolution((1, 4))
    large = minimax_h3_target_resolution((4, 8))
    assert (large.height * large.width) > (small.height * small.width)


# ---------------------------------------------------------------------------
# Runner registration (imports the tt_dit pipeline stack — container only)
# ---------------------------------------------------------------------------


def test_minimax_h3_in_available_runners():
    ttnn = pytest.importorskip("ttnn")  # noqa: F841  (skips outside the container)
    from config.constants import ModelRunners
    from tt_model_runners.runner_fabric import AVAILABLE_RUNNERS

    assert ModelRunners.TT_MINIMAX_H3 in AVAILABLE_RUNNERS
    assert callable(AVAILABLE_RUNNERS[ModelRunners.TT_MINIMAX_H3])


# ---------------------------------------------------------------------------
# Audio mux (needs numpy + ffmpeg; no ttnn / no hardware)
# ---------------------------------------------------------------------------

_HAVE_FFMPEG = (
    shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
)
_needs_ffmpeg = pytest.mark.skipif(
    not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not present"
)


def _stream_types(path: str) -> list[str]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=nw=1:nk=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def _dummy_frames(n=8, h=64, w=64):
    return (np.random.rand(n, h, w, 3) * 255).astype(np.uint8)


def _dummy_stereo(sample_rate=32000, seconds=1):
    # Channels-first (2, N) float32 in [-1, 1] — MiniMax-H3's native audio shape.
    n = sample_rate * seconds
    t = np.linspace(0, seconds, n, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    return np.stack([left, right]).astype(np.float32)


@_needs_ffmpeg
def test_export_to_mp4_video_only_has_no_audio_stream():
    from utils.video_manager import VideoManager

    path = VideoManager().export_to_mp4(_dummy_frames(), fps=8)
    types = _stream_types(path)
    assert "video" in types
    assert "audio" not in types


@_needs_ffmpeg
def test_export_to_mp4_with_audio_muxes_aac_track():
    from utils.video_manager import VideoManager

    path = VideoManager().export_to_mp4(
        _dummy_frames(), fps=8, audio=_dummy_stereo(), sample_rate=32000
    )
    types = _stream_types(path)
    assert "video" in types
    assert "audio" in types


@_needs_ffmpeg
def test_export_to_mp4_pulls_audio_off_result_object():
    """An object exposing .frames/.audio/.sample_rate (VideoWithAudio-shaped)
    should be muxed without the caller passing audio explicitly."""
    from utils.video_manager import VideoManager

    class _Result:
        frames = _dummy_frames()
        audio = _dummy_stereo()
        sample_rate = 32000

    path = VideoManager().export_to_mp4(_Result(), fps=8)
    types = _stream_types(path)
    assert "video" in types
    assert "audio" in types


@_needs_ffmpeg
def test_write_wav_handles_mono_and_both_layouts():
    from utils.video_manager import _audio_to_int16_frames

    # channels-first stereo (2, N)
    frames, ch = _audio_to_int16_frames(_dummy_stereo())
    assert ch == 2 and frames.dtype == np.int16 and frames.shape[1] == 2
    # mono (N,)
    frames, ch = _audio_to_int16_frames(np.zeros(1000, dtype=np.float32))
    assert ch == 1 and frames.shape[1] == 1
    # samples-first stereo (N, 2)
    frames, ch = _audio_to_int16_frames(np.zeros((1000, 2), dtype=np.float32))
    assert ch == 2 and frames.shape == (1000, 2)
