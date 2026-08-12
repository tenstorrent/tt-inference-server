# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reusable structural and CLIP metrics for short generated videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SAMPLE_COUNT = 8
FRAME_METRIC_SIZE = (224, 224)
BLACK_BRIGHTNESS_THRESHOLD = 2.0
FLAT_VARIANCE_THRESHOLD = 4.0
FROZEN_DELTA_THRESHOLD = 0.5
INVALID_FRAME_FRACTION = 0.8


class MissingVideoQualityDependency(RuntimeError):
    """Raised when optional quality-evaluation dependencies are unavailable."""


@dataclass(frozen=True)
class SampledVideoFrames:
    """Uniformly sampled RGB frames and source decode counts."""

    frames: tuple[Any, ...]
    total_decoded_frames: int
    sampled_indices: tuple[int, ...]


class BatchedCLIPScorer:
    """Batch image/text scorer using the same OpenCLIP defaults as CLIPEncoder."""

    def __init__(
        self,
        *,
        clip_version: str = "ViT-B/32",
        pretrained: str = "openai",
        cache_dir: str | None = None,
    ) -> None:
        try:
            import open_clip  # pyright: ignore[reportMissingImports]
            import torch  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise MissingVideoQualityDependency(
                "CLIP scoring requires torch and open_clip_torch"
            ) from exc

        self._open_clip = open_clip
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_version,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )
        self.model.eval()

    def encode_images(self, frames: Iterable[Any]) -> Any:
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("at least one frame is required")
        image_batch = self._torch.stack(
            [self.preprocess(frame) for frame in frame_list]
        )
        with self._torch.no_grad():
            features = self.model.encode_image(image_batch).float()
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_texts(self, texts: Iterable[str]) -> Any:
        text_list = list(texts)
        if not text_list:
            raise ValueError("at least one text is required")
        tokens = self._open_clip.tokenize(text_list)
        with self._torch.no_grad():
            features = self.model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def score(
        self,
        *,
        frames: Iterable[Any],
        texts: Iterable[str],
    ) -> tuple[Any, Any]:
        """Return ``(similarity*100, image_features)`` for frames × texts."""

        image_features = self.encode_images(frames)
        text_features = self.encode_texts(texts)
        similarities = (image_features @ text_features.T) * 100
        return similarities.cpu().numpy(), image_features

    def consecutive_image_similarity(self, image_features: Any) -> list[float]:
        if image_features.shape[0] < 2:
            return []
        similarities = (image_features[:-1] * image_features[1:]).sum(dim=-1) * 100
        return [float(value) for value in similarities.cpu().tolist()]


def probe_video(
    video_path: Path,
    *,
    expected_duration: float,
    expected_ratio: str,
    duration_tolerance: float = 0.5,
    ratio_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Use ffprobe to verify that a downloaded output is valid video media."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise MissingVideoQualityDependency("ffprobe is required for video metadata")

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(result.stdout)
        video_stream = next(
            stream
            for stream in metadata["streams"]
            if stream.get("codec_type") == "video"
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(metadata["format"]["duration"])
    except (
        KeyError,
        TypeError,
        ValueError,
        StopIteration,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ValueError(f"video is not decodable by ffprobe: {exc}") from exc

    ratio_parts = expected_ratio.split(":", 1)
    if len(ratio_parts) != 2:
        raise ValueError(f"invalid expected ratio {expected_ratio!r}")
    expected_ratio_value = float(ratio_parts[0]) / float(ratio_parts[1])
    actual_ratio = width / height if height else 0.0
    ratio_error = abs(actual_ratio - expected_ratio_value) / expected_ratio_value

    return {
        "valid": (
            width > 0
            and height > 0
            and abs(duration - expected_duration) <= duration_tolerance
            and ratio_error <= ratio_tolerance
        ),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "duration_error_seconds": abs(duration - expected_duration),
        "aspect_ratio": actual_ratio,
        "aspect_ratio_error": ratio_error,
        "codec": video_stream.get("codec_name"),
        "frame_rate": video_stream.get("r_frame_rate"),
        "frame_count": video_stream.get("nb_frames"),
    }


def sample_video_frames(
    video_path: Path,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> SampledVideoFrames:
    """Decode a short video and uniformly select resized RGB frames."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise MissingVideoQualityDependency(
            "frame decoding requires imageio, imageio-ffmpeg, and Pillow"
        ) from exc

    decoded = []
    try:
        for frame in iio.imiter(video_path):
            decoded.append(
                Image.fromarray(frame).convert("RGB").resize(FRAME_METRIC_SIZE)
            )
    except Exception as exc:  # noqa: BLE001 - decoder backends expose many errors
        raise ValueError(f"video frame decoding failed: {exc}") from exc

    total = len(decoded)
    if total < 2:
        raise ValueError(f"video contains only {total} decodable frame(s)")

    selected_count = min(sample_count, total)
    indices = tuple(
        round(index * (total - 1) / (selected_count - 1))
        for index in range(selected_count)
    )
    return SampledVideoFrames(
        frames=tuple(decoded[index] for index in indices),
        total_decoded_frames=total,
        sampled_indices=indices,
    )


def structural_frame_metrics(sampled: SampledVideoFrames) -> dict[str, Any]:
    """Calculate brightness, flatness, frozen-frame, and motion diagnostics."""

    try:
        import numpy as np  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise MissingVideoQualityDependency(
            "structural video metrics require numpy"
        ) from exc

    arrays = [np.asarray(frame, dtype=np.float32) for frame in sampled.frames]
    brightness = [float(array.mean()) for array in arrays]
    variances = [float(array.var()) for array in arrays]
    deltas = [
        float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])
    ]

    black_fraction = sum(
        value < BLACK_BRIGHTNESS_THRESHOLD for value in brightness
    ) / len(brightness)
    flat_fraction = sum(value < FLAT_VARIANCE_THRESHOLD for value in variances) / len(
        variances
    )
    frozen_fraction = (
        sum(value < FROZEN_DELTA_THRESHOLD for value in deltas) / len(deltas)
        if deltas
        else 1.0
    )
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    median_delta = float(np.median(deltas)) if deltas else 0.0
    p90_delta = float(np.percentile(deltas, 90)) if deltas else 0.0
    delta_stddev = float(np.std(deltas)) if deltas else 0.0
    motion_coefficient_of_variation = (
        delta_stddev / mean_delta if mean_delta > 0 else 0.0
    )
    sudden_change_threshold = max(10.0, median_delta * 4)
    sudden_change_count = sum(value > sudden_change_threshold for value in deltas)

    return {
        "sampled_frame_count": len(arrays),
        "total_decoded_frames": sampled.total_decoded_frames,
        "sampled_indices": list(sampled.sampled_indices),
        "average_brightness": float(np.mean(brightness)),
        "minimum_brightness": min(brightness),
        "average_pixel_variance": float(np.mean(variances)),
        "black_frame_fraction": black_fraction,
        "flat_frame_fraction": flat_fraction,
        "mean_frame_delta": mean_delta,
        "median_frame_delta": median_delta,
        "p90_frame_delta": p90_delta,
        "frame_delta_stddev": delta_stddev,
        "motion_coefficient_of_variation": motion_coefficient_of_variation,
        "sudden_change_count": sudden_change_count,
        "frozen_pair_fraction": frozen_fraction,
        "is_black": black_fraction >= INVALID_FRAME_FRACTION,
        "is_flat": flat_fraction >= INVALID_FRAME_FRACTION,
        "is_frozen": frozen_fraction >= INVALID_FRAME_FRACTION,
    }


def clip_video_metrics(
    *,
    scorer: BatchedCLIPScorer,
    sampled: SampledVideoFrames,
    prompt: str,
    required_concepts: Iterable[str] = (),
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> dict[str, Any]:
    """Score whole-prompt alignment, concept coverage, and progression."""

    try:
        import numpy as np  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise MissingVideoQualityDependency("CLIP aggregation requires numpy") from exc

    concepts = tuple(required_concepts)
    texts = [prompt, *concepts]
    start_index: int | None = None
    end_index: int | None = None
    if expected_start:
        start_index = len(texts)
        texts.append(expected_start)
    if expected_end:
        end_index = len(texts)
        texts.append(expected_end)

    similarities, image_features = scorer.score(
        frames=sampled.frames,
        texts=texts,
    )
    prompt_scores = similarities[:, 0]
    concept_metrics = {}
    for index, concept in enumerate(concepts, start=1):
        scores = similarities[:, index]
        concept_metrics[concept] = {
            "mean": float(np.mean(scores)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
        }

    progression = None
    if start_index is not None and end_index is not None:
        boundary = max(1, len(sampled.frames) // 3)
        early_start = float(np.mean(similarities[:boundary, start_index]))
        late_start = float(np.mean(similarities[-boundary:, start_index]))
        early_end = float(np.mean(similarities[:boundary, end_index]))
        late_end = float(np.mean(similarities[-boundary:, end_index]))
        progression = {
            "early_start_score": early_start,
            "late_start_score": late_start,
            "early_end_score": early_end,
            "late_end_score": late_end,
            "progression_margin": ((early_start - late_start) + (late_end - early_end))
            / 2,
        }

    image_similarities = scorer.consecutive_image_similarity(image_features)
    return {
        "prompt_scores": [float(value) for value in prompt_scores.tolist()],
        "mean_prompt_score": float(np.mean(prompt_scores)),
        "minimum_prompt_score": float(np.min(prompt_scores)),
        "p10_prompt_score": float(np.percentile(prompt_scores, 10)),
        "prompt_score_stddev": float(np.std(prompt_scores)),
        "concept_scores": concept_metrics,
        "consecutive_image_similarity": image_similarities,
        "mean_image_consistency": (
            float(np.mean(image_similarities)) if image_similarities else None
        ),
        "progression": progression,
    }


def analyze_video_quality(
    video_path: Path,
    *,
    prompt: str,
    required_concepts: Iterable[str],
    expected_start: str | None,
    expected_end: str | None,
    expected_duration: float,
    expected_ratio: str,
    sample_count: int,
    clip_scorer: BatchedCLIPScorer | None,
) -> dict[str, Any]:
    """Run all enabled quality metrics for one downloaded video."""

    probe = probe_video(
        video_path,
        expected_duration=expected_duration,
        expected_ratio=expected_ratio,
    )
    sampled = sample_video_frames(video_path, sample_count=sample_count)
    structural = structural_frame_metrics(sampled)
    clip = (
        clip_video_metrics(
            scorer=clip_scorer,
            sampled=sampled,
            prompt=prompt,
            required_concepts=required_concepts,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        if clip_scorer is not None
        else None
    )
    return {
        "probe": probe,
        "structural": structural,
        "clip": clip,
        "valid_video": bool(probe["valid"])
        and not structural["is_black"]
        and not structural["is_flat"],
    }


__all__ = [
    "BatchedCLIPScorer",
    "MissingVideoQualityDependency",
    "SampledVideoFrames",
    "analyze_video_quality",
    "clip_video_metrics",
    "probe_video",
    "sample_video_frames",
    "structural_frame_metrics",
]
