# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from tests.server_tests.base_test import BaseTest
from tests.test_config import TestConfig

logger = logging.getLogger(__name__)

# Use same dataset directory as FVD test for consistency
DATASET_DIR = "tests/server_tests/datasets/video_fvd_subset"
FVMD_RESULTS_FILE = "fvmd_results.json"

# Motion feature histogram configuration
NUM_VELOCITY_BINS = 32
NUM_ACCELERATION_BINS = 32
NUM_DIRECTION_BINS = 8


@dataclass
class VideoFVMDTestRequest:
    action: Literal["download", "compute_fvmd", "full_eval"]
    server_url: str | None = None
    reference_videos_path: str | None = None
    generated_videos_path: str | None = None
    download_count: int = 2
    # HuggingFace FineVideo dataset config (same as FVD)
    hf_dataset_name: str = "HuggingFaceFV/finevideo"
    hf_split: str = "train"
    category: str = "Sports"  # Sports has good motion for FVMD


class VideoFVMDTest(BaseTest):
    """
    Fréchet Video Motion Distance (FVMD) evaluation test.

    FVMD measures temporal consistency by analyzing motion patterns in videos.
    It computes velocity and acceleration fields from frame-to-frame motion,
    aggregates them into statistical histograms, and measures the Fréchet
    distance between reference and generated video motion distributions.

    This implementation uses optical flow estimation for motion tracking.
    True FVMD uses PIPs++ for keypoint tracking, but optical flow provides
    a reasonable approximation without additional dependencies.

    FVMD assesses motion consistency by analyzing:
    - Speed patterns (magnitude of velocity)
    - Acceleration patterns (change in velocity)
    - Motion direction distributions

    Lower scores indicate better temporal consistency (0 = identical motion).

    Reference: FVMD metric from video generation evaluation literature.
    """

    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self):
        logger.info("Running VideoFVMDTest")
        request = self.targets.get("request")
        logger.info("Running VideoFVMDTest with request: %s", request)

        if not isinstance(request, VideoFVMDTestRequest):
            return {
                "success": False,
                "error": "VideoFVMDTestRequest not provided in targets",
            }

        if request.action == "download":
            logger.info(
                "Downloading %s video samples from category '%s'",
                request.download_count,
                request.category,
            )
            self._download_video_samples(
                count=request.download_count,
                category=request.category,
                dataset_name=request.hf_dataset_name,
            )
            return {
                "success": True,
                "action": "download",
                "count": request.download_count,
                "category": request.category,
            }

        elif request.action == "compute_fvmd":
            logger.info("Computing FVMD score")
            fvmd_score = self._compute_fvmd(
                reference_path=request.reference_videos_path,
                generated_path=request.generated_videos_path,
            )
            return {
                "success": True,
                "action": "compute_fvmd",
                "fvmd_score": fvmd_score,
            }

        elif request.action == "full_eval":
            logger.info("Running full FVMD evaluation pipeline")
            self._download_video_samples(
                count=request.download_count,
                category=request.category,
                dataset_name=request.hf_dataset_name,
            )
            fvmd_score = self._compute_fvmd(
                reference_path=request.reference_videos_path,
                generated_path=request.generated_videos_path,
            )
            self.eval_results["fvmd"] = fvmd_score
            return {
                "success": True,
                "action": "full_eval",
                "eval_results": self.eval_results,
            }

        return {"success": False, "error": f"Unknown action: {request.action}"}

    def _download_video_samples(
        self,
        count: int = 2,
        category: str = "Sports",
        dataset_name: str = "HuggingFaceFV/finevideo",
    ) -> None:
        """
        Download video samples from HuggingFace FineVideo dataset.

        Same implementation as FVD test - downloads to shared dataset directory.
        Sports category is recommended for FVMD as it has significant motion.

        Reference: https://huggingface.co/datasets/HuggingFaceFV/finevideo
        """
        from datasets import load_dataset

        logger.info(
            f"Downloading {count} videos from '{category}' category "
            f"(dataset: {dataset_name})"
        )

        output_path = Path(DATASET_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        videos_dir = output_path / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(dataset_name, split="train", streaming=True)

        def is_target_category(sample: dict) -> bool:
            sample_json = sample.get("json", {})
            if isinstance(sample_json, dict):
                return sample_json.get("content_parent_category") == category
            return False

        filtered_dataset = filter(is_target_category, dataset)

        downloaded_metadata = []
        for idx, sample in enumerate(filtered_dataset):
            if idx >= count:
                break

            sample_json = sample.get("json", {})
            video_bytes = sample.get("mp4")

            if video_bytes is None:
                logger.warning(f"Sample {idx} has no video data, skipping")
                continue

            video_title = sample_json.get("youtube_title", f"video_{idx}")
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in video_title
            )[:50]
            filename = f"{idx:03d}_{safe_title}.mp4"

            video_path = videos_dir / filename
            with video_path.open("wb") as video_file:
                video_file.write(video_bytes)

            meta_path = metadata_dir / f"{idx:03d}_metadata.json"
            with meta_path.open("w", encoding="utf-8") as meta_file:
                json.dump(sample_json, meta_file, indent=2)

            downloaded_metadata.append(
                {
                    "index": idx,
                    "filename": filename,
                    "category": category,
                    "youtube_title": sample_json.get("youtube_title"),
                    "duration_seconds": sample_json.get("duration_seconds"),
                    "resolution": sample_json.get("resolution"),
                }
            )

            logger.info(f"Downloaded [{idx + 1}/{count}]: {filename}")

        combined_metadata = {
            "dataset": dataset_name,
            "category": category,
            "count": len(downloaded_metadata),
            "videos": downloaded_metadata,
        }

        metadata_path = output_path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2)

        logger.info(
            f"Downloaded {len(downloaded_metadata)} videos from '{category}' "
            f"category to {output_path}"
        )

    def _load_videos_as_tensors(
        self,
        video_dir: Path,
        num_frames: int = 32,  # More frames for motion analysis
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Load videos from directory and convert to tensor format.

        Uses more frames than FVD for better motion analysis.

        Returns:
            Array of shape (num_videos, num_frames, height, width, channels)
        """
        import imageio.v3 as iio

        logger.info(f"Loading videos from {video_dir}")

        video_files = sorted(video_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"No MP4 files found in {video_dir}")

        logger.info(f"Found {len(video_files)} video files")

        all_videos = []
        for video_path in video_files:
            try:
                frames = self._load_single_video(
                    video_path, num_frames, target_size, iio
                )
                all_videos.append(frames)
                logger.info(f"Loaded {video_path.name}: {frames.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {video_path.name}: {e}")
                continue

        if not all_videos:
            raise RuntimeError("No videos could be loaded successfully")

        videos_tensor = np.stack(all_videos, axis=0)
        logger.info(f"Loaded {len(all_videos)} videos, shape: {videos_tensor.shape}")

        return videos_tensor

    def _load_single_video(
        self,
        video_path: Path,
        num_frames: int,
        target_size: tuple[int, int],
        iio,
    ) -> np.ndarray:
        """Load a single video and sample frames uniformly."""
        all_frames = iio.imread(video_path, plugin="pyav")
        total_frames = len(all_frames)

        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")

        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (num_frames - total_frames))

        sampled_frames = []
        height, width = target_size

        for idx in indices:
            frame = all_frames[idx]
            pil_frame = Image.fromarray(frame)
            pil_frame = pil_frame.resize((width, height), Image.Resampling.BILINEAR)
            resized_frame = np.array(pil_frame)

            if len(resized_frame.shape) == 2:
                resized_frame = np.stack([resized_frame] * 3, axis=-1)
            elif resized_frame.shape[-1] == 4:
                resized_frame = resized_frame[:, :, :3]

            sampled_frames.append(resized_frame)

        return np.stack(sampled_frames, axis=0)

    def _compute_optical_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between two consecutive frames.

        Uses gradient-based Lucas-Kanade style estimation without OpenCV.
        Returns flow in x and y directions.

        Args:
            frame1: First frame (H, W, C) or (H, W) grayscale
            frame2: Second frame (H, W, C) or (H, W) grayscale

        Returns:
            Tuple of (flow_x, flow_y) each of shape (H, W)
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = np.mean(frame1.astype(np.float32), axis=-1)
        else:
            gray1 = frame1.astype(np.float32)

        if len(frame2.shape) == 3:
            gray2 = np.mean(frame2.astype(np.float32), axis=-1)
        else:
            gray2 = frame2.astype(np.float32)

        # Compute spatial gradients using Sobel-like kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        # Use average of both frames for gradient computation
        avg_frame = (gray1 + gray2) / 2.0

        # Compute gradients via convolution
        from scipy.ndimage import convolve

        Ix = convolve(avg_frame, kernel_x)
        Iy = convolve(avg_frame, kernel_y)
        It = gray2 - gray1  # Temporal gradient

        # Lucas-Kanade: solve for flow using local window
        # Simplified: use pseudo-inverse approach
        window_size = 5
        half_win = window_size // 2

        h, w = gray1.shape
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)

        # Downsample for efficiency (compute flow at sparse points)
        stride = 4
        for y in range(half_win, h - half_win, stride):
            for x in range(half_win, w - half_win, stride):
                # Extract local window
                Ix_win = Ix[
                    y - half_win : y + half_win + 1, x - half_win : x + half_win + 1
                ].flatten()
                Iy_win = Iy[
                    y - half_win : y + half_win + 1, x - half_win : x + half_win + 1
                ].flatten()
                It_win = It[
                    y - half_win : y + half_win + 1, x - half_win : x + half_win + 1
                ].flatten()

                # Build system: A @ [vx, vy]^T = -b
                A = np.column_stack([Ix_win, Iy_win])
                b = It_win

                # Solve using pseudo-inverse with regularization
                AtA = A.T @ A + 1e-6 * np.eye(2)
                Atb = A.T @ b

                try:
                    flow = np.linalg.solve(AtA, -Atb)
                    # Fill surrounding area
                    flow_x[
                        y - stride // 2 : y + stride // 2,
                        x - stride // 2 : x + stride // 2,
                    ] = flow[0]
                    flow_y[
                        y - stride // 2 : y + stride // 2,
                        x - stride // 2 : x + stride // 2,
                    ] = flow[1]
                except np.linalg.LinAlgError:
                    pass

        return flow_x, flow_y

    def _extract_motion_features(self, videos: np.ndarray) -> np.ndarray:
        """
        Extract motion features from videos using optical flow.

        For each video:
        1. Compute optical flow between consecutive frames
        2. Calculate velocity (flow magnitude) and direction
        3. Calculate acceleration (change in velocity)
        4. Build histograms of these motion features
        5. Concatenate into a single feature vector

        Args:
            videos: Video tensor of shape (N, T, H, W, C)

        Returns:
            Motion features of shape (N, feature_dim)
        """
        logger.info(f"Extracting motion features from {videos.shape[0]} videos")

        num_videos, num_frames, _, _, _ = videos.shape
        all_features = []

        for video_idx in range(num_videos):
            video = videos[video_idx]

            # Compute optical flow for all frame pairs
            velocities_x = []
            velocities_y = []

            for t in range(num_frames - 1):
                frame1 = video[t]
                frame2 = video[t + 1]
                flow_x, flow_y = self._compute_optical_flow(frame1, frame2)
                velocities_x.append(flow_x)
                velocities_y.append(flow_y)

            velocities_x = np.array(velocities_x)
            velocities_y = np.array(velocities_y)

            # Compute velocity magnitude and direction
            velocity_magnitude = np.sqrt(velocities_x**2 + velocities_y**2)
            velocity_direction = np.arctan2(velocities_y, velocities_x)

            # Compute acceleration (change in velocity)
            accel_x = np.diff(velocities_x, axis=0)
            accel_y = np.diff(velocities_y, axis=0)
            acceleration_magnitude = np.sqrt(accel_x**2 + accel_y**2)

            # Build feature histograms
            features = self._build_motion_histograms(
                velocity_magnitude, velocity_direction, acceleration_magnitude
            )

            all_features.append(features)
            logger.info(
                f"Processed video {video_idx + 1}/{num_videos}, "
                f"feature dim: {features.shape[0]}"
            )

        features_array = np.array(all_features)
        logger.info(f"Extracted motion features shape: {features_array.shape}")

        return features_array

    def _build_motion_histograms(
        self,
        velocity_magnitude: np.ndarray,
        velocity_direction: np.ndarray,
        acceleration_magnitude: np.ndarray,
    ) -> np.ndarray:
        """
        Build statistical histograms from motion fields.

        Creates normalized histograms for:
        - Velocity magnitude distribution
        - Velocity direction distribution
        - Acceleration magnitude distribution
        - Joint velocity-direction histogram

        Args:
            velocity_magnitude: (T-1, H, W) velocity magnitudes
            velocity_direction: (T-1, H, W) velocity directions in radians
            acceleration_magnitude: (T-2, H, W) acceleration magnitudes

        Returns:
            Concatenated histogram features (1D array)
        """
        features = []

        # Velocity magnitude histogram
        vel_mag_flat = velocity_magnitude.flatten()
        vel_mag_flat = vel_mag_flat[np.isfinite(vel_mag_flat)]
        if len(vel_mag_flat) > 0:
            # Use percentile-based bins to handle outliers
            max_vel = np.percentile(vel_mag_flat, 99)
            vel_hist, _ = np.histogram(
                vel_mag_flat,
                bins=NUM_VELOCITY_BINS,
                range=(0, max(max_vel, 1e-6)),
                density=True,
            )
        else:
            vel_hist = np.zeros(NUM_VELOCITY_BINS)
        features.append(vel_hist)

        # Velocity direction histogram (circular)
        vel_dir_flat = velocity_direction.flatten()
        vel_dir_flat = vel_dir_flat[np.isfinite(vel_dir_flat)]
        if len(vel_dir_flat) > 0:
            dir_hist, _ = np.histogram(
                vel_dir_flat,
                bins=NUM_DIRECTION_BINS,
                range=(-np.pi, np.pi),
                density=True,
            )
        else:
            dir_hist = np.zeros(NUM_DIRECTION_BINS)
        features.append(dir_hist)

        # Acceleration magnitude histogram
        accel_flat = acceleration_magnitude.flatten()
        accel_flat = accel_flat[np.isfinite(accel_flat)]
        if len(accel_flat) > 0:
            max_accel = np.percentile(accel_flat, 99)
            accel_hist, _ = np.histogram(
                accel_flat,
                bins=NUM_ACCELERATION_BINS,
                range=(0, max(max_accel, 1e-6)),
                density=True,
            )
        else:
            accel_hist = np.zeros(NUM_ACCELERATION_BINS)
        features.append(accel_hist)

        # Additional statistics: mean, std, percentiles
        stats = []
        for arr in [vel_mag_flat, accel_flat]:
            if len(arr) > 0:
                stats.extend(
                    [
                        np.mean(arr),
                        np.std(arr),
                        np.percentile(arr, 25),
                        np.percentile(arr, 50),
                        np.percentile(arr, 75),
                    ]
                )
            else:
                stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        features.append(np.array(stats))

        return np.concatenate(features)

    def _calculate_activation_statistics(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of motion features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        # Handle single sample case
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        return mu, sigma

    def _compute_fvmd(
        self,
        reference_path: str | None = None,
        generated_path: str | None = None,
    ) -> float:
        """
        Compute Fréchet Video Motion Distance between reference and generated videos.

        FVMD measures temporal consistency by comparing motion feature distributions.
        Motion features include velocity and acceleration histograms computed from
        optical flow between consecutive frames.

        Lower score indicates better motion consistency (0 = identical motion).
        """
        from utils.sdxl_accuracy_utils.fid_score import calculate_frechet_distance

        ref_path = (
            Path(reference_path) if reference_path else Path(DATASET_DIR) / "videos"
        )
        gen_path = (
            Path(generated_path) if generated_path else Path(DATASET_DIR) / "videos"
        )

        logger.info(f"Computing FVMD: {ref_path} vs {gen_path}")

        # Load videos with more frames for motion analysis
        reference_videos = self._load_videos_as_tensors(ref_path, num_frames=32)
        generated_videos = self._load_videos_as_tensors(gen_path, num_frames=32)

        # Extract motion features (velocity/acceleration histograms)
        ref_features = self._extract_motion_features(reference_videos)
        gen_features = self._extract_motion_features(generated_videos)

        # Calculate statistics
        mu_ref, sigma_ref = self._calculate_activation_statistics(ref_features)
        mu_gen, sigma_gen = self._calculate_activation_statistics(gen_features)

        # Calculate FVMD using Fréchet distance
        fvmd_score = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

        logger.info(f"FVMD Score: {fvmd_score:.4f}")

        # Save results
        results_path = Path(DATASET_DIR) / FVMD_RESULTS_FILE
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "fvmd_score": float(fvmd_score),
                    "feature_dim": ref_features.shape[1]
                    if ref_features.ndim > 1
                    else 1,
                    "num_reference_videos": len(ref_features),
                    "num_generated_videos": len(gen_features),
                },
                f,
                indent=2,
            )

        return float(fvmd_score)
