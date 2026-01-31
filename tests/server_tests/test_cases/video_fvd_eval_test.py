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

DATASET_DIR = "tests/server_tests/datasets/video_fvd_subset"
FVD_RESULTS_FILE = "fvd_results.json"


# Available categories in FineVideo dataset (122 total, common ones listed)
FINEVIDEO_CATEGORIES = [
    "Sports",
    "Education",
    "Music",
    "Gaming",
    "Travel",
    "Food",
    "Science",
    "Technology",
    "Entertainment",
    "News",
]


@dataclass
class VideoFVDTestRequest:
    action: Literal["download", "compute_fvd", "full_eval"]
    server_url: str | None = None
    reference_videos_path: str | None = None
    generated_videos_path: str | None = None
    download_count: int = 2
    # HuggingFace FineVideo dataset config
    hf_dataset_name: str = "HuggingFaceFV/finevideo"
    hf_split: str = "train"
    category: str = "Travel"  # Filter by content_parent_category


class VideoFVDTest(BaseTest):
    """
    Fréchet Video Distance (FVD) evaluation test.

    This implementation uses InceptionV3 for frame-level feature extraction,
    reusing the existing fid_score.py infrastructure. For each video, features
    are extracted from all sampled frames and averaged to get a video-level
    representation. FVD is computed as the Fréchet distance between reference
    and generated video feature distributions.

    Note: True FVD uses I3D (Inception 3D) which captures temporal information.
    This frame-level approach focuses on spatial quality similarity.

    Lower scores indicate better quality (0 = identical distributions).
    """

    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self):
        logger.info("Running VideoFVDTest")
        request = self.targets.get("request")
        logger.info("Running VideoFVDTest with request: %s", request)

        if not isinstance(request, VideoFVDTestRequest):
            return {
                "success": False,
                "error": "VideoFVDTestRequest not provided in targets",
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

        elif request.action == "compute_fvd":
            logger.info("Computing FVD score")
            fvd_score = self._compute_fvd(
                reference_path=request.reference_videos_path,
                generated_path=request.generated_videos_path,
            )
            return {
                "success": True,
                "action": "compute_fvd",
                "fvd_score": fvd_score,
            }

        elif request.action == "full_eval":
            logger.info("Running full FVD evaluation pipeline")
            self._download_video_samples(
                count=request.download_count,
                category=request.category,
                dataset_name=request.hf_dataset_name,
            )
            fvd_score = self._compute_fvd(
                reference_path=request.reference_videos_path,
                generated_path=request.generated_videos_path,
            )
            self.eval_results["fvd"] = fvd_score
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

        FineVideo contains 43,751 CC-BY YouTube videos across 122 categories.
        Uses streaming mode to avoid downloading the full 600GB dataset.

        Args:
            count: Number of videos to download
            category: Filter by content_parent_category (e.g., "Sports", "Education")
            dataset_name: HuggingFace dataset identifier

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

        # Load dataset in streaming mode (avoids downloading full 600GB)
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        # Filter by category
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

            # Extract metadata
            video_title = sample_json.get("youtube_title", f"video_{idx}")
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in video_title
            )[:50]
            filename = f"{idx:03d}_{safe_title}.mp4"

            # Save video file
            video_path = videos_dir / filename
            with video_path.open("wb") as video_file:
                video_file.write(video_bytes)

            # Save individual metadata
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

        # Save combined metadata
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
        num_frames: int = 16,
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Load videos from directory and convert to tensor format.

        Uses imageio (transitive dependency of diffusers) for video reading.

        Args:
            video_dir: Directory containing MP4 video files
            num_frames: Number of frames to sample from each video (default: 16)
            target_size: Target frame size (height, width) for I3D input

        Returns:
            Array of shape (num_videos, num_frames, height, width, channels)
        """
        logger.info(f"Loading videos as tensors from {video_dir}")
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
                    video_path, num_frames, target_size, iio, Image
                )
                all_videos.append(frames)
                logger.info(f"Loaded {video_path.name}: {frames.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {video_path.name}: {e}")
                continue

        if not all_videos:
            raise RuntimeError("No videos could be loaded successfully")

        videos_tensor = np.stack(all_videos, axis=0)
        logger.info(
            f"Loaded {len(all_videos)} videos, tensor shape: {videos_tensor.shape}"
        )

        return videos_tensor

    def _load_single_video(
        self,
        video_path: Path,
        num_frames: int,
        target_size: tuple[int, int],
        iio,
        Image,
    ) -> np.ndarray:
        """
        Load a single video and sample frames uniformly.

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            target_size: Target frame size (height, width)
            iio: imageio.v3 module (passed to avoid repeated imports)
            Image: PIL.Image module (passed to avoid repeated imports)

        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        logger.info(f"Loading single video: {video_path}")
        # Read all frames from video
        all_frames = iio.imread(video_path, plugin="pyav")
        total_frames = len(all_frames)

        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")

        # Sample frames uniformly
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # If video has fewer frames than needed, repeat last frame
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (num_frames - total_frames))

        sampled_frames = []
        height, width = target_size

        for idx in indices:
            frame = all_frames[idx]

            # Resize frame using PIL
            pil_frame = Image.fromarray(frame)
            pil_frame = pil_frame.resize((width, height), Image.Resampling.BILINEAR)
            resized_frame = np.array(pil_frame)

            # Ensure RGB (3 channels)
            if len(resized_frame.shape) == 2:
                resized_frame = np.stack([resized_frame] * 3, axis=-1)
            elif resized_frame.shape[-1] == 4:
                resized_frame = resized_frame[:, :, :3]

            sampled_frames.append(resized_frame)

        return np.stack(sampled_frames, axis=0)

    def _extract_video_features(self, videos: np.ndarray) -> np.ndarray:
        """
        Extract features using InceptionV3 network (frame-level).

        Reuses the existing InceptionV3 from fid_score.py.
        For each video, extracts features from all frames and averages them
        to get a single feature vector per video.

        Args:
            videos: Video tensor of shape (N, T, H, W, C)
                    N = number of videos
                    T = number of frames per video
                    H, W = height, width (should be 299 for InceptionV3)
                    C = channels (3 for RGB)

        Returns:
            Features of shape (N, 2048) - InceptionV3 pool3 features
        """
        logger.info(f"Extracting features from {videos.shape[0]} videos")
        import torch
        from torchvision import transforms as TF

        from utils.sdxl_accuracy_utils.inception import InceptionV3

        logger.info(f"Extracting InceptionV3 features from {videos.shape[0]} videos")

        # Initialize InceptionV3 model (2048-dim features from pool3 layer)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx])
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Transform for InceptionV3: resize to 299x299 and normalize
        transform = TF.Compose(
            [
                TF.Resize((299, 299)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        num_videos, num_frames, _, _, _ = videos.shape
        video_features = []

        for video_idx in range(num_videos):
            frame_features = []

            for frame_idx in range(num_frames):
                # Get frame and convert to PIL Image
                frame = videos[video_idx, frame_idx]

                # Ensure uint8 format for PIL
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                pil_frame = Image.fromarray(frame)
                tensor_frame = transform(pil_frame).unsqueeze(0).to(device)

                # Extract features
                with torch.no_grad():
                    features = model(tensor_frame)[0]
                    features = features.squeeze().cpu().numpy()
                    frame_features.append(features)

            # Average features across all frames for this video
            avg_features = np.mean(frame_features, axis=0)
            video_features.append(avg_features)

            logger.info(
                f"Processed video {video_idx + 1}/{num_videos}, "
                f"avg feature shape: {avg_features.shape}"
            )

        features_array = np.array(video_features)
        logger.info(f"Extracted features shape: {features_array.shape}")

        return features_array

    def _calculate_activation_statistics(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of feature activations.

        Args:
            features: Feature array of shape (N, feature_dim)

        Returns:
            Tuple of (mean, covariance) for the feature distribution
        """
        logger.info("Calculating activation statistics")
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _compute_fvd(
        self,
        reference_path: str | None = None,
        generated_path: str | None = None,
    ) -> float:
        """
        Compute Fréchet Video Distance between reference and generated videos.

        Uses InceptionV3 for frame-level feature extraction (reuses fid_score.py).
        For each video, features are averaged across frames to get a single
        video-level feature vector. FVD is then computed as the Fréchet distance
        between the distributions of reference and generated video features.

        Note: This is a frame-level approximation of FVD. True FVD uses I3D
        which captures temporal information. This approach focuses on spatial
        quality similarity.

        Lower score indicates better quality (0 = identical distributions).
        """
        logger.info("Computing FVD between reference and generated videos")
        from utils.sdxl_accuracy_utils.fid_score import calculate_frechet_distance

        ref_path = (
            Path(reference_path) if reference_path else Path(DATASET_DIR) / "videos"
        )
        gen_path = (
            Path(generated_path) if generated_path else Path(DATASET_DIR) / "videos"
        )

        logger.info(f"Computing FVD: {ref_path} vs {gen_path}")

        # Load videos as tensors
        reference_videos = self._load_videos_as_tensors(ref_path)
        generated_videos = self._load_videos_as_tensors(gen_path)

        # Extract InceptionV3 features (frame-level, averaged per video)
        ref_features = self._extract_video_features(reference_videos)
        gen_features = self._extract_video_features(generated_videos)

        # Calculate statistics (mean and covariance)
        mu_ref, sigma_ref = self._calculate_activation_statistics(ref_features)
        mu_gen, sigma_gen = self._calculate_activation_statistics(gen_features)

        # Calculate FVD using existing fid_score implementation
        fvd_score = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

        logger.info(f"FVD Score: {fvd_score:.4f}")

        # Save results
        results_path = Path(DATASET_DIR) / FVD_RESULTS_FILE
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump({"fvd_score": float(fvd_score)}, f, indent=2)

        return float(fvd_score)
