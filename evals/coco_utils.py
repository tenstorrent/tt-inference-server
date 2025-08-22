# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
COCO dataset evaluation utilities for YOLOv4 object detection.
"""

import base64
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
import requests

from utils.image_client import ImageClient

logger = logging.getLogger(__name__)


@dataclass
class COCODetection:
    """Represents a detection in COCO format."""
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    score: float


@dataclass
class COCOMetrics:
    """COCO evaluation metrics."""
    mAP: float  # Mean Average Precision at IoU=0.50:0.95
    mAP_50: float  # Mean Average Precision at IoU=0.50
    mAP_75: float  # Mean Average Precision at IoU=0.75
    mAP_small: float  # Mean Average Precision for small objects
    mAP_medium: float  # Mean Average Precision for medium objects
    mAP_large: float  # Mean Average Precision for large objects
    mAR_1: float  # Mean Average Recall with 1 detection per image
    mAR_10: float  # Mean Average Recall with 10 detections per image
    mAR_100: float  # Mean Average Recall with 100 detections per image
    mAR_small: float  # Mean Average Recall for small objects
    mAR_medium: float  # Mean Average Recall for medium objects
    mAR_large: float  # Mean Average Recall for large objects


def ensure_pycocotools():
    """Ensure pycocotools is installed, install if necessary."""
    try:
        import pycocotools
    except ImportError:
        logger.info("pycocotools not found. Installing...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pycocotools", "-q"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                logger.error(f"Failed to install pycocotools: {result.stderr}")
                raise RuntimeError("Could not install pycocotools")
            logger.info("Successfully installed pycocotools")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout installing pycocotools")
        except Exception as e:
            raise RuntimeError(f"Failed to install pycocotools: {e}")


def load_coco_annotations(annotations_path: str) -> Dict[str, Any]:
    """Load COCO annotations from JSON file."""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def get_coco_image_paths(dataset_path: str, annotations: Dict[str, Any]) -> Dict[int, str]:
    """Get mapping of image_id to image file path."""
    image_paths = {}
    dataset_path = Path(dataset_path)
    
    for image_info in annotations['images']:
        image_id = image_info['id']
        filename = image_info['file_name']
        image_path = dataset_path / filename
        
        if image_path.exists():
            image_paths[image_id] = str(image_path)
        else:
            logger.warning(f"Image not found: {image_path}")
    
    return image_paths


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string


def convert_yolov4_to_coco_detection(
    detection: Dict[str, Any], 
    image_id: int, 
    class_id_mapping: Dict[str, int]
) -> Optional[COCODetection]:
    """
    Convert YOLOv4 detection format to COCO detection format.
    
    YOLOv4 format: {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float}, 
                    "confidence": float, "class_id": int, "class_name": str}
    COCO format: [x, y, width, height] where (x,y) is top-left corner
    """
    try:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Convert from (x1, y1, x2, y2) to COCO format (x, y, width, height)
        width = x2 - x1
        height = y2 - y1
        
        # Map class name to COCO category ID
        class_name = detection["class_name"]
        if class_name not in class_id_mapping:
            logger.warning(f"Unknown class name: {class_name}")
            return None
        
        category_id = class_id_mapping[class_name]
        
        return COCODetection(
            image_id=image_id,
            category_id=category_id,
            bbox=[x1, y1, width, height],
            score=detection["confidence"]
        )
    except Exception as e:
        logger.error(f"Error converting detection: {e}")
        return None


def get_coco_class_mapping() -> Dict[str, int]:
    """Get mapping from class names to COCO category IDs."""
    # COCO class names in order (index + 1 = category_id)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Create mapping from class name to COCO category ID (1-indexed)
    return {class_name: idx + 1 for idx, class_name in enumerate(COCO_CLASSES)}


def run_yolov4_coco_evaluation(
    service_port: str,
    output_path: str,
    coco_dataset_path: str,
    coco_annotations_path: str,
    max_images: Optional[int] = None,
    jwt_secret: Optional[str] = None
) -> Dict[str, float]:
    """
    Run COCO evaluation against YOLOv4 service.
    
    Args:
        service_port: Port of the YOLOv4 inference server
        output_path: Path to save evaluation results
        coco_dataset_path: Path to COCO validation images
        coco_annotations_path: Path to COCO annotations JSON file
        max_images: Limit number of images to evaluate (for testing)
        jwt_secret: JWT secret for API authentication
        
    Returns:
        Dictionary containing evaluation metrics
    """
    ensure_pycocotools()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    logger.info(f"Loading COCO annotations from {coco_annotations_path}")
    annotations = load_coco_annotations(coco_annotations_path)
    image_paths = get_coco_image_paths(coco_dataset_path, annotations)
    class_mapping = get_coco_class_mapping()
    
    logger.info(f"Found {len(image_paths)} images for evaluation")
    if max_images:
        # Limit to max_images for faster testing
        image_paths = dict(list(image_paths.items())[:max_images])
        logger.info(f"Limited to {len(image_paths)} images for evaluation")
    
    # Initialize image client
    client = ImageClient(base_url=f"http://localhost:{service_port}", jwt_secret=jwt_secret)
    
    # Run inference on all images
    logger.info("Running inference on COCO validation set...")
    all_detections = []
    processed_count = 0
    
    for image_id, image_path in image_paths.items():
        try:
            # Convert image to base64
            image_base64 = image_to_base64(image_path)
            
            # Run inference
            response = client.search_image(image_base64)
            
            if response.status_code == 200:
                results = response.json()
                
                # Convert detections to COCO format
                if "detections" in results:
                    for detection in results["detections"]:
                        coco_detection = convert_yolov4_to_coco_detection(
                            detection, image_id, class_mapping
                        )
                        if coco_detection:
                            all_detections.append(coco_detection)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(image_paths)} images")
            else:
                logger.error(f"Inference failed for image {image_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            continue
    
    logger.info(f"Generated {len(all_detections)} detections from {processed_count} images")
    
    # Convert detections to COCO results format
    coco_results = []
    for detection in all_detections:
        coco_results.append(asdict(detection))
    
    # Save detection results
    results_file = Path(output_path) / "coco_detections.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(coco_results, f)
    logger.info(f"Saved detection results to {results_file}")
    
    # Initialize COCO evaluator
    coco_gt = COCO(coco_annotations_path)
    
    # Filter ground truth to only include images we processed
    processed_image_ids = list(image_paths.keys())
    coco_gt.imgs = {k: v for k, v in coco_gt.imgs.items() if k in processed_image_ids}
    
    # Load detection results
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(str(results_file))
        
        # Run COCO evaluation
        logger.info("Running COCO evaluation...")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = processed_image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = COCOMetrics(
            mAP=coco_eval.stats[0],      # AP at IoU=0.50:0.95
            mAP_50=coco_eval.stats[1],   # AP at IoU=0.50
            mAP_75=coco_eval.stats[2],   # AP at IoU=0.75
            mAP_small=coco_eval.stats[3],  # AP for small objects
            mAP_medium=coco_eval.stats[4], # AP for medium objects
            mAP_large=coco_eval.stats[5],  # AP for large objects
            mAR_1=coco_eval.stats[6],    # AR maxDets=1
            mAR_10=coco_eval.stats[7],   # AR maxDets=10
            mAR_100=coco_eval.stats[8],  # AR maxDets=100
            mAR_small=coco_eval.stats[9],  # AR for small objects
            mAR_medium=coco_eval.stats[10], # AR for medium objects
            mAR_large=coco_eval.stats[11],  # AR for large objects
        )
        
        # Save metrics
        metrics_file = Path(output_path) / "coco_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Log key metrics
        logger.info(f"COCO Evaluation Results:")
        logger.info(f"  mAP@0.5:0.95: {metrics.mAP:.4f}")
        logger.info(f"  mAP@0.5:     {metrics.mAP_50:.4f}")
        logger.info(f"  mAP@0.75:    {metrics.mAP_75:.4f}")
        logger.info(f"  mAR@100:     {metrics.mAR_100:.4f}")
        
        return asdict(metrics)
    else:
        logger.error("No valid detections generated")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
