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
import io

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
    coco_dataset,
    service_port: str,
    output_path: str,
    max_images: Optional[int] = None,
    jwt_secret: Optional[str] = None
) -> Dict[str, float]:
    """
    Run COCO evaluation against YOLOv4 service using a Hugging Face Dataset object.
    
    Args:
        coco_dataset: The loaded Hugging Face COCO dataset object (e.g., from datasets.load_dataset).
        service_port: Port of the YOLOv4 inference server.
        output_path: Path to save evaluation results.
        max_images: Limit number of images to evaluate (for testing).
        jwt_secret: JWT secret for API authentication.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    ensure_pycocotools()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class_mapping = get_coco_class_mapping()
    
    if max_images:
        coco_dataset = coco_dataset.select(range(max_images))
    
    logger.info(f"Evaluating on {len(coco_dataset)} images from the dataset.")

    # Initialize image client
    client = ImageClient(base_url=f"http://localhost:{service_port}", jwt_secret=jwt_secret)
    
    # Run inference on all images
    logger.info("Running inference on COCO validation set...")
    all_detections = []
    processed_count = 0
    
    # Create a temporary ground truth file for the evaluator
    gt_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco_dataset.features['objects'].feature['category'].names
    }
    
    for item in coco_dataset:
        image_id = item['image_id']
        image = item['image']
        
        # Add image and annotation info to our ground truth structure
        gt_annotations["images"].append({"id": image_id, "width": image.width, "height": image.height, "file_name": f"{image_id}.jpg"})
        for i, bbox in enumerate(item['objects']['bbox']):
            gt_annotations["annotations"].append({
                "id": len(gt_annotations["annotations"]),
                "image_id": image_id,
                "category_id": item['objects']['category'][i],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            response = client.search_image(image_base64)
            
            if response.status_code == 200:
                results = response.json()
                if "detections" in results:
                    for detection in results["detections"]:
                        coco_detection = convert_yolov4_to_coco_detection(
                            detection, image_id, class_mapping
                        )
                        if coco_detection:
                            all_detections.append(coco_detection)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(coco_dataset)} images")
            else:
                logger.error(f"Inference failed for image {image_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            continue
            
    logger.info(f"Generated {len(all_detections)} detections from {processed_count} images")

    # Save detection results
    results_file = Path(output_path) / "coco_detections.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump([asdict(d) for d in all_detections], f)
    logger.info(f"Saved detection results to {results_file}")

    # Save the temporary ground truth file
    gt_file = Path(output_path) / "temp_coco_gt.json"
    with open(gt_file, 'w') as f:
        json.dump(gt_annotations, f)
        
    # Initialize COCO evaluator
    coco_gt = COCO(str(gt_file))
    
    if len(all_detections) > 0:
        coco_dt = coco_gt.loadRes(str(results_file))
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = [img['id'] for img in gt_annotations['images']]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = COCOMetrics(
            mAP=coco_eval.stats[0],
            mAP_50=coco_eval.stats[1],
            mAP_75=coco_eval.stats[2],
            mAP_small=coco_eval.stats[3],
            mAP_medium=coco_eval.stats[4],
            mAP_large=coco_eval.stats[5],
            mAR_1=coco_eval.stats[6],
            mAR_10=coco_eval.stats[7],
            mAR_100=coco_eval.stats[8],
            mAR_small=coco_eval.stats[9],
            mAR_medium=coco_eval.stats[10],
            mAR_large=coco_eval.stats[11],
        )
        
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
