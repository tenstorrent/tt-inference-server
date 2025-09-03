# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Visualization utilities for COCO evaluation results.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logger = logging.getLogger(__name__)


def ensure_visualization_dependencies():
    """Ensure required visualization dependencies are available."""
    try:
        import PIL
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing required dependency for visualization: {e}")
        raise RuntimeError(f"Visualization requires PIL and numpy: {e}")


def get_coco_colors() -> List[Tuple[int, int, int]]:
    """Get a list of distinct colors for COCO categories."""
    # Generate 80 distinct colors for the 80 COCO classes
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = []
    for i in range(80):
        # Generate bright, distinct colors
        hue = (i * 137.5) % 360  # Golden angle approximation
        saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly
        
        # Convert HSV to RGB and ensure proper integer values
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        r_int = max(0, min(255, int(round(r * 255))))
        g_int = max(0, min(255, int(round(g * 255))))
        b_int = max(0, min(255, int(round(b * 255))))
        colors.append((r_int, g_int, b_int))
    
    return colors


def get_default_font():
    """Get a default font for drawing text."""
    try:
        # Try to get a decent sized font
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (IOError, OSError):
        try:
            # Fallback to basic font
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            # Use default font as last resort
            try:
                return ImageFont.load_default()
            except:
                return None


def draw_bbox_with_label(
    draw: ImageDraw.Draw,
    bbox: List[float],  # [x, y, width, height] in COCO format
    class_name: str,
    confidence: float,
    color: Tuple[int, int, int] = (255, 0, 0),  # Red
    font = None
):
    """
    Draw a bounding box with label on an image.
    
    Args:
        draw: ImageDraw object
        bbox: Bounding box in COCO format [x, y, width, height]
        class_name: Class name for the detection
        confidence: Confidence score
        color: RGB color tuple for the bounding box
        font: Font to use for text (uses default if None)
    """
    # Validate and sanitize inputs
    if len(bbox) != 4:
        logger.warning(f"Invalid bbox format: {bbox}")
        return
    
    x, y, width, height = bbox
    
    # Ensure coordinates are valid numbers
    try:
        x, y, width, height = float(x), float(y), float(width), float(height)
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid bbox dimensions: width={width}, height={height}")
            return
    except (ValueError, TypeError):
        logger.warning(f"Invalid bbox coordinates: {bbox}")
        return
    
    x2 = x + width
    y2 = y + height
    
    # Ensure color is properly formatted
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        color = (255, 0, 0)  # Default to red
    color = tuple(max(0, min(255, int(c))) for c in color)
    
    # Draw bounding box
    draw.rectangle([x, y, x2, y2], outline=color, width=3)
    
    # Prepare label text
    label_text = f"{class_name}: {confidence:.2f}"
    
    if font is None:
        font = get_default_font()
    
    # Get text size
    if font:
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        # Estimate text size if no font available
        text_width = len(label_text) * 8
        text_height = 12
    
    # Create background box for text (red background as requested)
    label_bg_bbox = [x, y - text_height - 4, x + text_width + 6, y]
    draw.rectangle(label_bg_bbox, fill=color, outline=color)
    
    # Draw white text on red background
    text_position = (x + 3, y - text_height - 2)
    draw.text(text_position, label_text, fill=(255, 255, 255), font=font)


def visualize_coco_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    class_mapping: Optional[Dict[str, int]] = None,
    image_id: int = 0,
    save_path: Optional[Path] = None
) -> Image.Image:
    """
    Visualize COCO detections on an image.
    
    Args:
        image: PIL Image object
        detections: List of YOLOv4 detection dictionaries
        ground_truth: Optional list of ground truth annotations
        class_mapping: Mapping from class names to COCO category IDs
        image_id: Image ID for logging purposes
        save_path: Optional path to save the visualized image
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    ensure_visualization_dependencies()
    
    # Make a copy of the image to draw on and ensure it's in RGB mode
    vis_image = image.copy()
    if vis_image.mode != 'RGB':
        vis_image = vis_image.convert('RGB')
    
    draw = ImageDraw.Draw(vis_image)
    font = get_default_font()
    colors = get_coco_colors()
    
    # Draw detections from model predictions
    for det_idx, detection in enumerate(detections):
        try:
            # Get color based on class_id or class_name
            if "class_id" in detection:
                color_idx = detection["class_id"] % len(colors)
            elif "class_name" in detection and class_mapping:
                color_idx = class_mapping.get(detection["class_name"], 0) % len(colors)
            else:
                color_idx = det_idx % len(colors)
            
            # Ensure we have a valid color tuple
            color = colors[color_idx]
            if not isinstance(color, tuple) or len(color) != 3:
                color = (255, 0, 0)  # Default to red if color is invalid
            
            # Ensure all color components are integers
            color = tuple(int(c) for c in color)
            
            # Convert YOLOv4 bbox format to COCO format for drawing
            bbox_dict = detection["bbox"]
            x1, y1, x2, y2 = bbox_dict["x1"], bbox_dict["y1"], bbox_dict["x2"], bbox_dict["y2"]
            
            # Convert from normalized coordinates to pixel coordinates
            img_width, img_height = image.size
            x1_px = x1 * img_width
            y1_px = y1 * img_height
            x2_px = x2 * img_width
            y2_px = y2 * img_height
            
            # Convert to COCO format [x, y, width, height]
            width = x2_px - x1_px
            height = y2_px - y1_px
            coco_bbox = [x1_px, y1_px, width, height]
            
            class_name = detection.get("class_name", f"class_{detection.get('class_id', 'unknown')}")
            confidence = detection.get("confidence", 1.0)
            
            draw_bbox_with_label(
                draw, coco_bbox, class_name, confidence, color, font
            )
            
        except Exception as e:
            logger.warning(f"Failed to draw detection {det_idx}: {e}")
            continue
    
    # Optionally draw ground truth boxes in a different style (green, dashed)
    if ground_truth:
        for gt_idx, gt_ann in enumerate(ground_truth):
            try:
                # Ground truth is already in COCO format
                gt_bbox = gt_ann["bbox"]
                gt_category_id = gt_ann["category_id"]
                
                # Use green color for ground truth
                gt_color = (0, 255, 0)  # Green
                # Ensure ground truth color is properly formatted
                gt_color = tuple(int(c) for c in gt_color)
                
                # Get class name from category ID
                gt_class_name = "unknown"
                if class_mapping:
                    for name, cat_id in class_mapping.items():
                        if cat_id == gt_category_id:
                            gt_class_name = name
                            break
                
                x, y, width, height = gt_bbox
                x2 = x + width
                y2 = y + height
                
                # Draw ground truth with thinner green box
                draw.rectangle([x, y, x2, y2], outline=gt_color, width=2)
                
                # Optional: draw GT label
                gt_label = f"GT: {gt_class_name}"
                if font:
                    text_bbox = draw.textbbox((0, 0), gt_label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width = len(gt_label) * 8
                    text_height = 12
                
                # Place GT label below the box
                label_y = y2 + 2
                label_bg_bbox = [x, label_y, x + text_width + 6, label_y + text_height + 4]
                draw.rectangle(label_bg_bbox, fill=gt_color, outline=gt_color)
                draw.text((x + 3, label_y + 2), gt_label, fill=(255, 255, 255), font=font)
                
            except Exception as e:
                logger.warning(f"Failed to draw ground truth annotation {gt_idx}: {e}")
                continue
    
    # Save the visualized image if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image.save(save_path, "PNG")
    
    return vis_image


def create_visualization_summary(
    output_path: Path,
    total_images: int,
    total_detections: int,
    metrics: Dict[str, float]
):
    """
    Create a summary file with visualization statistics.
    
    Args:
        output_path: Base output path for saving files
        total_images: Total number of images processed
        total_detections: Total number of detections found
        metrics: Evaluation metrics dictionary
    """
    summary = {
        "visualization_summary": {
            "total_images_processed": total_images,
            "total_detections_generated": total_detections,
            "images_with_visualizations": total_images,
            "visualization_output_directory": str(output_path / "visualizations"),
            "evaluation_directory": str(output_path),
            "file_format": "PNG"
        },
        "evaluation_metrics": metrics
    }
    
    summary_path = output_path / "visualization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)