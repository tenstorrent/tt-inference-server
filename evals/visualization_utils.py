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


def _validate_and_clip_detection_coordinates(x1: float, y1: float, x2: float, y2: float,
                                            img_width: int, img_height: int) -> Optional[tuple]:
    """Validate and clip detection coordinates to image boundaries."""
    x1_px = max(0, min(x1, img_width))
    y1_px = max(0, min(y1, img_height))
    x2_px = max(x1_px + 1, min(x2, img_width))
    y2_px = max(y1_px + 1, min(y2, img_height))
    
    width = x2_px - x1_px
    height = y2_px - y1_px
    
    if width < 2.0 or height < 2.0:  # Minimum meaningful size
        return None
        
    return x1_px, y1_px, width, height


def _detect_coordinate_format(coords: List[float]) -> bool:
    """Detect if coordinates are normalized [0,1] or pixel values."""
    max_coord = max(coords)
    min_coord = min(coords)
    
    if max_coord <= 1.0 and min_coord >= 0.0:
        return True  # Normalized
    elif max_coord > 320 or any(c < 0 for c in coords):
        return True  # Assume normalized
    elif max_coord <= 320 and min_coord >= 0:
        return not all(abs(c - round(c)) < 0.01 for c in coords if c > 1)
    else:
        return True  # Default to normalized

def _transform_detection_coordinates(x1: float, y1: float, x2: float, y2: float,
                                   img_width: int, img_height: int,
                                   model_input_size: tuple = (320, 320)) -> Optional[tuple]:
    """Transform YOLOv4 detection coordinates back to original image space."""
    target_w, target_h = model_input_size
    scale = min(target_w / img_width, target_h / img_height)
    
    scaled_w = int(img_width * scale)
    scaled_h = int(img_height * scale)
    pad_x = (target_w - scaled_w) // 2
    pad_y = (target_h - scaled_h) // 2
    
    if scaled_w <= 0 or scaled_h <= 0:
        return None
        
    coords = [x1, y1, x2, y2]
    is_normalized = _detect_coordinate_format(coords)
    
    if is_normalized:
        x1_padded = max(0, min(1, x1)) * target_w
        y1_padded = max(0, min(1, y1)) * target_h
        x2_padded = max(0, min(1, x2)) * target_w
        y2_padded = max(0, min(1, y2)) * target_h
    else:
        x1_padded = max(0, min(target_w, x1))
        y1_padded = max(0, min(target_h, y1))
        x2_padded = max(0, min(target_w, x2))
        y2_padded = max(0, min(target_h, y2))
    
    # Remove padding and validate
    x1_content = x1_padded - pad_x
    y1_content = y1_padded - pad_y
    x2_content = x2_padded - pad_x
    y2_content = y2_padded - pad_y
    
    # Early validation
    if (x1_content < -scaled_w or x2_content < -scaled_w or 
        y1_content < -scaled_h or y2_content < -scaled_h or
        x1_content > scaled_w * 2 or x2_content > scaled_w * 2 or
        y1_content > scaled_h * 2 or y2_content > scaled_h * 2):
        return None
    
    # Scale back to original image size
    x1_px = max(0, (x1_content / scaled_w) * img_width)
    y1_px = max(0, (y1_content / scaled_h) * img_height)
    x2_px = min(img_width, (x2_content / scaled_w) * img_width)
    y2_px = min(img_height, (y2_content / scaled_h) * img_height)
    
    return _validate_and_clip_detection_coordinates(x1_px, y1_px, x2_px, y2_px, img_width, img_height)


def _transform_yolov11_detection_coordinates(x1: float, y1: float, x2: float, y2: float,
                                           img_width: int, img_height: int) -> Optional[tuple]:
    """Transform YOLOv11 detection coordinates"""
    # YOLOv11 coordinates are already in pixel coordinates, just clip to image bounds
    x1_px = max(0, min(img_width, x1))
    y1_px = max(0, min(img_height, y1))
    x2_px = max(0, min(img_width, x2))
    y2_px = max(0, min(img_height, y2))
    
    return _validate_and_clip_detection_coordinates(x1_px, y1_px, x2_px, y2_px, img_width, img_height)

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
            # Get display color
            if "class_id" in detection:
                color_idx = detection["class_id"] % len(colors)
            elif "class_name" in detection and class_mapping:
                color_idx = class_mapping.get(detection["class_name"], 0) % len(colors)
            else:
                color_idx = det_idx % len(colors)
            
            color = colors[color_idx]
            if not isinstance(color, tuple) or len(color) != 3:
                color = (255, 0, 0)
            color = tuple(int(c) for c in color)
            
            # Transform YOLOv4 coordinates to original image coordinates
            bbox_dict = detection["bbox"]
            x1, y1, x2, y2 = bbox_dict["x1"], bbox_dict["y1"], bbox_dict["x2"], bbox_dict["y2"]
            
            coords = _transform_detection_coordinates(
                x1, y1, x2, y2, image.size[0], image.size[1]
            )
            if coords is None:
                continue
                
            x1_px, y1_px, width, height = coords
            class_name = detection.get("class_name", f"class_{detection.get('class_id', 'unknown')}")
            confidence = detection.get("confidence", 1.0)
            
            draw_bbox_with_label(
                draw, [x1_px, y1_px, width, height], class_name, confidence, color, font
            )
            
        except Exception as e:
            logger.warning(f"Failed to draw detection {det_idx}: {e}")
            continue
    
    # Draw ground truth annotations in green
    if ground_truth:
        _draw_ground_truth_annotations(
            draw, ground_truth, class_mapping, image.size, image_id, font
        )
    # Save visualization if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image.save(save_path, "PNG")
    
    return vis_image


def visualize_yolov11_coco_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    class_mapping: Optional[Dict[str, int]] = None,
    image_id: int = 0,
    save_path: Optional[Path] = None,
    min_confidence: float = 0.25  # Add confidence threshold for visualization
) -> Image.Image:
    """
    Visualize YOLOv11 COCO detections on an image.
    
    Args:
        image: PIL Image object
        detections: List of YOLOv11 detection dictionaries (pixel coordinates)
        ground_truth: Optional list of ground truth annotations
        class_mapping: Mapping from class names to COCO category IDs
        image_id: Image ID for logging purposes
        save_path: Optional path to save the visualized image
        min_confidence: Minimum confidence threshold for visualization (default: 0.25)
        
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
    
    # Filter detections by confidence for cleaner visualization
    filtered_detections = [d for d in detections if d.get("confidence", 0) >= min_confidence]
    
    #logger.info(f"Creating YOLOv11 visualization for image {image_id}: {len(filtered_detections)}/{len(detections)} detections (conf >= {min_confidence})")
    
    # Draw detections from model predictions
    for det_idx, detection in enumerate(filtered_detections):
        try:
            # Get display color
            if "class_id" in detection:
                color_idx = detection["class_id"] % len(colors)
            elif "class_name" in detection and class_mapping:
                color_idx = class_mapping.get(detection["class_name"], 0) % len(colors)
            else:
                color_idx = det_idx % len(colors)
            
            color = colors[color_idx]
            if not isinstance(color, tuple) or len(color) != 3:
                color = (255, 0, 0)
            color = tuple(int(c) for c in color)
            
            # Transform YOLOv11 coordinates (already in pixel space, just clip)
            bbox_dict = detection["bbox"]
            x1, y1, x2, y2 = bbox_dict["x1"], bbox_dict["y1"], bbox_dict["x2"], bbox_dict["y2"]
            
            coords = _transform_yolov11_detection_coordinates(
                x1, y1, x2, y2, image.size[0], image.size[1]
            )
            if coords is None:
                continue
                
            x1_px, y1_px, width, height = coords
            class_name = detection.get("class_name", f"class_{detection.get('class_id', 'unknown')}")
            confidence = detection.get("confidence", 1.0)
            
            #logger.debug(f"Drawing YOLOv11 bbox: ({x1_px:.1f}, {y1_px:.1f}, {width:.1f}, {height:.1f}) - {class_name} ({confidence:.2f})")
            
            draw_bbox_with_label(
                draw, [x1_px, y1_px, width, height], class_name, confidence, color, font
            )
            
        except Exception as e:
            logger.warning(f"Failed to draw YOLOv11 detection {det_idx}: {e}")
            continue
    
    # Optionally draw ground truth annotations in green (can be disabled)
    # Comment out the next 4 lines if you don't want GT annotations
    if ground_truth:
        _draw_ground_truth_annotations(
            draw, ground_truth, class_mapping, image.size, image_id, font
        )
    
    
    # Save visualization if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image.save(save_path, "PNG")
        #logger.info(f"Saved YOLOv11 visualization to: {save_path}")
    
    return vis_image

def _draw_ground_truth_label(draw, x, y, label_text, color, font):
    """Draw ground truth label below bounding box."""
    if font:
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_width = len(label_text) * 8
        text_height = 12
    
    label_bg_bbox = [x, y + 2, x + text_width + 6, y + text_height + 6]
    draw.rectangle(label_bg_bbox, fill=color, outline=color)
    draw.text((x + 3, y + 4), label_text, fill=(255, 255, 255), font=font)


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