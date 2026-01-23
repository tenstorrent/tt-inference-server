#!/usr/bin/env python3
"""
8-Device Data Parallel YOLOv8s Benchmark

Demonstrates running YOLOv8s inference on 8 images simultaneously,
one image per Tenstorrent device (T3K/Galaxy configuration).

Usage:
    python3 test_8device_parallel.py [--save-output]
"""
import sys
import time
import argparse
import torch
import cv2
import numpy as np

# TT imports
import ttnn
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import (
    get_mesh_mappers,
    load_coco_class_names,
    postprocess,
)


def create_test_images(num_images, size=(640, 640)):
    """Create test images with different colors for visual distinction."""
    images = []
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)
    ]
    for i in range(num_images):
        # Create colored gradient image
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        color = colors[i % len(colors)]
        for y in range(size[1]):
            for x in range(size[0]):
                img[y, x] = [
                    int(color[0] * (x / size[0])),
                    int(color[1] * (y / size[1])),
                    int(color[2] * ((x + y) / (size[0] + size[1])))
                ]
        # Add text label
        cv2.putText(img, f"Device {i}", (50, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        images.append(img)
    return images


def preprocess_images(images):
    """Convert images to tensor batch for inference."""
    tensors = []
    for img in images:
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize and convert to tensor
        tensor = torch.from_numpy(img_rgb).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        tensors.append(tensor)
    # Stack into batch
    return torch.cat(tensors, dim=0)


def draw_detections(image, result, names, conf_threshold=0.5):
    """Draw bounding boxes on image."""
    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    
    for box, score, cls in zip(boxes, scores, classes):
        conf = float(score)
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        class_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        label = f"{class_name} {conf:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description="8-Device YOLOv8s Parallel Benchmark")
    parser.add_argument("--save-output", action="store_true", help="Save output images")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  8-Device Data Parallel YOLOv8s Benchmark")
    print("=" * 70)

    # Get available devices
    device_ids = ttnn.get_device_ids()
    num_devices = len(device_ids)
    print(f"\n📟 Found {num_devices} TT devices: {device_ids}")

    if num_devices < 2:
        print("⚠️  Only 1 device found. This demo requires multiple devices.")
        print("   Running in single-device mode for comparison.\n")

    # Create mesh device
    mesh_shape = ttnn.MeshShape(1, num_devices)
    device_params = {
        "l1_small_size": YOLOV8S_L1_SMALL_SIZE,
        "trace_region_size": 6434816,
        "num_command_queues": 2,
    }

    print(f"\n🔧 Creating mesh device with shape {mesh_shape}...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    mesh_device.enable_program_cache()
    
    actual_devices = mesh_device.get_num_devices()
    print(f"✅ Mesh device created with {actual_devices} devices")

    # Get mesh mappers
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # Create runner with batch_size = num_devices
    batch_size = actual_devices
    print(f"\n📦 Loading YOLOv8s model (batch_size={batch_size})...")
    print("   This includes trace compilation (~2 min on first run)...")
    
    t_load_start = time.time()
    runner = YOLOv8sPerformantRunner(
        mesh_device,
        device_batch_size=batch_size,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    t_load_end = time.time()
    print(f"✅ Model loaded in {t_load_end - t_load_start:.1f}s")

    # Create test images
    print(f"\n🖼️  Creating {batch_size} test images (640x640)...")
    test_images = create_test_images(batch_size)
    input_tensor = preprocess_images(test_images)
    print(f"   Input tensor shape: {input_tensor.shape}")

    # Warmup
    print("\n🔥 Warmup (2 iterations)...")
    for i in range(2):
        _ = runner.run(input_tensor)
        print(f"   Warmup {i+1}/2 done")
    ttnn.synchronize_device(mesh_device)

    # Benchmark
    iterations = args.iterations
    print(f"\n⏱️  Benchmark ({iterations} iterations)...")
    
    times = []
    for i in range(iterations):
        t_start = time.time()
        preds = runner.run(input_tensor)
        ttnn.synchronize_device(mesh_device)
        t_end = time.time()
        times.append(t_end - t_start)
        print(f"   Iteration {i+1}/{iterations}: {(t_end-t_start)*1000:.1f}ms")

    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = batch_size / avg_time

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Devices:           {actual_devices}")
    print(f"  Batch size:        {batch_size} images")
    print(f"  Iterations:        {iterations}")
    print(f"  ")
    print(f"  Avg time/batch:    {avg_time*1000:.2f}ms")
    print(f"  Min time/batch:    {min_time*1000:.2f}ms")
    print(f"  Max time/batch:    {max_time*1000:.2f}ms")
    print(f"  ")
    print(f"  Throughput:        {throughput:.1f} images/sec")
    print(f"  Per-device FPS:    ~{throughput/actual_devices:.1f} FPS")
    print("=" * 70)

    # Optional: save output images with detections
    if args.save_output:
        print("\n💾 Saving output images...")
        names = load_coco_class_names()
        
        # Get predictions as torch tensor
        preds_torch = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=output_mesh_composer)
        
        # Process each image
        results = postprocess(preds_torch, input_tensor, test_images, 
                             [[str(i)] for i in range(batch_size)], names)
        
        for i, (img, result) in enumerate(zip(test_images, results)):
            out_img = draw_detections(img.copy(), result, names)
            filename = f"output_device_{i}.jpg"
            cv2.imwrite(filename, out_img)
            print(f"   Saved: {filename}")

    # Cleanup
    print("\n🧹 Cleaning up...")
    runner.release()
    ttnn.close_mesh_device(mesh_device)
    print("✅ Done!\n")


if __name__ == "__main__":
    main()
