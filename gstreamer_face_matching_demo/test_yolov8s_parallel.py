#!/usr/bin/env python3
"""Quick test: YOLOv8s data parallel on 8 devices."""

import os
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import time
import torch
import ttnn

from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import get_mesh_mappers

def main():
    print("\n" + "="*60)
    print("  YOLOv8s Data Parallel Test (8 devices)")
    print("="*60)
    
    # Get devices
    device_ids = ttnn.get_device_ids()
    num_devices = len(device_ids)
    print(f"\n📟 Found {num_devices} TT devices")
    
    if num_devices < 8:
        print(f"⚠️  Only {num_devices} devices available")
    
    # Create mesh device
    mesh_shape = ttnn.MeshShape(1, num_devices)
    device_params = {
        "l1_small_size": YOLOV8S_L1_SMALL_SIZE,
        "trace_region_size": 6434816,
        "num_command_queues": 2,
    }
    
    print(f"\n🔧 Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    mesh_device.enable_program_cache()
    print(f"✅ Mesh device opened with {mesh_device.get_num_devices()} devices")
    
    # Get mesh mappers
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    
    # Load model
    print(f"\n📦 Loading YOLOv8s model...")
    print("   ⏳ First run includes compilation (~2 min)...")
    
    runner = YOLOv8sPerformantRunner(
        mesh_device,
        device_batch_size=num_devices,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    
    # Warmup
    print("\n🔥 Warming up...")
    dummy = torch.randn(num_devices, 3, 640, 640)
    for i in range(3):
        _ = runner.run(dummy)
        print(f"   Warmup {i+1}/3")
    ttnn.synchronize_device(mesh_device)
    print("✅ Model ready!")
    
    # Benchmark
    print("\n⚡ Running benchmark (100 iterations)...")
    times = []
    for i in range(100):
        dummy = torch.randn(num_devices, 3, 640, 640)
        t0 = time.time()
        _ = runner.run(dummy)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        times.append(t1 - t0)
        
        if (i + 1) % 20 == 0:
            avg = sum(times[-20:]) / 20
            fps = num_devices / avg
            print(f"   Iter {i+1}: {fps:.0f} FPS ({1000*avg:.1f}ms per batch)")
    
    # Final stats
    avg_time = sum(times[10:]) / len(times[10:])  # Skip first 10
    total_fps = num_devices / avg_time
    per_device_fps = 1 / avg_time
    
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  Devices:        {num_devices}")
    print(f"  Total FPS:      {total_fps:.0f}")
    print(f"  Per-device FPS: {per_device_fps:.1f}")
    print(f"  Batch latency:  {1000*avg_time:.1f}ms")
    print("="*60)
    
    # Cleanup
    runner.release()
    ttnn.close_mesh_device(mesh_device)
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
