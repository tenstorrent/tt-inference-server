#!/usr/bin/env python3
"""Data parallel test: YuNet + SFace on 8 devices (full face recognition)."""

import os
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import time
import torch
import ttnn

# Import YuNet
from models.experimental.yunet.common import get_default_weights_path, load_torch_model, YUNET_L1_SMALL_SIZE
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model

# Import SFace
from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
from models.experimental.sface.reference.sface_model import load_sface_from_onnx
from models.experimental.sface.tt.ttnn_sface import create_sface_model


def get_mesh_mappers(device):
    """Standard data parallel mappers."""
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def main():
    print("\n" + "="*60)
    print("  Mesh Data Parallel Test (YuNet + SFace on 8 devices)")
    print("="*60)
    
    # Get devices
    device_ids = ttnn.get_device_ids()
    num_devices = len(device_ids)
    print(f"\n📟 Found {num_devices} TT devices")
    
    # Use larger L1 size for both models
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    
    # Create mesh device (same as YOLOv8s)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    device_params = {
        "l1_small_size": l1_size,
        "trace_region_size": 6434816,
        "num_command_queues": 2,
    }
    
    print(f"\n🔧 Opening mesh device with shape (1, {num_devices})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    mesh_device.enable_program_cache()
    print(f"✅ Mesh device opened with {mesh_device.get_num_devices()} devices")
    
    # Get mesh mappers
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    print(f"\n📊 Mesh mappers:")
    print(f"   inputs_mesh_mapper:  {type(inputs_mesh_mapper).__name__}")
    print(f"   weights_mesh_mapper: {type(weights_mesh_mapper).__name__}")
    print(f"   output_mesh_composer: {type(output_mesh_composer).__name__}")
    
    # Load YuNet torch model
    print(f"\n📦 Loading YuNet model...")
    yunet_weights = get_default_weights_path()
    yunet_torch = load_torch_model(yunet_weights).to(torch.bfloat16)
    yunet_model = create_yunet_model(mesh_device, yunet_torch, weights_mesh_mapper=weights_mesh_mapper)
    print("✅ YuNet ready!")
    
    # Load SFace torch model
    print(f"\n📦 Loading SFace model...")
    sface_onnx = get_sface_onnx_path()
    sface_torch = load_sface_from_onnx(sface_onnx)
    sface_torch.eval()
    sface_model = create_sface_model(mesh_device, sface_torch, weights_mesh_mapper=weights_mesh_mapper)
    print("✅ SFace ready!")
    
    # Warmup YuNet
    print("\n🔥 Warming up YuNet...")
    for i in range(3):
        dummy = torch.randn(num_devices, 640, 640, 3, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(dummy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
        ttnn.synchronize_device(mesh_device)
        print(f"   Warmup {i+1}/3")
    
    # Warmup SFace
    print("\n🔥 Warming up SFace...")
    for i in range(3):
        dummy = torch.randn(num_devices, 112, 112, 3, dtype=torch.float32)
        tt_input = ttnn.from_torch(dummy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        output = sface_model(tt_input)
        ttnn.synchronize_device(mesh_device)
        print(f"   Warmup {i+1}/3")
    
    # Benchmark YuNet only
    print("\n⚡ Benchmarking YuNet only (50 iterations)...")
    yunet_times = []
    for i in range(50):
        dummy = torch.randn(num_devices, 640, 640, 3, dtype=torch.bfloat16)
        t0 = time.time()
        tt_input = ttnn.from_torch(dummy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        yunet_times.append(t1 - t0)
        if (i + 1) % 25 == 0:
            avg = sum(yunet_times[-25:]) / 25
            print(f"   Iter {i+1}: {num_devices/avg:.0f} FPS")
    
    # Benchmark SFace only
    print("\n⚡ Benchmarking SFace only (50 iterations)...")
    sface_times = []
    for i in range(50):
        dummy = torch.randn(num_devices, 112, 112, 3, dtype=torch.float32)
        t0 = time.time()
        tt_input = ttnn.from_torch(dummy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        output = sface_model(tt_input)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        sface_times.append(t1 - t0)
        if (i + 1) % 25 == 0:
            avg = sum(sface_times[-25:]) / 25
            print(f"   Iter {i+1}: {num_devices/avg:.0f} FPS")
    
    # Benchmark Combined (YuNet + SFace) - single sync at end
    print("\n⚡ Benchmarking YuNet + SFace combined (50 iterations)...")
    combined_times = []
    
    # Pre-create inputs to remove allocation from timing
    yunet_inputs = [torch.randn(num_devices, 640, 640, 3, dtype=torch.bfloat16) for _ in range(50)]
    sface_inputs = [torch.randn(num_devices, 112, 112, 3, dtype=torch.float32) for _ in range(50)]
    
    for i in range(50):
        t0 = time.time()
        
        # YuNet - no sync, let it pipeline
        tt_yunet = ttnn.from_torch(yunet_inputs[i], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        cls_out, box_out, obj_out, kpt_out = yunet_model(tt_yunet)
        
        # SFace - runs after YuNet completes on device
        tt_sface = ttnn.from_torch(sface_inputs[i], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=inputs_mesh_mapper)
        output = sface_model(tt_sface)
        
        # Single sync at end
        ttnn.synchronize_device(mesh_device)
        
        t1 = time.time()
        combined_times.append(t1 - t0)
        if (i + 1) % 25 == 0:
            avg = sum(combined_times[-25:]) / 25
            print(f"   Iter {i+1}: {num_devices/avg:.0f} FPS")
    
    # Final stats
    yunet_avg = sum(yunet_times[5:]) / len(yunet_times[5:])
    sface_avg = sum(sface_times[5:]) / len(sface_times[5:])
    combined_avg = sum(combined_times[5:]) / len(combined_times[5:])
    
    print("\n" + "="*60)
    print("  RESULTS - Data Parallel Face Recognition")
    print("="*60)
    print(f"  Devices:           {num_devices}")
    print(f"")
    print(f"  YuNet only:")
    print(f"    Total FPS:       {num_devices/yunet_avg:.0f}")
    print(f"    Batch latency:   {1000*yunet_avg:.1f}ms")
    print(f"")
    print(f"  SFace only:")
    print(f"    Total FPS:       {num_devices/sface_avg:.0f}")
    print(f"    Batch latency:   {1000*sface_avg:.1f}ms")
    print(f"")
    print(f"  YuNet + SFace (full pipeline):")
    print(f"    Total FPS:       {num_devices/combined_avg:.0f}")
    print(f"    Batch latency:   {1000*combined_avg:.1f}ms")
    print("="*60)
    
    # Cleanup
    ttnn.close_mesh_device(mesh_device)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
