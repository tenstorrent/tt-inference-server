# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Latency benchmarks for TT-Comfy Bridge IPC.

Validates that Unix socket + msgpack overhead is <5ms.
"""

import time
import asyncio
import struct
import msgpack
from pathlib import Path
import statistics


async def benchmark_unix_socket_roundtrip(num_iterations=1000):
    """
    Benchmark Unix socket roundtrip latency.
    
    Tests the time to send a message and receive a response.
    Target: <5ms average latency.
    """
    socket_path = "/tmp/benchmark-tt-comfy.sock"
    
    # Simple echo server
    async def echo_server(reader, writer):
        while True:
            try:
                # Read length
                length_bytes = await reader.readexactly(4)
                length = struct.unpack('>I', length_bytes)[0]
                
                # Read data
                data = await reader.readexactly(length)
                
                # Echo back
                writer.write(length_bytes + data)
                await writer.drain()
            except:
                break
        writer.close()
        await writer.wait_closed()
    
    # Start server
    if Path(socket_path).exists():
        Path(socket_path).unlink()
    
    server = await asyncio.start_unix_server(echo_server, path=socket_path)
    
    # Give server time to start
    await asyncio.sleep(0.1)
    
    # Connect client
    reader, writer = await asyncio.open_unix_connection(socket_path)
    
    # Warmup
    for _ in range(10):
        msg = msgpack.packb({"test": "data"})
        length = struct.pack('>I', len(msg))
        writer.write(length + msg)
        await writer.drain()
        await reader.readexactly(4)
        msg_len = struct.unpack('>I', await reader.readexactly(4))[0]
        await reader.readexactly(msg_len)
    
    # Benchmark
    latencies = []
    
    for i in range(num_iterations):
        test_data = {"iteration": i, "test": "benchmark_data" * 10}
        msg = msgpack.packb(test_data)
        length = struct.pack('>I', len(msg))
        
        start = time.perf_counter()
        
        # Send
        writer.write(length + msg)
        await writer.drain()
        
        # Receive
        resp_length_bytes = await reader.readexactly(4)
        resp_length = struct.unpack('>I', resp_length_bytes)[0]
        resp_data = await reader.readexactly(resp_length)
        
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
    
    # Cleanup
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    Path(socket_path).unlink()
    
    # Results
    avg_latency = statistics.mean(latencies)
    p50_latency = statistics.median(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
    max_latency = max(latencies)
    
    print(f"\n{'='*60}")
    print(f"Unix Socket Roundtrip Latency Benchmark")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"Average:    {avg_latency:.3f} ms")
    print(f"Median:     {p50_latency:.3f} ms")
    print(f"P95:        {p95_latency:.3f} ms")
    print(f"P99:        {p99_latency:.3f} ms")
    print(f"Max:        {max_latency:.3f} ms")
    print(f"{'='*60}")
    
    # Validate target
    if avg_latency < 5.0:
        print(f"✅ PASS: Average latency {avg_latency:.3f}ms < 5ms target")
    else:
        print(f"❌ FAIL: Average latency {avg_latency:.3f}ms >= 5ms target")
    
    return avg_latency < 5.0


async def benchmark_msgpack_serialization(num_iterations=10000):
    """
    Benchmark msgpack serialization overhead.
    
    Tests the time to pack and unpack typical message payloads.
    """
    test_messages = [
        {"op": "init_model", "data": {"model_type": "sdxl", "device_id": "0"}},
        {
            "op": "full_inference",
            "data": {
                "model_id": "sdxl_0",
                "prompt": "A beautiful landscape" * 20,
                "negative_prompt": "low quality, blurry",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "seed": 42
            }
        },
        {
            "op": "encode_prompt",
            "data": {
                "model_id": "sdxl_0",
                "prompt": "Test prompt",
                "negative_prompt": "negative"
            }
        }
    ]
    
    pack_times = []
    unpack_times = []
    
    for msg in test_messages:
        for _ in range(num_iterations):
            # Pack
            start = time.perf_counter()
            packed = msgpack.packb(msg, use_bin_type=True)
            pack_time = (time.perf_counter() - start) * 1000
            pack_times.append(pack_time)
            
            # Unpack
            start = time.perf_counter()
            unpacked = msgpack.unpackb(packed, raw=False)
            unpack_time = (time.perf_counter() - start) * 1000
            unpack_times.append(unpack_time)
    
    avg_pack = statistics.mean(pack_times)
    avg_unpack = statistics.mean(unpack_times)
    avg_total = avg_pack + avg_unpack
    
    print(f"\n{'='*60}")
    print(f"Msgpack Serialization Benchmark")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations * len(test_messages)}")
    print(f"Avg Pack:   {avg_pack:.4f} ms")
    print(f"Avg Unpack: {avg_unpack:.4f} ms")
    print(f"Avg Total:  {avg_total:.4f} ms")
    print(f"{'='*60}")
    
    if avg_total < 0.1:
        print(f"✅ PASS: Serialization overhead {avg_total:.4f}ms is negligible")
    else:
        print(f"⚠️  WARNING: Serialization overhead {avg_total:.4f}ms may be significant")
    
    return avg_total < 0.1


def benchmark_shared_memory_transfer():
    """
    Benchmark shared memory tensor transfer overhead.
    
    Tests the time to transfer tensors via shared memory.
    """
    import torch
    from server.tensor_bridge import TensorBridge
    
    bridge = TensorBridge()
    
    # Test different tensor sizes
    test_tensors = [
        ("Small (1KB)", torch.randn(16, 16)),
        ("Medium (1MB)", torch.randn(512, 512)),
        ("Large (16MB)", torch.randn(2048, 2048)),
        ("XLarge (64MB)", torch.randn(4096, 4096)),
    ]
    
    print(f"\n{'='*60}")
    print(f"Shared Memory Tensor Transfer Benchmark")
    print(f"{'='*60}")
    
    for name, tensor in test_tensors:
        # Measure transfer to shared memory
        start = time.perf_counter()
        handle = bridge.tensor_to_shm(tensor)
        to_shm_time = (time.perf_counter() - start) * 1000
        
        # Measure reconstruction from shared memory
        start = time.perf_counter()
        reconstructed = bridge.shm_to_tensor(handle)
        from_shm_time = (time.perf_counter() - start) * 1000
        
        # Verify correctness
        assert torch.allclose(tensor, reconstructed), "Tensor mismatch after transfer"
        
        # Cleanup
        bridge.release_shm(handle)
        
        total_time = to_shm_time + from_shm_time
        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        print(f"{name:20} ({size_mb:.2f} MB):")
        print(f"  To SHM:    {to_shm_time:.3f} ms")
        print(f"  From SHM:  {from_shm_time:.3f} ms")
        print(f"  Total:     {total_time:.3f} ms")
        print(f"  Bandwidth: {size_mb / (total_time / 1000):.2f} MB/s")
    
    print(f"{'='*60}")
    print(f"✅ Shared memory transfer working correctly")
    
    bridge.cleanup_all()


async def run_all_benchmarks():
    """Run all latency benchmarks."""
    print("\n" + "="*60)
    print("TT-Comfy Bridge Latency Benchmarks")
    print("="*60)
    
    # 1. Unix socket roundtrip
    socket_pass = await benchmark_unix_socket_roundtrip(num_iterations=1000)
    
    # 2. Msgpack serialization
    msgpack_pass = await benchmark_msgpack_serialization(num_iterations=10000)
    
    # 3. Shared memory transfer
    benchmark_shared_memory_transfer()
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    
    if socket_pass and msgpack_pass:
        print(f"✅ ALL BENCHMARKS PASSED")
        print(f"   IPC overhead is within acceptable limits (<5ms)")
    else:
        print(f"❌ SOME BENCHMARKS FAILED")
        print(f"   IPC overhead may be too high for production use")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())

