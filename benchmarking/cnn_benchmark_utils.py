# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Generic CNN benchmarking utilities for performance testing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.image_client import ImageClient

logger = logging.getLogger(__name__)


@dataclass
class CNNBenchmarkConfig:
    """Configuration for a CNN benchmark run."""
    concurrent_requests: int
    image_width: int
    image_height: int
    num_iterations: int = 100
    warmup_iterations: int = 10


@dataclass
class CNNBenchmarkResult:
    """Results from a CNN benchmark run."""
    concurrent_requests: int
    image_width: int
    image_height: int
    num_iterations: int
    total_requests: int
    total_time_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float
    throughput_images_per_sec: float


def generate_test_image(width: int, height: int) -> str:
    """Generate a test image as base64 string."""
    import base64
    from PIL import Image
    import io
    
    # Create a random RGB image
    import random
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64


def benchmark_single_request(
    client: ImageClient,
    image_base64: str,
    request_id: int
) -> float:
    """Execute a single benchmark request and return latency in milliseconds."""
    start_time = time.time()
    
    try:
        response = client.search_image(image_base64)
        if response.status_code != 200:
            logger.error(f"Request {request_id} failed with status {response.status_code}")
            return -1
    except Exception as e:
        logger.error(f"Request {request_id} failed with error: {e}")
        return -1
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return latency_ms


def run_benchmark_iteration(
    client: ImageClient,
    config: CNNBenchmarkConfig,
    iteration_num: int,
    is_warmup: bool = False
) -> List[float]:
    """Run a single iteration of the benchmark with concurrent requests."""
    latencies = []
    test_image = generate_test_image(config.image_width, config.image_height)
    
    with ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
        # Submit all concurrent requests
        futures = []
        for i in range(config.concurrent_requests):
            future = executor.submit(
                benchmark_single_request,
                client,
                test_image,
                i + (iteration_num * config.concurrent_requests)
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            latency = future.result()
            if latency > 0:  # Valid result
                latencies.append(latency)
            else:
                logger.warning(f"Skipping failed request in iteration {iteration_num}")
    
    return latencies


def calculate_percentiles(latencies: List[float]) -> Tuple[float, float, float]:
    """Calculate p50, p95, and p99 percentiles."""
    if not latencies:
        return 0.0, 0.0, 0.0
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    p50 = sorted_latencies[int(n * 0.50)]
    p95 = sorted_latencies[int(n * 0.95)]
    p99 = sorted_latencies[int(n * 0.99)]
    
    return p50, p95, p99


def run_cnn_benchmark(
    client: ImageClient,
    config: CNNBenchmarkConfig,
    output_path: str,
    model_name: str
) -> CNNBenchmarkResult:
    """Run a complete CNN benchmark with the given configuration."""
    logger.info(f"Starting CNN benchmark for {model_name}")
    logger.info(f"Configuration: {config}")
    
    # Warmup phase
    if config.warmup_iterations > 0:
        logger.info(f"Running {config.warmup_iterations} warmup iterations...")
        for i in range(config.warmup_iterations):
            run_benchmark_iteration(client, config, i, is_warmup=True)
        logger.info("Warmup complete")
    
    # Benchmark phase
    logger.info(f"Running {config.num_iterations} benchmark iterations...")
    all_latencies = []
    start_time = time.time()
    
    for i in range(config.num_iterations):
        iteration_latencies = run_benchmark_iteration(client, config, i)
        all_latencies.extend(iteration_latencies)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{config.num_iterations} iterations")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    total_requests = len(all_latencies)
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p50, p95, p99 = calculate_percentiles(all_latencies)
    
    # FPS is based on average latency for single request
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    # Throughput is total images processed per second
    throughput = total_requests / total_time if total_time > 0 else 0
    
    result = CNNBenchmarkResult(
        concurrent_requests=config.concurrent_requests,
        image_width=config.image_width,
        image_height=config.image_height,
        num_iterations=config.num_iterations,
        total_requests=total_requests,
        total_time_seconds=total_time,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        fps=fps,
        throughput_images_per_sec=throughput
    )
    
    # Save results
    result_file = Path(output_path) / f"cnn_benchmark_{model_name}_c{config.concurrent_requests}_w{config.image_width}_h{config.image_height}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {result_file}")
    logger.info(f"Average latency: {avg_latency:.2f}ms, FPS: {fps:.2f}, Throughput: {throughput:.2f} images/sec")
    
    return result


def load_cnn_performance_reference() -> Dict[str, List[Dict[str, Any]]]:
    """Load CNN performance reference targets from JSON file."""
    from workflows.utils import get_repo_root_path
    
    reference_file = get_repo_root_path() / "benchmarking" / "benchmark_targets" / "cnn_performance_reference.json"
    
    if not reference_file.exists():
        logger.warning(f"CNN performance reference file not found: {reference_file}")
        return {}
    
    with open(reference_file, 'r') as f:
        return json.load(f)


def compare_with_reference(
    result: CNNBenchmarkResult,
    model_name: str,
    device_name: str
) -> Dict[str, Any]:
    """Compare benchmark results with reference targets."""
    reference_data = load_cnn_performance_reference()
    
    if model_name not in reference_data:
        logger.warning(f"No reference data found for model: {model_name}")
        return {"has_reference": False}
    
    model_references = reference_data[model_name]
    
    # Find matching reference configuration
    matching_ref = None
    for ref in model_references:
        if (ref.get("device", "").lower() == device_name.lower() and
            ref.get("concurrent_requests") == result.concurrent_requests and
            ref.get("image_width") == result.image_width and
            ref.get("image_height") == result.image_height):
            matching_ref = ref
            break
    
    if not matching_ref:
        logger.warning(f"No matching reference configuration found for {model_name} on {device_name}")
        return {"has_reference": False}
    
    # Compare with theoretical targets
    theoretical_fps = matching_ref.get("targets", {}).get("theoretical", {}).get("fps", 0)
    
    comparison = {
        "has_reference": True,
        "theoretical_fps": theoretical_fps,
        "actual_fps": result.fps,
        "fps_ratio": result.fps / theoretical_fps if theoretical_fps > 0 else 0,
        "meets_functional": result.fps >= theoretical_fps * 0.1 if theoretical_fps > 0 else False,
        "meets_complete": result.fps >= theoretical_fps * 0.5 if theoretical_fps > 0 else False,
        "meets_target": result.fps >= theoretical_fps * 1.0 if theoretical_fps > 0 else False,
    }
    
    return comparison


def run_cnn_benchmark_sweep(
    model_name: str,
    device_name: str,
    service_port: str,
    output_path: str,
    jwt_secret: Optional[str] = None,
    configurations: Optional[List[CNNBenchmarkConfig]] = None
) -> List[Tuple[CNNBenchmarkResult, Dict[str, Any]]]:
    """Run a sweep of CNN benchmarks with different configurations."""
    # Default configurations if not provided
    if configurations is None:
        configurations = [
            CNNBenchmarkConfig(concurrent_requests=1, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=1, image_width=640, image_height=640),
            CNNBenchmarkConfig(concurrent_requests=8, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=8, image_width=640, image_height=640),
            CNNBenchmarkConfig(concurrent_requests=32, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=32, image_width=640, image_height=640),
        ]
    
    # Initialize client
    client = ImageClient(
        base_url=f"http://localhost:{service_port}",
        jwt_secret=jwt_secret
    )
    
    # Wait for server to be healthy
    logger.info("Waiting for CNN server to be healthy...")
    health_status, runner_in_use = client.get_health()
    if not health_status:
        raise RuntimeError("CNN server is not healthy")
    logger.info(f"CNN server is healthy. Runner in use: {runner_in_use}")
    
    # Run benchmarks
    results = []
    for config in configurations:
        logger.info(f"\nRunning benchmark with configuration: {config}")
        
        try:
            result = run_cnn_benchmark(client, config, output_path, model_name)
            comparison = compare_with_reference(result, model_name, device_name)
            results.append((result, comparison))
            
            # Log comparison results
            if comparison["has_reference"]:
                logger.info(f"Performance vs theoretical: {comparison['fps_ratio']:.2%}")
                logger.info(f"Meets functional: {comparison['meets_functional']}")
                logger.info(f"Meets complete: {comparison['meets_complete']}")
                logger.info(f"Meets target: {comparison['meets_target']}")
        
        except Exception as e:
            logger.error(f"Benchmark failed for configuration {config}: {e}")
            continue
    
    # Save summary
    summary_file = Path(output_path) / f"cnn_benchmark_summary_{model_name}.json"
    summary_data = {
        "model": model_name,
        "device": device_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "config": asdict(config),
                "result": asdict(result),
                "comparison": comparison
            }
            for (result, comparison), config in zip(results, configurations)
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"\nBenchmark sweep complete. Summary saved to {summary_file}")
    
    return results
