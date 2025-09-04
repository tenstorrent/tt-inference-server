# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
from utils.audio_client import AudioClient
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioBenchmarkConfig:
    concurrent_requests: int = 1
    num_iterations: int = 100
    warmup_iterations: int = 10
    audio_duration_seconds: float = 5.0
    sample_rate: int = 16000
    timeout_seconds: int = 90


@dataclass
class AudioBenchmarkResult:
    concurrent_requests: int
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_requests_per_second: float
    audio_duration_seconds: float
    successful_requests: int
    failed_requests: int


def benchmark_single_audio_request(
    client: AudioClient,
    audio_base64: str,
    request_id: int
) -> float:
    """Execute a single audio transcription request and return latency in milliseconds."""
    start_time = time.time()
    
    try:
        response = client.transcribe_audio(audio_base64)
        if response.status_code != 200:
            logger.error(f"Audio request {request_id} failed with status {response.status_code}")
            return -1
        
        # Optionally validate response contains transcription
        try:
            response_data = response.json()
            if not response_data:
                logger.warning(f"Empty response for audio request {request_id}")
        except Exception:
            pass  # Response might be plain text
            
    except Exception as e:
        logger.error(f"Audio request {request_id} failed with error: {e}")
        return -1
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return latency_ms


def run_audio_benchmark_iteration(
    client: AudioClient,
    config: AudioBenchmarkConfig,
    iteration_num: int,
    is_warmup: bool = False
) -> List[float]:
    """Run a single benchmark iteration with concurrent audio requests."""
    
    latencies = []
    # Generate test audio for this iteration
    test_audio = client.generate_random_audio(
        duration_seconds=config.audio_duration_seconds, 
        sample_rate=config.sample_rate
    )
    
    with ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
        # Submit all concurrent requests
        futures = []
        for i in range(config.concurrent_requests):
            future = executor.submit(
                benchmark_single_audio_request,
                client,
                test_audio,
                i + (iteration_num * config.concurrent_requests)
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            latency = future.result()
            if latency > 0:  # Valid result
                latencies.append(latency)
            else:
                logger.warning(f"Skipping failed audio request in iteration {iteration_num}")
    
    return latencies


def calculate_percentiles(latencies: List[float]) -> Tuple[float, float, float]:
    """Calculate 50th, 95th, and 99th percentiles."""
    if not latencies:
        return 0.0, 0.0, 0.0
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    p50 = sorted_latencies[int(n * 0.5)]
    p95 = sorted_latencies[int(n * 0.95)]
    p99 = sorted_latencies[int(n * 0.99)]
    
    return p50, p95, p99


def run_audio_benchmark(
    client: AudioClient,
    config: AudioBenchmarkConfig,
    output_path: str,
    model_name: str
) -> AudioBenchmarkResult:
    """Run a complete audio benchmark with the given configuration."""
    logger.info(f"Starting Audio benchmark for {model_name}")
    logger.info(f"Configuration: {config}")
    
    # Warmup phase
    if config.warmup_iterations > 0:
        logger.info(f"Running {config.warmup_iterations} warmup iterations...")
        for i in range(config.warmup_iterations):
            run_audio_benchmark_iteration(client, config, i, is_warmup=True)
        logger.info("Warmup complete")
    
    # Benchmark phase
    logger.info(f"Running {config.num_iterations} benchmark iterations...")
    all_latencies = []
    start_time = time.time()
    
    for i in range(config.num_iterations):
        iteration_latencies = run_audio_benchmark_iteration(client, config, i)
        all_latencies.extend(iteration_latencies)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{config.num_iterations} iterations")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    total_requests = len(all_latencies)
    successful_requests = len([l for l in all_latencies if l > 0])
    failed_requests = total_requests - successful_requests
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p50, p95, p99 = calculate_percentiles(all_latencies)
    
    # Throughput is total requests processed per second
    throughput = total_requests / total_time if total_time > 0 else 0
    
    result = AudioBenchmarkResult(
        concurrent_requests=config.concurrent_requests,
        total_requests=total_requests,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        throughput_requests_per_second=throughput,
        audio_duration_seconds=config.audio_duration_seconds,
        successful_requests=successful_requests,
        failed_requests=failed_requests
    )
    
    logger.info(f"Audio Benchmark Results: {result}")
    return result


def compare_with_reference(
    result: AudioBenchmarkResult,
    model_name: str,
    device_name: str
) -> Dict[str, Any]:
    """Compare benchmark results with reference performance."""
    comparison = {
        "model": model_name,
        "device": device_name,
        "config": {
            "concurrent_requests": result.concurrent_requests,
            "audio_duration": result.audio_duration_seconds,
        },
        "results": {
            "throughput": result.throughput_requests_per_second,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "success_rate": result.successful_requests / result.total_requests if result.total_requests > 0 else 0,
        },
        "meets_expectations": result.successful_requests > result.total_requests * 0.95,  # 95% success rate
    }
    
    return comparison


def run_audio_benchmark_sweep(
    model_name: str,
    device_name: str,
    service_port: str,
    output_path: str,
    jwt_secret: Optional[str] = None,
    configurations: Optional[List[AudioBenchmarkConfig]] = None
) -> List[Tuple[AudioBenchmarkResult, Dict[str, Any]]]:
    """Run a sweep of audio benchmarks with different configurations."""
    # Default configurations if not provided
    if configurations is None:
        configurations = [
            AudioBenchmarkConfig(concurrent_requests=1, audio_duration_seconds=5.0),
            AudioBenchmarkConfig(concurrent_requests=32, audio_duration_seconds=5.0),
        ]
    
    # Initialize client
    client = AudioClient(
        base_url=f"http://localhost:{service_port}",
        jwt_secret=jwt_secret
    )
    
    # Wait for server to be healthy
    logger.info("Waiting for Audio server to be healthy...")
    health_status, runner_in_use = client.get_health()
    if not health_status:
        raise RuntimeError("Audio server is not healthy")
    logger.info(f"Audio server is healthy. Runner in use: {runner_in_use}")
    
    # Run benchmarks
    results = []
    for config in configurations:
        logger.info(f"Running audio benchmark with config: {config}")
        
        # Generate timestamped output file
        run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = (
            Path(output_path)
            / f"audio_benchmark_{model_name}_{run_timestamp}_concurrent-{config.concurrent_requests}_duration-{config.audio_duration_seconds}s.json"
        )
        
        result = run_audio_benchmark(client, config, str(result_filename), model_name)
        comparison = compare_with_reference(result, model_name, device_name)
        
        # Save detailed results
        detailed_results = {
            "config": config.__dict__,
            "results": result.__dict__,
            "comparison": comparison,
            "timestamp": run_timestamp,
            "model": model_name,
            "device": device_name,
            "task_type": "audio"
        }
        
        with open(result_filename, "w") as f:
            json.dump(detailed_results, f, indent=4)
        
        logger.info(f"Saved audio benchmark results to: {result_filename}")
        results.append((result, comparison))
    
    return results
