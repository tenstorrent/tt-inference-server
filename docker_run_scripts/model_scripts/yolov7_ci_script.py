#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import time
import base64
import io
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent / "app"
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

# Add the real project root to access benchmarking modules
real_project_root = Path(__file__).resolve().parent.parent.parent
if real_project_root not in sys.path:
    sys.path.insert(0, str(real_project_root))

from workflows.model_spec import ModelSpec, ModelTypes
from workflows.workflow_types import WorkflowType, DeviceTypes
from workflows.utils import get_run_id
from benchmarking.cnn_benchmark_utils import (
    CNNBenchmarkConfig,
    CNNBenchmarkResult,
    generate_test_image,
    calculate_percentiles,
    load_cnn_performance_reference,
    compare_with_reference,
)
from benchmarking.benchmark_config import BENCHMARK_CONFIGS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_workflow_logs_dir():
    """Get the workflow logs directory from CACHE_ROOT environment variable."""
    cache_root = os.getenv("CACHE_ROOT", "/home/container_app_user/cache_root")
    workflow_logs_dir = Path(cache_root) / "workflow_logs"
    try:
        workflow_logs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission error creating {workflow_logs_dir}: {e}")
        logger.error(f"Parent directory permissions: {oct(workflow_logs_dir.stat().st_mode)}")
        logger.error(f"Parent directory owner: {workflow_logs_dir.stat().st_uid}")
        raise
    
    return workflow_logs_dir


class DirectInferenceImageClient:
    """Mock ImageClient that runs YOLOv7 inference directly instead of HTTP calls."""
    
    def __init__(self, model_name: str, device_name: str, **kwargs):
        self.model_name = model_name
        self.device_name = device_name
        self.model = None
        self.inference_count = 0

    def get_health(self) -> Tuple[bool, str]:
        """Mock health check - always returns healthy."""
        raise NotImplementedError("get_health not implemented")
    
    def search_image(self, image_data: str):
        """Mock image search that simulates YOLOv7 inference."""
        raise NotImplementedError("search_image not implemented")


class MockDirectInferenceImageClient(DirectInferenceImageClient):
    """Mock ImageClient that runs YOLOv7 inference directly instead of HTTP calls."""
    
    def __init__(self, model_name: str, device_name: str, **kwargs):
        super().__init__(model_name, device_name, **kwargs)
    
    def get_health(self) -> Tuple[bool, str]:
        """Mock health check - always returns healthy."""
        return True, "direct_inference"
    
    def search_image(self, image_data: str):
        """Mock image search that simulates YOLOv7 inference."""
        base_latency = 1.0  # ms
        
        latency_variation = random.uniform(0.9, 1.1)
        simulated_latency = base_latency * latency_variation
        
        start_time = time.time()
        time.sleep(simulated_latency / 1000.0)
        
        self.inference_count += 1
        
        class MockResponse:
            def __init__(self, status_code, json_data, elapsed_time):
                self.status_code = status_code
                self._json_data = json_data
                self.elapsed = MockElapsed(elapsed_time)
            
            def json(self):
                return self._json_data
        
        class MockElapsed:
            def __init__(self, seconds):
                self.total_seconds_val = seconds
                
            def total_seconds(self):
                return self.total_seconds_val
        
        mock_results = {
            "detections": [
                {
                    "class": "person",
                    "confidence": 0.95,
                    "bbox": [100, 200, 300, 400]
                },
                {
                    "class": "car",
                    "confidence": 0.87,
                    "bbox": [400, 100, 200, 150]
                }
            ],
            "inference_time_ms": simulated_latency,
            "model": self.model_name
        }
        
        elapsed_time = time.time() - start_time
        return MockResponse(200, mock_results, elapsed_time)


def benchmark_single_request_direct(
    client: DirectInferenceImageClient,
    image_base64: str,
    request_id: int
) -> float:
    """Run a single benchmark request using direct inference."""
    try:
        start_time = time.time()
        response = client.search_image(image_base64)
        end_time = time.time()
        
        if response.status_code == 200:
            latency_ms = (end_time - start_time) * 1000
            logger.debug(f"Request {request_id} completed in {latency_ms:.2f}ms")
            return latency_ms
        else:
            logger.error(f"Request {request_id} failed with status {response.status_code}")
            return -1
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        return -1


def run_benchmark_iteration_direct(
    client: DirectInferenceImageClient,
    config: CNNBenchmarkConfig,
    iteration_num: int,
    test_image: str,
    is_warmup: bool = False
) -> List[float]:
    """Run a single iteration of the benchmark with concurrent requests using direct inference."""
    latencies = []
    
    with ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
        futures = []
        for i in range(config.concurrent_requests):
            future = executor.submit(
                benchmark_single_request_direct,
                client,
                test_image,
                i + (iteration_num * config.concurrent_requests)
            )
            futures.append(future)
        
        for future in as_completed(futures):
            latency = future.result()
            if latency > 0:
                latencies.append(latency)
            else:
                logger.warning(f"Skipping failed request in iteration {iteration_num}")
    
    return latencies


def save_llm_compatible_benchmark_result(
    result: CNNBenchmarkResult,
    config: CNNBenchmarkConfig,
    model_spec,
    output_path: str,
    device_name: str,
    run_id: str
) -> str:
    """Save individual benchmark result in format compatible with LLM reporting system."""
    from datetime import datetime
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    isl = config.image_width
    osl = config.image_height
    max_concurrency = config.concurrent_requests
    num_requests = result.total_requests
    
    result_filename = (
        Path(output_path) 
        / f"benchmark_{run_id}_c{max_concurrency}_w{config.image_width}_h{config.image_height}.json"
    )
    
    benchmark_data = {
        "timestamp": run_timestamp,
        "model_name": model_spec.model_id,
        "model_id": model_spec.hf_model_repo,
        "backend": model_spec.impl.impl_name,
        "device": device_name,
        "max_con": str(max_concurrency),
        "mean_fps_user": result.fps,
        "std_fps_user": result.fps * 0.05,
        "mean_fps_batch": result.throughput_images_per_sec,
        "std_fps_batch": result.throughput_images_per_sec * 0.05,
        "num_images": str(config.concurrent_requests),
        "num_requests": str(num_requests),
        "filename": result_filename.name,
        "task_type": "image"
    }
    
    result_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(result_filename, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    logger.info(f"LLM-compatible benchmark result saved: {result_filename}")
    return str(result_filename)


def run_cnn_benchmark_direct(
    client: DirectInferenceImageClient,
    config: CNNBenchmarkConfig,
    output_path: str,
    model_name: str,
    model_spec=None,
    device_name: str = "n150",
    run_id: str = None
) -> CNNBenchmarkResult:
    """Run a complete CNN benchmark using direct inference."""
    logger.info(f"Starting CNN benchmark for {model_name}")
    logger.info(f"Configuration: {config}")
    
    logger.info(f"Generating test image ({config.image_width}x{config.image_height})...")
    test_image = generate_test_image(config.image_width, config.image_height)
    logger.info("Test image generated and cached for reuse")
    
    if config.warmup_iterations > 0:
        logger.info(f"Running {config.warmup_iterations} warmup iterations...")
        for i in range(config.warmup_iterations):
            run_benchmark_iteration_direct(client, config, i, test_image, is_warmup=True)
        logger.info("Warmup complete")
    
    logger.info(f"Running {config.num_iterations} benchmark iterations...")
    all_latencies = []
    start_time = time.time()
    
    for i in range(config.num_iterations):
        iteration_latencies = run_benchmark_iteration_direct(client, config, i, test_image)
        all_latencies.extend(iteration_latencies)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{config.num_iterations} iterations")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    total_requests = len(all_latencies)
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p50, p95, p99 = calculate_percentiles(all_latencies)
    
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
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
    
    if model_spec and run_id:
        save_llm_compatible_benchmark_result(result, config, model_spec, output_path, device_name, run_id)
    
    result_file = Path(output_path) / f"cnn_benchmark_{model_name}_c{config.concurrent_requests}_w{config.image_width}_h{config.image_height}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {result_file}")
    logger.info(f"Average latency: {avg_latency:.2f}ms, FPS: {fps:.2f}, Throughput: {throughput:.2f} images/sec")
    
    return result


def run_cnn_benchmark_sweep_direct(
    model_name: str,
    device_name: str,
    output_path: str,
    model_spec=None,
    run_id: str = None,
    configurations: Optional[List[CNNBenchmarkConfig]] = None
) -> List[Tuple[CNNBenchmarkResult, Dict[str, Any]]]:
    """Run a sweep of CNN benchmarks using direct inference."""
    if configurations is None:
        configurations = [
            CNNBenchmarkConfig(concurrent_requests=1, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=1, image_width=640, image_height=640),
            CNNBenchmarkConfig(concurrent_requests=8, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=8, image_width=640, image_height=640),
            CNNBenchmarkConfig(concurrent_requests=32, image_width=320, image_height=320),
            CNNBenchmarkConfig(concurrent_requests=32, image_width=640, image_height=640),
        ]
    
    client = MockDirectInferenceImageClient(model_name=model_name, device_name=device_name)
    
    logger.info("Initializing direct inference client...")
    health_status, runner_in_use = client.get_health()
    if not health_status:
        raise RuntimeError("Direct inference client initialization failed")
    logger.info(f"Direct inference client ready. Mode: {runner_in_use}")
    
    results = []
    for config in configurations:
        logger.info(f"\nRunning benchmark with config: concurrent={config.concurrent_requests}, size={config.image_width}x{config.image_height}")
        
        try:
            result = run_cnn_benchmark_direct(client, config, output_path, model_name, model_spec, device_name, run_id)
            comparison = compare_with_reference(result, model_name, device_name)
            results.append((result, comparison))
            
            if comparison["has_reference"]:
                logger.info(f"Performance vs theoretical: {comparison['fps_ratio']:.2%}")
                logger.info(f"Meets functional: {comparison['meets_functional']}")
                logger.info(f"Meets complete: {comparison['meets_complete']}")
                logger.info(f"Meets target: {comparison['meets_target']}")
        
        except Exception as e:
            logger.error(f"Benchmark failed for configuration {config}: {e}")
            continue
    
    summary_file = Path(output_path) / f"cnn_benchmark_summary_{model_name}.json"
    summary_results = []
    
    for i, (result, comparison) in enumerate(results):
        if i < len(configurations):
            config = configurations[i]
            summary_results.append({
                "config": asdict(config),
                "result": asdict(result),
                "comparison": comparison
            })
    
    summary_data = {
        "model": model_name,
        "device": device_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": summary_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"\nBenchmark sweep complete. Summary saved to {summary_file}")
    
    return results


def create_dummy_eval_output(model_spec, output_path):
    """Create dummy evaluation output files matching run_evals.py structure."""
    logger.info("Creating dummy evaluation output...")
    
    eval_output_dir = output_path / "evals_output"
    eval_model_dir = eval_output_dir / f"eval_{model_spec.model_id}"
    
    if model_spec.model_type.name == "CNN":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_run_dir = eval_output_dir / f"eval_{model_spec.model_name}_n150_{timestamp}_n500"
        eval_run_dir.mkdir(parents=True, exist_ok=True)
        
        vis_dir = eval_run_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        coco_metrics = {
            "bbox": {
                "AP": 0.412,
                "AP50": 0.628,
                "AP75": 0.445,
                "APs": 0.238,
                "APm": 0.456,
                "APl": 0.531
            },
            "metadata": {
                "model": model_spec.model_name,
                "dataset": "coco_detection_val2017",
                "timestamp": timestamp,
                "device": model_spec.cli_args.get("device", "n150"),
                "workflow": "evals",
                "images_evaluated": 500
            }
        }
        
        coco_results_file = eval_run_dir / "coco_detection_val2017_metrics.json"
        with open(coco_results_file, 'w') as f:
            json.dump(coco_metrics, f, indent=2)
        
        for i in range(5):
            dummy_vis_file = vis_dir / f"image_{i*100}_detections.png"
            dummy_vis_file.write_text("dummy_image_data")
        
        logger.info(f"Created COCO evaluation results: {coco_results_file}")
        return [coco_results_file]
    
    else:
        eval_model_dir.mkdir(parents=True, exist_ok=True)
        hf_repo_dir = eval_model_dir / model_spec.hf_model_repo.replace('/', '__')
        hf_repo_dir.mkdir(parents=True, exist_ok=True)
        
        eval_results = {
            "results": {
                "hellaswag": {
                    "acc": 0.85,
                    "acc_stderr": 0.01,
                    "acc_norm": 0.87,
                    "acc_norm_stderr": 0.01
                },
                "mmlu": {
                    "acc": 0.72,
                    "acc_stderr": 0.02
                }
            },
            "versions": {
                "hellaswag": 1,
                "mmlu": 1
            },
            "config": {
                "model": model_spec.hf_model_repo,
                "model_args": f"model={model_spec.hf_model_repo},base_url=http://127.0.0.1:8000/v1/completions",
                "batch_size": "1",
                "device": model_spec.cli_args.get("device", "n150"),
                "workflow": "evals"
            }
        }
        
        results_file = hf_repo_dir / f"results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        logger.info(f"Created evaluation results: {results_file}")
        return [results_file]


def run_benchmarks_workflow(model_spec, workflow_logs_dir):
    """Run benchmarks workflow and create output files."""
    logger.info("Running benchmarks workflow...")
    
    benchmark_output_dir = workflow_logs_dir / "benchmarks_output"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = getattr(model_spec.cli_args, 'run_id', None)
    if not run_id:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        workflow = model_spec.cli_args.get("workflow", "benchmarks")
        run_id = get_run_id(timestamp, model_spec.model_id, workflow)
    
    logger.info(f"Using run_id: {run_id}")
    
    device_name = model_spec.cli_args.get("device", "n150")
    
    if model_spec.model_type != ModelTypes.CNN:
        logger.error(f"Direct inference benchmarks only support CNN models, got {model_spec.model_type}")
        return 1
    
    reference_data = load_cnn_performance_reference()
    configurations = []
    
    if model_spec.model_name in reference_data:
        model_references = reference_data[model_spec.model_name].get(device_name.lower(), [])
        
        for ref in model_references:
            config = CNNBenchmarkConfig(
                concurrent_requests=ref.get("concurrent_requests", 1),
                image_width=ref.get("image_width", 640),
                image_height=ref.get("image_height", 640),
                num_iterations=100,
                warmup_iterations=10,
            )
            configurations.append(config)
    
    if not configurations:
        logger.warning(
            f"No benchmark configurations found for {model_spec.model_name} on {device_name}. Using defaults."
        )
        configurations = [
            CNNBenchmarkConfig(
                concurrent_requests=1, image_width=640, image_height=640
            ),
            CNNBenchmarkConfig(
                concurrent_requests=8, image_width=640, image_height=640
            ),
        ]
    
    try:
        results = run_cnn_benchmark_sweep_direct(
            model_name=model_spec.model_name,
            device_name=device_name,
            output_path=str(benchmark_output_dir),
            model_spec=model_spec,
            run_id=run_id,
            configurations=configurations
        )
        
        logger.info(
            f"\nBenchmark Summary for {model_spec.model_name} on {device_name}:"
        )
        logger.info(f"{'Config':<30} {'FPS':<10} {'Latency (ms)':<15} {'Status':<20}")
        logger.info("-" * 75)
        
        for result, comparison in results:
            config_str = f"c{result.concurrent_requests}_w{result.image_width}_h{result.image_height}"
            status = "N/A"
            if comparison["has_reference"]:
                if comparison["meets_target"]:
                    status = "MEETS TARGET ✅"
                elif comparison["meets_complete"]:
                    status = "COMPLETE 🟢"
                elif comparison["meets_functional"]:
                    status = "FUNCTIONAL 🟡"
                else:
                    status = "BELOW FUNCTIONAL ⛔"
            
            logger.info(
                f"{config_str:<30} {result.fps:<10.2f} {result.avg_latency_ms:<15.2f} {status:<20}"
            )
        
        logger.info(f"✅ Benchmarks workflow completed. Created {len(results)} benchmark results.")
        return 0
        
    except Exception as e:
        logger.error(f"⛔ Benchmarks workflow failed: {e}")
        return 1


def run_evals_workflow(model_spec, workflow_logs_dir):
    """Run evaluations workflow and create output files."""
    logger.info("Running evaluations workflow...")
    eval_files = create_dummy_eval_output(model_spec, workflow_logs_dir)
    logger.info(f"✅ Evaluations workflow completed. Created {len(eval_files)} files.")
    return 0


def run_release_workflow(model_spec, workflow_logs_dir):
    """Run release workflow (benchmarks + evals + report generation)."""
    logger.info("Running release workflow...")
    
    benchmark_result = run_benchmarks_workflow(model_spec, workflow_logs_dir)
    if benchmark_result != 0:
        logger.error("Benchmarks failed in release workflow")
        return benchmark_result
    
    eval_result = run_evals_workflow(model_spec, workflow_logs_dir)
    if eval_result != 0:
        logger.error("Evaluations failed in release workflow")
        return eval_result
    
    release_output_dir = workflow_logs_dir / "release_output"
    release_output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_id = f"{model_spec.model_id}_{timestamp}"
    
    release_report = {
        "report_id": report_id,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "device": model_spec.cli_args.get("device", "n150"),
        "timestamp": timestamp,
        "benchmarks_summary": {
            "ttft_avg": 0.12,
            "tpot_avg": 0.06,
            "e2el_avg": 3.2
        },
        "evals_summary": {
            "hellaswag_acc": 0.85,
            "mmlu_acc": 0.72,
            "coco_ap": 0.412 if model_spec.model_type.name == "CNN" else None
        },
        "status": "completed"
    }
    
    report_file = release_output_dir / f"report_{report_id}.json"
    with open(report_file, 'w') as f:
        json.dump(release_report, f, indent=4)
    
    logger.info(f"✅ Release workflow completed. Report saved: {report_file}")
    return 0


def main():
    """Main function to run the YOLOv7 CI script."""
    logger.info("Starting YOLOv7 CI script...")
    
    model_spec_json_path = os.getenv("TT_MODEL_SPEC_JSON_PATH")
    if not model_spec_json_path:
        logger.error("TT_MODEL_SPEC_JSON_PATH environment variable not set")
        return 1
    
    if not Path(model_spec_json_path).exists():
        logger.error(f"Model spec JSON file not found: {model_spec_json_path}")
        return 1
    
    model_spec = ModelSpec.from_json(model_spec_json_path)
    logger.info(f"Loaded model spec for: {model_spec.model_name}")
    logger.info(f"Model ID: {model_spec.model_id}")
    logger.info(f"Device: {model_spec.cli_args.get('device', 'unknown')}")

    workflow_str = model_spec.cli_args.get("workflow")

    try:
        workflow_type = WorkflowType.from_string(workflow_str)
        logger.info(f"Running workflow: {workflow_type.name}")
    except Exception as e:
        logger.error(f"Invalid workflow type: {workflow_str}, error: {e}")
        return 1
    
    workflow_logs_dir = get_workflow_logs_dir()
    logger.info(f"Workflow logs directory: {workflow_logs_dir}")
    
    try:
        if workflow_type == WorkflowType.BENCHMARKS:
            result = run_benchmarks_workflow(model_spec, workflow_logs_dir)
        elif workflow_type == WorkflowType.EVALS:
            result = run_evals_workflow(model_spec, workflow_logs_dir)
        elif workflow_type == WorkflowType.RELEASE:
            result = run_release_workflow(model_spec, workflow_logs_dir)
        else:
            logger.error(f"Unsupported workflow type: {workflow_type.name}")
            result = 1
    finally:
        # Clean up YOLOv7 postprocess directories after workflow completes
        if model_spec.model_name.lower() == "yolov7":
            from tt_model_runners.yolov7_runner import TTYolov7Runner
            TTYolov7Runner.cleanup_postprocess_directories()
    
    return result


if __name__ == "__main__":
    sys.exit(main())
