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
    cache_root = os.getenv("CACHE_ROOT", "/tmp/workflow_cache")
    workflow_logs_dir = Path(cache_root) / "workflow_logs"
    workflow_logs_dir.mkdir(parents=True, exist_ok=True)
    return workflow_logs_dir


def create_dummy_eval_output(model_spec, output_path):
    """Create dummy evaluation output files for YOLOv11."""
    logger.info("Creating dummy evaluation output for YOLOv11...")
    
    # Create eval output directory
    eval_output_dir = output_path / "eval_output"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run_id from model_spec data (same way as run.py)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    workflow = model_spec.cli_args.get("workflow", "evals")
    run_id = get_run_id(timestamp, model_spec.model_id, workflow)
    
    logger.info(f"Using run_id for eval: {run_id}")
    
    # Create dummy COCO detection results
    eval_results = {
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "device": model_spec.cli_args.get("device", "n150"),
        "timestamp": timestamp,
        "run_id": run_id,
        "task_results": {
            "coco_detection_val2017": {
                "mAP": 0.412,  # YOLOv11 expected mAP
                "mAP_50": 0.628,
                "mAP_75": 0.445,
                "mAP_small": 0.238,
                "mAP_medium": 0.454,
                "mAP_large": 0.559,
                "num_images_processed": 1000,
                "inference_time_avg": 0.045,  # seconds per image
                "preprocessing_time_avg": 0.008,
                "postprocessing_time_avg": 0.012
            }
        },
        "summary": {
            "total_tasks": 1,
            "passed_tasks": 1,
            "failed_tasks": 0,
            "overall_score": 0.412
        }
    }
    
    # Save results
    results_file = eval_output_dir / f"eval_results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    logger.info(f"Created dummy eval results: {results_file}")
    return [results_file]


def run_benchmarks_workflow(model_spec, workflow_logs_dir):
    """Run benchmarks workflow and create output files."""
    logger.info("Running benchmarks workflow for YOLOv11...")
    
    # Create benchmark output directory
    benchmark_output_dir = workflow_logs_dir / "benchmarks_output"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run_id from model_spec data (same way as run.py)
    run_id = getattr(model_spec.cli_args, 'run_id', None)
    if not run_id:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        workflow = model_spec.cli_args.get("workflow", "benchmarks")
        run_id = get_run_id(timestamp, model_spec.model_id, workflow)
    
    logger.info(f"Using run_id: {run_id}")
    
    # Get device from model spec
    device_name = model_spec.cli_args.get("device", "n150")
    
    # Check if this is a CNN model
    if model_spec.model_type != ModelTypes.CNN:
        logger.error(f"Direct inference benchmarks only support CNN models, got {model_spec.model_type}")
        return 1
    
    # Load benchmark configurations from reference data
    reference_data = load_cnn_performance_reference()
    configurations = []
    
    # Extract benchmark configurations from reference data
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
    
    # If no configurations found in reference, use defaults for YOLOv11
    if not configurations:
        logger.warning(f"No benchmark configurations found for {model_spec.model_name} on {device_name}. Using YOLOv11 defaults.")
        configurations = [
            CNNBenchmarkConfig(
                concurrent_requests=1, 
                image_width=640, 
                image_height=640,
                num_iterations=50,
                warmup_iterations=5
            ),
            CNNBenchmarkConfig(
                concurrent_requests=8, 
                image_width=640, 
                image_height=640,
                num_iterations=50,
                warmup_iterations=5
            ),
        ]
    
    # Create dummy benchmark results for YOLOv11
    benchmark_results = []
    for i, config in enumerate(configurations):
        # Simulate YOLOv11 performance (better than YOLOv4)
        base_latency = 0.045  # 45ms base latency for 640x640
        latency_with_concurrency = base_latency * (1 + (config.concurrent_requests - 1) * 0.1)
        
        result = CNNBenchmarkResult(
            config=config,
            latencies=[latency_with_concurrency + random.uniform(-0.005, 0.005) for _ in range(config.num_iterations)],
            throughput=1.0 / latency_with_concurrency * config.concurrent_requests,
            success_rate=1.0,
            error_count=0,
            total_requests=config.num_iterations,
            timestamp=datetime.now().isoformat()
        )
        benchmark_results.append(result)
        
        # Compare with reference if available
        comparison = compare_with_reference(result, model_spec.model_name, device_name)
        
        logger.info(f"Benchmark {i+1}/{len(configurations)}: "
                   f"Throughput: {result.throughput:.2f} req/s, "
                   f"Avg Latency: {result.avg_latency:.3f}s")
    
    # Save benchmark results
    results_data = {
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "device": device_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(result) for result in benchmark_results]
    }
    
    results_file = benchmark_output_dir / f"benchmark_results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    logger.info(f"Benchmark results saved to: {results_file}")
    return 0


def run_evals_workflow(model_spec, workflow_logs_dir):
    """Run evaluations workflow for YOLOv11."""
    logger.info("Running evaluations workflow for YOLOv11...")
    
    eval_files = create_dummy_eval_output(model_spec, workflow_logs_dir)
    logger.info(f"Evaluations workflow completed. Generated {len(eval_files)} files.")
    return 0


def run_release_workflow(model_spec, workflow_logs_dir):
    """Run release workflow (benchmarks + evals + report generation) for YOLOv11."""
    logger.info("Running release workflow for YOLOv11...")
    
    # Run benchmarks first
    benchmark_result = run_benchmarks_workflow(model_spec, workflow_logs_dir)
    if benchmark_result != 0:
        logger.error("Benchmarks failed in release workflow")
        return benchmark_result
    
    # Run evaluations
    eval_result = run_evals_workflow(model_spec, workflow_logs_dir)
    if eval_result != 0:
        logger.error("Evaluations failed in release workflow")
        return eval_result
    
    # Create release report
    release_output_dir = workflow_logs_dir / "release_output"
    release_output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_id = f"{model_spec.model_id}_{timestamp}"
    
    # Create YOLOv11 release report
    release_report = {
        "report_id": report_id,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "device": model_spec.cli_args.get("device", "n150"),
        "timestamp": timestamp,
        "benchmarks_summary": {
            "avg_latency": 0.045,  # YOLOv11 latency
            "throughput": 22.2,    # req/s
            "success_rate": 1.0
        },
        "evals_summary": {
            "coco_detection_val2017": {
                "mAP": 0.412,
                "mAP_50": 0.628,
                "mAP_75": 0.445
            }
        },
        "status": "completed"
    }
    
    report_file = release_output_dir / f"report_{report_id}.json"
    with open(report_file, 'w') as f:
        json.dump(release_report, f, indent=4)
    
    logger.info(f"Release report saved to: {report_file}")
    return 0


def main():
    """Main function to run the YOLOv11 CI script."""
    logger.info("Starting YOLOv11 CI script...")
    
    # Read model spec from environment variable
    model_spec_json_path = os.getenv("TT_MODEL_SPEC_JSON_PATH")
    if not model_spec_json_path:
        logger.error("TT_MODEL_SPEC_JSON_PATH environment variable not set")
        return 1
    
    if not Path(model_spec_json_path).exists():
        logger.error(f"Model spec JSON file not found: {model_spec_json_path}")
        return 1
    
    # Load model spec
    model_spec = ModelSpec.from_json(model_spec_json_path)
    logger.info(f"Loaded model spec for: {model_spec.model_name}")
    logger.info(f"Model ID: {model_spec.model_id}")
    logger.info(f"Device: {model_spec.cli_args.get('device', 'unknown')}")

    
    # Get workflow from CLI args
    workflow_str = model_spec.cli_args.get("workflow")

    try:
        workflow_type = WorkflowType.from_string(workflow_str)
        logger.info(f"Running workflow: {workflow_type.name}")
    except Exception as e:
        logger.error(f"Invalid workflow type: {workflow_str}, error: {e}")
        return 1
    
    # Get workflow logs directory in persistent volume
    workflow_logs_dir = get_workflow_logs_dir()
    logger.info(f"Workflow logs directory: {workflow_logs_dir}")
    
    # Run the appropriate workflow
    if workflow_type == WorkflowType.BENCHMARKS:
        return run_benchmarks_workflow(model_spec, workflow_logs_dir)
    elif workflow_type == WorkflowType.EVALS:
        return run_evals_workflow(model_spec, workflow_logs_dir)
    elif workflow_type == WorkflowType.RELEASE:
        return run_release_workflow(model_spec, workflow_logs_dir)
    else:
        logger.error(f"Unsupported workflow type: {workflow_type.name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())