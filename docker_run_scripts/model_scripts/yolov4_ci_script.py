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
from time import time

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent / "app"
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_spec import ModelSpec
from workflows.workflow_types import WorkflowType

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
        # Try to provide more helpful error information
        logger.error(f"Parent directory permissions: {oct(workflow_logs_dir.stat().st_mode)}")
        logger.error(f"Parent directory owner: {workflow_logs_dir.stat().st_uid}")
        raise

    
    return workflow_logs_dir


def create_dummy_benchmark_output(model_spec, output_path):
    """Create dummy benchmark output files matching run_benchmarks.py structure."""
    logger.info("Creating dummy benchmark output...")
    
    # Create benchmark output directory structure
    benchmark_output_dir = output_path / "benchmarks_output"
    try:
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission error creating {benchmark_output_dir}: {e}")
        # Try to provide more helpful error information
        logger.error(f"Parent directory permissions: {oct(output_path.stat().st_mode)}")
        logger.error(f"Parent directory owner: {output_path.stat().st_uid}")
        raise
    
    
    # Create dummy benchmark data matching the structure from run_benchmarks.py
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create multiple benchmark files to simulate different parameter combinations
    benchmark_configs = [
        {"isl": 128, "osl": 128, "max_concurrency": 1, "num_prompts": 100},
        {"isl": 256, "osl": 256, "max_concurrency": 4, "num_prompts": 100},
        {"isl": 512, "osl": 512, "max_concurrency": 8, "num_prompts": 50},
    ]
    
    benchmark_files = []
    for config in benchmark_configs:
        filename = (
            benchmark_output_dir / 
            f"benchmark_{model_spec.model_id}_{run_timestamp}_isl-{config['isl']}_osl-{config['osl']}_maxcon-{config['max_concurrency']}_n-{config['num_prompts']}.json"
        )
        
        # Create dummy benchmark data
        benchmark_data = {
            "benchmarks": {
                "ttft": 0.15 + (config['isl'] / 1000),  # Dummy TTFT based on input length
                "tpot": 0.05 + (config['osl'] / 5000),  # Dummy TPOT based on output length
                "itl": 0.08,
                "e2el": 2.5 + (config['num_prompts'] / 50),
            },
            "metadata": {
                "model": model_spec.model_id,
                "timestamp": run_timestamp,
                "config": config,
                "device": model_spec.cli_args.get("device", "n150"),
                "workflow": "benchmarks"
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=4)
        
        benchmark_files.append(filename)
        logger.info(f"Created benchmark file: {filename}")
    
    return benchmark_files


def create_dummy_eval_output(model_spec, output_path):
    """Create dummy evaluation output files matching run_evals.py structure."""
    logger.info("Creating dummy evaluation output...")
    
    # Create eval output directory structure matching run_evals.py
    eval_output_dir = output_path / "evals_output"
    eval_model_dir = eval_output_dir / f"eval_{model_spec.model_id}"
    
    # For CNN models, create COCO evaluation structure
    if model_spec.model_type.name == "CNN":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_run_dir = eval_output_dir / f"eval_{model_spec.model_name}_n150_{timestamp}_n500"
        eval_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations directory
        vis_dir = eval_run_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy COCO evaluation results
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
        
        # Save COCO metrics
        coco_results_file = eval_run_dir / "coco_detection_val2017_metrics.json"
        with open(coco_results_file, 'w') as f:
            json.dump(coco_metrics, f, indent=2)
        
        # Create dummy visualization files
        for i in range(5):
            dummy_vis_file = vis_dir / f"image_{i*100}_detections.png"
            dummy_vis_file.write_text("dummy_image_data")
        
        logger.info(f"Created COCO evaluation results: {coco_results_file}")
        return [coco_results_file]
    
    else:
        # For LLM models, create standard lm-eval structure
        eval_model_dir.mkdir(parents=True, exist_ok=True)
        hf_repo_dir = eval_model_dir / model_spec.hf_model_repo.replace('/', '__')
        hf_repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy evaluation results
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
        
        results_file = hf_repo_dir / f"results_{int(time())}.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        logger.info(f"Created evaluation results: {results_file}")
        return [results_file]


def run_benchmarks_workflow(model_spec, workflow_logs_dir):
    """Run benchmarks workflow and create output files."""
    logger.info("Running benchmarks workflow...")
    benchmark_files = create_dummy_benchmark_output(model_spec, workflow_logs_dir)
    logger.info(f"✅ Benchmarks workflow completed. Created {len(benchmark_files)} files.")
    return 0


def run_evals_workflow(model_spec, workflow_logs_dir):
    """Run evaluations workflow and create output files."""
    logger.info("Running evaluations workflow...")
    eval_files = create_dummy_eval_output(model_spec, workflow_logs_dir)
    logger.info(f"✅ Evaluations workflow completed. Created {len(eval_files)} files.")
    return 0


def run_release_workflow(model_spec, workflow_logs_dir):
    """Run release workflow (benchmarks + evals + report generation)."""
    logger.info("Running release workflow...")
    
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
    
    # Create dummy release report
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
    """Main function to run the YOLOv4 CI script."""
    logger.info("Starting YOLOv4 CI script...")
    
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