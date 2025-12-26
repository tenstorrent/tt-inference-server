#!/usr/bin/env python3
"""
Script to test maximum gpu_memory_utilization for different models.
Tests each model with gpu_memory_utilization values from 0.1 to 0.9,
and records the highest value that works before OOM errors occur.
"""

import subprocess
import sys
from vllm import LLM, SamplingParams


MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",
]

GPU_MEMORY_UTILIZATION_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_tt_smi():
    """Run tt-smi -r command to reset GPU state."""
    try:
        result = subprocess.run(
            ["tt-smi", "-r"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"tt-smi -r output: {result.stdout}")
        if result.stderr:
            print(f"tt-smi -r stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Warning: tt-smi -r timed out")
    except Exception as e:
        print(f"Warning: Failed to run tt-smi -r: {e}")


def test_model_with_utilization(model_name, gpu_memory_utilization):
    """
    Test a model with a specific gpu_memory_utilization value.
    Returns (success: bool, error_message: str or None)
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"{'='*60}")

    try:
        prompt = "Hello, it's me"
        engine_args = {
            "model": model_name,
            "max_model_len": 128,
            "max_num_seqs": 1,
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": 128,
            "seed": 9472,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": gpu_memory_utilization,
            "additional_config": {
                "enable_const_eval": False,
                "min_context_len": 32,
            },
        }

        print(f"Loading model...")
        llm_engine = LLM(**engine_args)

        print(f"Starting model warmup")
        warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        result = llm_engine.generate(
            prompt, warmup_sampling_params, "warmup_task_id"
        )
        print(f"Model warmup completed successfully")

        # Clean up
        del llm_engine

        return True, None

    except Exception as e:
        error_msg = str(e)
        print(f"Error occurred: {error_msg}")

        # Check if it's an OOM error
        is_oom = (
            "Out of Memory" in error_msg or
            "OOM" in error_msg.upper() or
            "not enough space" in error_msg.lower() or
            "TT_FATAL" in error_msg
        )

        return False, error_msg if is_oom else error_msg


def find_max_utilization(model_name):
    """
    Find the maximum gpu_memory_utilization that works for a model.
    Returns the highest working value and any error message.
    """
    max_working_utilization = None
    last_error = None

    for utilization in GPU_MEMORY_UTILIZATION_VALUES:
        success, error_msg = test_model_with_utilization(model_name, utilization)

        # Run tt-smi -r after each test
        print(f"\nRunning tt-smi -r to reset GPU state...")
        run_tt_smi()

        if success:
            max_working_utilization = utilization
            print(f"✓ Success with gpu_memory_utilization={utilization}")
        else:
            # Check if it's an OOM error
            if error_msg and (
                "Out of Memory" in error_msg or
                "OOM" in error_msg.upper() or
                "not enough space" in error_msg.lower() or
                "TT_FATAL" in error_msg
            ):
                print(f"✗ OOM error with gpu_memory_utilization={utilization}")
                last_error = error_msg
                break
            else:
                print(f"✗ Non-OOM error with gpu_memory_utilization={utilization}: {error_msg}")
                last_error = error_msg
                # Continue testing even with non-OOM errors to see if higher values work

    return max_working_utilization, last_error


def generate_report(results):
    """Generate a markdown report from the test results."""
    report = """# GPU Memory Utilization Test Report

This report shows the maximum `gpu_memory_utilization` value that each model can work with before encountering Out of Memory (OOM) errors.

## Test Configuration

- **Test Range**: 0.1 to 0.9 (increments of 0.1)
- **Error Detection**: Tests stop when encountering OOM errors similar to:
  - `TT_FATAL: Out of Memory: Not enough space to allocate`
- **GPU Reset**: `tt-smi -r` was executed after each test run

## Results

| Model | Maximum Working GPU Memory Utilization | Status | Notes |
|-------|----------------------------------------|--------|-------|
"""

    for model_name, max_util, error in results:
        if max_util is not None:
            status = "✓ Working"
            notes = f"Successfully tested up to {max_util}"
        else:
            status = "✗ Failed"
            notes = f"Error: {error[:100] if error else 'Unknown error'}"

        report += f"| {model_name} | {max_util if max_util is not None else 'N/A'} | {status} | {notes} |\n"

    report += "\n## Test Details\n\n"

    for model_name, max_util, error in results:
        report += f"### {model_name}\n\n"
        if max_util is not None:
            report += f"- **Maximum Working Utilization**: {max_util}\n"
            report += f"- **Status**: Successfully tested\n"
        else:
            report += f"- **Status**: Failed\n"
            if error:
                report += f"- **Error**: {error}\n"
        report += "\n"

    report += "---\n\n"
    report += "*Report generated automatically by test_gpu_memory_utilization.py*\n"

    return report


def main():
    """Main function to run all tests and generate report."""
    print("="*60)
    print("GPU Memory Utilization Test Suite")
    print("="*60)

    results = []

    for model_name in MODELS:
        print(f"\n\n{'#'*60}")
        print(f"Testing Model: {model_name}")
        print(f"{'#'*60}\n")

        max_util, error = find_max_utilization(model_name)
        results.append((model_name, max_util, error))

        print(f"\n{'='*60}")
        print(f"Result for {model_name}:")
        print(f"  Maximum Working Utilization: {max_util}")
        if error:
            print(f"  Error: {error[:200]}")
        print(f"{'='*60}\n")

    # Generate and save report
    report = generate_report(results)
    report_path = "gpu_memory_utilization_report.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}\n")

    # Print summary
    print("\n## Summary\n")
    for model_name, max_util, error in results:
        print(f"{model_name}: {max_util if max_util is not None else 'Failed'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

