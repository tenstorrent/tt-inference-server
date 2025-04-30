#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Token Encoding/Decoding Grid Analysis

This script runs token roundtrip analysis across multiple combinations of
input lengths and maximum lengths to analyze how tokenization lossiness
varies with different sequence length parameters.

It saves comprehensive results for later analysis and creates a summary report.
"""

import os
import json
import time
import logging
import argparse
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import subprocess
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run token analysis across a spectrum of lengths")

    parser.add_argument("--model", type=str,
                        default="gpt2",
                        help="Model identifier for tokenizer")

    parser.add_argument("--fallback-model", type=str,
                        default=None,
                        help="Fallback model if primary fails to load")

    parser.add_argument("--num-prompts", type=int,
                        default=10000,
                        help="Number of prompts to analyze per configuration")

    parser.add_argument("--input-lens", type=str,
                        default="32,64,128,256,512,1024,2048,4096,8192",
                        help="Comma-separated list of input lengths to test")

    parser.add_argument("--max-lens", type=str,
                        default="32,64,128,256,512,1024,2048,4096,8192",
                        help="Comma-separated list of maximum lengths to test")

    parser.add_argument("--distribution", type=str,
                        default="fixed",
                        choices=["fixed", "uniform", "normal"],
                        help="Distribution of token lengths")

    parser.add_argument("--processes", type=int,
                        default=None,
                        help="Number of parallel processes (default: auto)")

    parser.add_argument("--output-dir", type=str,
                        default="token_analysis_results",
                        help="Directory to save results")

    parser.add_argument("--parallel-script-path", type=str,
                        default="tests/parallellized_prompt_lossiness_testing.py",
                        help="Path to parallel token analysis script")

    return parser.parse_args()


def run_analysis_for_config(config, args, output_dir):
    """Run token analysis for a specific input/max length configuration"""
    input_len, max_len = config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{output_dir}/analysis_{input_len}_{max_len}_{timestamp}.json"

    # If input_len > max_len, use max_len as the input_len
    effective_input_len = min(input_len, max_len)

    # Build command
    cmd = [
        "python", args.parallel_script_path,
        "--model", args.model,
        "--num-prompts", str(args.num_prompts),
        "--input-len", str(effective_input_len),
        "--max-len", str(max_len),
        "--distribution", args.distribution,
        "--output", result_file
    ]

    if args.processes:
        cmd.extend(["--processes", str(args.processes)])

    if args.fallback_model:
        cmd.extend(["--fallback-model", args.fallback_model])

    # Execute command
    try:
        start_time = time.time()
        logger.info(f"Starting analysis for input_len={effective_input_len}, max_len={max_len}")

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse results if file was created
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result_data = json.load(f)

            # Extract key statistics
            single_stats = result_data["single_roundtrip_stats"]
            multi_stats = result_data["multiple_roundtrips_stats"]

            elapsed_time = time.time() - start_time
            logger.info(
                f"Completed analysis for input_len={effective_input_len}, max_len={max_len} in {elapsed_time:.2f}s")

            return {
                "input_len": effective_input_len,
                "max_len": max_len,
                "perfect_token_match_pct": single_stats["perfect_token_match_pct"],
                "perfect_text_match_pct": single_stats["perfect_text_match_pct"],
                "avg_text_similarity": single_stats["avg_text_similarity"],
                "avg_char_changes": single_stats["avg_char_changes"],
                "token_count_stable_pct": multi_stats["token_count_stable_pct"],
                "text_stabilized_pct": multi_stats["text_stabilized_pct"],
                "result_file": result_file,
                "elapsed_time": elapsed_time
            }
        else:
            logger.error(f"Result file not created for input_len={effective_input_len}, max_len={max_len}")
            return {
                "input_len": effective_input_len,
                "max_len": max_len,
                "error": "Result file not created",
                "stdout": process.stdout,
                "stderr": process.stderr,
                "elapsed_time": time.time() - start_time
            }

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running analysis for input_len={effective_input_len}, max_len={max_len}: {e}")
        return {
            "input_len": effective_input_len,
            "max_len": max_len,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
            "elapsed_time": time.time() - start_time
        }


def generate_summary_report(results, output_dir, args):
    """Generate summary report from all results"""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Save raw results to CSV
    csv_path = os.path.join(output_dir, "summary_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary results to {csv_path}")

    # Generate heatmaps for key metrics
    metrics = [
        "perfect_token_match_pct",
        "perfect_text_match_pct",
        "avg_text_similarity",
        "avg_char_changes",
        "token_count_stable_pct",
        "text_stabilized_pct"
    ]

    for metric in metrics:
        if metric in df.columns:
            try:
                # Create pivot table
                pivot = df.pivot(index="input_len", columns="max_len", values=metric)

                # Plot heatmap
                plt.figure(figsize=(12, 10))
                plt.title(f"{metric} by Input Length and Max Length")

                # For avg_char_changes, lower is better
                cmap = "RdYlGn_r" if metric == "avg_char_changes" else "RdYlGn"

                # Create heatmap
                heatmap = plt.pcolormesh(pivot.columns, pivot.index, pivot, cmap=cmap)
                plt.colorbar(heatmap, label=metric)

                plt.xlabel("Max Length")
                plt.ylabel("Input Length")
                plt.xscale('log', base=2)
                plt.yscale('log', base=2)

                # Add grid and text annotations
                for i, input_len in enumerate(pivot.index):
                    for j, max_len in enumerate(pivot.columns):
                        if not pd.isna(pivot.iloc[i, j]):
                            plt.text(max_len, input_len, f"{pivot.iloc[i, j]:.1f}",
                                     ha="center", va="center",
                                     color="black" if 30 < pivot.iloc[i, j] < 70 else "white")

                # Save plot
                plot_path = os.path.join(output_dir, f"heatmap_{metric}.png")
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Generated heatmap for {metric} at {plot_path}")

            except Exception as e:
                logger.error(f"Error generating heatmap for {metric}: {e}")

    # Generate line plots for selected metrics
    plt.figure(figsize=(15, 10))

    for metric in metrics[:4]:  # Just use the first 4 metrics for the line plot
        if metric in df.columns:
            # Group by input_len and calculate mean for each metric
            grouped = df.groupby("input_len")[metric].mean()

            # Plot line
            plt.plot(grouped.index, grouped.values, marker='o', label=metric)

    plt.title("Metrics by Input Length (Averaged across Max Lengths)")
    plt.xlabel("Input Length")
    plt.ylabel("Percentage / Value")
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()

    line_plot_path = os.path.join(output_dir, "metrics_by_input_length.png")
    plt.savefig(line_plot_path)
    plt.close()

    # Generate markdown summary
    md_path = os.path.join(output_dir, "summary_report.md")
    with open(md_path, 'w') as f:
        f.write("# Token Analysis Grid Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** {args.model}\n\n")
        f.write(f"**Prompts per Configuration:** {args.num_prompts}\n\n")
        f.write(f"**Distribution:** {args.distribution}\n\n")

        f.write("## Summary\n\n")
        for metric in metrics:
            if metric in df.columns:
                f.write(f"### {metric}\n\n")
                f.write(f"- **Average:** {df[metric].mean():.2f}%\n")
                f.write(f"- **Min:** {df[metric].min():.2f}%\n")
                f.write(f"- **Max:** {df[metric].max():.2f}%\n\n")

                # Find best and worst configurations
                if metric == "avg_char_changes":
                    # For character changes, lower is better
                    best_idx = df[metric].idxmin()
                    worst_idx = df[metric].idxmax()
                else:
                    # For other metrics, higher is better
                    best_idx = df[metric].idxmax()
                    worst_idx = df[metric].idxmin()

                best_config = df.iloc[best_idx]
                worst_config = df.iloc[worst_idx]

                f.write(
                    f"- **Best configuration:** input_len={best_config['input_len']}, max_len={best_config['max_len']} → {best_config[metric]:.2f}%\n")
                f.write(
                    f"- **Worst configuration:** input_len={worst_config['input_len']}, max_len={worst_config['max_len']} → {worst_config[metric]:.2f}%\n\n")

        f.write("## Visualizations\n\n")
        for metric in metrics:
            f.write(f"### {metric} Heatmap\n\n")
            f.write(f"![{metric} Heatmap](heatmap_{metric}.png)\n\n")

        f.write("### Metrics by Input Length\n\n")
        f.write("![Metrics by Input Length](metrics_by_input_length.png)\n\n")

        f.write("## Raw Results\n\n")
        f.write("See [summary_results.csv](summary_results.csv) for the complete dataset.\n")

    logger.info(f"Generated summary report at {md_path}")
    return md_path


def main():
    args = parse_args()

    # Parse input and max lengths
    input_lens = [int(x) for x in args.input_lens.split(",")]
    max_lens = [int(x) for x in args.max_lens.split(",")]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{args.model.replace('/', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting grid analysis with model={args.model}")
    logger.info(f"Testing input_lens: {input_lens}")
    logger.info(f"Testing max_lens: {max_lens}")
    logger.info(f"Number of prompts per configuration: {args.num_prompts}")
    logger.info(f"Results will be saved to: {output_dir}")

    # Generate all configurations to test
    configs = []
    for input_len in input_lens:
        for max_len in max_lens:
            # Only test valid configurations where max_len >= input_len/2
            # This prevents extremely long runs for infeasible combinations
            if max_len >= input_len / 2:
                configs.append((input_len, max_len))

    # Sort configs by total tokens (input_len + max_len) to start with quicker tests
    configs.sort(key=lambda x: x[0] + x[1])

    # Determine number of worker processes for the outer loop
    max_outer_workers = min(8, len(configs))  # Limit to 8 concurrent processes max

    # Run analysis for each configuration in parallel
    logger.info(f"Running analysis for {len(configs)} configurations using {max_outer_workers} parallel processes")

    results = []

    # Create a partial function with fixed arguments
    run_fn = partial(run_analysis_for_config, args=args, output_dir=output_dir)

    # Run the tests in parallel
    with ProcessPoolExecutor(max_workers=max_outer_workers) as executor:
        futures = {executor.submit(run_fn, config): config for config in configs}

        for future in tqdm(as_completed(futures), total=len(configs), desc="Running grid analysis"):
            config = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed configuration: input_len={config[0]}, max_len={config[1]}")
            except Exception as e:
                logger.error(f"Error processing input_len={config[0]}, max_len={config[1]}: {e}")

    # Generate summary report
    logger.info("Generating summary report...")
    report_path = generate_summary_report(results, output_dir, args)

    logger.info(f"Analysis complete. Summary report available at: {report_path}")
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()