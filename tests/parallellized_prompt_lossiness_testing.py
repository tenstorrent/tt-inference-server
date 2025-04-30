# parallel_token_analysis.py
# !/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Parallel Token Roundtrip Analysis Script

This script analyzes how tokenization and detokenization affects prompts
to determine if encoding/decoding is lossy, using parallel processing for speed.
"""

import argparse
import os
import json
import logging
import time
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Tuple
import pandas as pd
from transformers import AutoTokenizer
import tqdm

from utils.prompt_configs import PromptConfig
from utils.prompt_generation import generate_random_prompts, tokenize_encode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_argument_parser():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description="Parallel analysis of token encoding/decoding lossiness")

    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model identifier (for tokenizer)")

    parser.add_argument("--fallback-model", type=str,
                        default="gpt2",
                        help="Fallback model to use if primary model fails")

    parser.add_argument("--input-len", type=int, default=16392,
                        help="Target input sequence length")

    parser.add_argument("--max-len", type=int, default=120000,
                        help="Maximum allowed prompt length")

    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of prompts to generate")

    parser.add_argument("--distribution", type=str,
                        choices=["fixed", "uniform", "normal"],
                        default="uniform",
                        help="Token length distribution")

    parser.add_argument("--processes", type=int, default=None,
                        help="Number of parallel processes to use (default: auto)")

    parser.add_argument("--output", type=str, default=None,
                        help="Path to save analysis results (JSON)")

    return parser


def get_tokenizer(model_name, fallback_model="gpt2"):
    """Get tokenizer with fallback if primary model fails"""
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer, model_name
    except Exception as e:
        logger.warning(f"Failed to load primary tokenizer: {str(e)}")
        try:
            logger.info(f"Trying fallback tokenizer: {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            logger.info(f"Using fallback tokenizer with vocab size: {tokenizer.vocab_size}")
            return tokenizer, fallback_model
        except Exception as e2:
            logger.error(f"Failed to load fallback tokenizer: {str(e2)}")
            raise RuntimeError("Could not load any tokenizer")


def analyze_prompt_roundtrip(prompt_data, tokenizer_model):
    """
    Analyze token lossiness for a single prompt (for parallel processing)

    Args:
        prompt_data: Tuple of (prompt_id, prompt_text)
        tokenizer_model: Model name to load tokenizer

    Returns:
        Dictionary with analysis results for this prompt
    """
    prompt_id, prompt = prompt_data

    # Load tokenizer within the process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # First encoding
    encoded = tokenize_encode(prompt, tokenizer, max_length=None, tokenizer_model=tokenizer_model)
    original_length = len(encoded)

    # Decode back to text
    decoded = tokenizer.decode(encoded)

    # Re-encode to see if we lose information
    re_encoded = tokenize_encode(decoded, tokenizer, max_length=None, tokenizer_model=tokenizer_model)
    re_encoded_length = len(re_encoded)

    # Compare token sequences
    tokens_match = encoded == re_encoded
    token_diff = set(encoded) - set(re_encoded)
    missing_tokens = len(token_diff)

    # Check if texts match exactly
    text_match = prompt == decoded

    # Character-level analysis
    char_diff = []
    for j, (a, b) in enumerate(zip(prompt, decoded + ' ' * (len(prompt) - len(decoded)))):
        if a != b:
            char_diff.append((j, a, b))

    # Calculate text similarity percentage
    if len(prompt) > 0:
        matching_chars = sum(1 for a, b in zip(prompt, decoded) if a == b)
        text_similarity = (matching_chars / len(prompt)) * 100
    else:
        text_similarity = 100.0

    # Multi-round analysis
    rounds = 5
    current_text = prompt
    token_counts = []
    text_similarities = []

    # Original token count
    token_counts.append(original_length)
    text_similarities.append(100.0)  # First text is identical

    # Perform multiple roundtrips
    for round_num in range(rounds):
        # Encode current text
        encoded = tokenize_encode(current_text, tokenizer, max_length=None, tokenizer_model=tokenizer_model)

        # Decode back to text
        decoded = tokenizer.decode(encoded)

        # Calculate text similarity
        if len(current_text) > 0:
            matching_chars = sum(1 for a, b in zip(current_text, decoded) if a == b)
            similarity = (matching_chars / len(current_text)) * 100
        else:
            similarity = 100.0

        # Save for next round
        current_text = decoded
        token_counts.append(len(encoded))
        text_similarities.append(similarity)

    # Calculate stability metrics
    token_count_stable = len(set(token_counts[1:])) == 1  # All re-encoded counts equal?
    text_stabilized = False
    stabilization_round = -1

    for r in range(1, rounds):
        if text_similarities[r] == 100.0:
            text_stabilized = True
            stabilization_round = r
            break

    # Store results
    result = {
        "prompt_id": prompt_id,
        "original_tokens": original_length,
        "re_encoded_tokens": re_encoded_length,
        "tokens_match": tokens_match,
        "missing_tokens": missing_tokens,
        "text_match": text_match,
        "char_changes": len(char_diff),
        "text_similarity": text_similarity,
        "token_counts": token_counts,
        "text_similarities": text_similarities,
        "token_count_stable": token_count_stable,
        "text_stabilized": text_stabilized,
        "stabilization_round": stabilization_round,
        "original_prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "decoded_prompt_preview": decoded[:50] + "..." if len(decoded) > 50 else decoded,
        "final_prompt_preview": current_text[:50] + "..." if len(current_text) > 50 else current_text,
    }

    return result


def main():
    """Main function to analyze token lossiness in parallel"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    start_time = time.time()

    # Load tokenizer with fallback
    try:
        tokenizer, actual_model = get_tokenizer(args.model, args.fallback_model)
    except Exception as e:
        logger.error(f"Could not load any tokenizer: {e}")
        return

    logger.info("Generating random prompts...")
    logger.info(f"  Using tokenizer: {actual_model}")
    logger.info(f"  Distribution: {args.distribution}")
    logger.info(f"  Target length: {args.input_len} tokens")
    logger.info(f"  Number of prompts: {args.num_prompts}")

    # Generate random prompts
    prompts = generate_random_prompts(
        num_prompts=args.num_prompts,
        max_length=args.max_len,
        input_seq_len=args.input_len,
        distribution=args.distribution,
        tokenizer_model=actual_model,
    )

    # Prepare data for parallel processing
    prompt_data = [(i, prompt) for i, prompt in enumerate(prompts)]

    # Set number of processes
    num_processes = args.processes or min(multiprocessing.cpu_count(), len(prompts))
    logger.info(f"Using {num_processes} parallel processes for analysis")

    # Create partial function with the tokenizer model
    analyze_func = partial(analyze_prompt_roundtrip, tokenizer_model=actual_model)

    # Run analysis in parallel
    logger.info("Running parallel analysis...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(analyze_func, prompt_data),
            total=len(prompt_data),
            desc="Analyzing prompts"
        ))

    # Calculate aggregate statistics
    single_stats = {
        "total_prompts": len(results),
        "avg_token_length": sum(r["original_tokens"] for r in results) / len(results),
        "perfect_token_match_count": sum(1 for r in results if r["tokens_match"]),
        "perfect_token_match_pct": sum(1 for r in results if r["tokens_match"]) / len(results) * 100,
        "perfect_text_match_count": sum(1 for r in results if r["text_match"]),
        "perfect_text_match_pct": sum(1 for r in results if r["text_match"]) / len(results) * 100,
        "avg_text_similarity": sum(r["text_similarity"] for r in results) / len(results),
        "avg_char_changes": sum(r["char_changes"] for r in results) / len(results),
        "max_char_changes": max(r["char_changes"] for r in results),
    }

    # Multi-roundtrip stats
    multi_stats = {
        "total_prompts": len(results),
        "token_count_stable_pct": sum(1 for r in results if r["token_count_stable"]) / len(results) * 100,
        "text_stabilized_pct": sum(1 for r in results if r["text_stabilized"]) / len(results) * 100,
    }

    # Calculate average stabilization round only for those that stabilized
    stabilized_rounds = [r["stabilization_round"] for r in results if r["stabilization_round"] >= 0]
    if stabilized_rounds:
        multi_stats["avg_stabilization_round"] = sum(stabilized_rounds) / len(stabilized_rounds)
    else:
        multi_stats["avg_stabilization_round"] = -1

    # Display sample prompt details (just a few to avoid cluttering the output)
    max_samples = min(5, len(results))
    logger.info(f"\nSAMPLE RESULTS ({max_samples} of {len(results)} prompts):")
    for result in results[:max_samples]:
        prompt_id = result["prompt_id"]
        logger.info(f"\nPrompt {prompt_id}:")
        logger.info(f"  Original tokens: {result['original_tokens']}")
        logger.info(f"  Re-encoded tokens: {result['re_encoded_tokens']}")

        if result["tokens_match"]:
            logger.info(f"  LOSSLESS: Token sequences match perfectly")
        else:
            logger.info(f"  LOSSY: Missing {result['missing_tokens']} unique tokens")

        if result["text_match"]:
            logger.info(f"  PERFECT TEXT: Original and decoded texts match exactly")
        else:
            logger.info(
                f"  TEXT CHANGED: {result['char_changes']} character differences, {result['text_similarity']:.2f}% similar")

        logger.info(f"  Original: {result['original_prompt_preview']}")
        logger.info(f"  Decoded: {result['decoded_prompt_preview']}")

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("ENCODE/DECODE ANALYSIS RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"Total prompts analyzed: {single_stats['total_prompts']} in {time.time() - start_time:.2f} seconds")
        logger.info(f"Average token length: {single_stats['avg_token_length']:.2f} tokens")
        logger.info("\nTOKEN LOSSINESS:")
        logger.info(
            f"Perfect token match: {single_stats['perfect_token_match_count']} prompts ({single_stats['perfect_token_match_pct']:.1f}%)")
        logger.info(
            f"Perfect text match: {single_stats['perfect_text_match_count']} prompts ({single_stats['perfect_text_match_pct']:.1f}%)")
        logger.info(f"Average text similarity: {single_stats['avg_text_similarity']:.2f}%")
        logger.info(f"Average character changes: {single_stats['avg_char_changes']:.2f}")
        logger.info(f"Maximum character changes: {single_stats['max_char_changes']}")

        # Multi-roundtrip stats
        logger.info("\nMULTIPLE ROUNDTRIP STABILITY:")
        logger.info(f"Token count stabilized: {multi_stats['token_count_stable_pct']:.1f}% of prompts")
        logger.info(f"Text stabilized: {multi_stats['text_stabilized_pct']:.1f}% of prompts")
        if multi_stats['text_stabilized_pct'] > 0 and multi_stats['avg_stabilization_round'] >= 0:
            logger.info(f"Average stabilization round: {multi_stats['avg_stabilization_round']:.2f}")
        logger.info("=" * 60)

    # Save results if requested
    if args.output:
        combined_results = {
            "prompt_results": results,
            "single_roundtrip_stats": single_stats,
            "multiple_roundtrips_stats": multi_stats,
            "metadata": {
                "model": actual_model,
                "distribution": args.distribution,
                "input_length": args.input_len,
                "max_length": args.max_len,
                "processes_used": num_processes,
                "execution_time_seconds": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        with open(args.output, 'w') as f:
            json.dump(combined_results, f, indent=2)
        logger.info(f"\nAnalysis results saved to {args.output}")

    logger.info(f"Used {num_processes} parallel processes for analysis")


if __name__ == "__main__":
    main()