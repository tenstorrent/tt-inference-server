#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Parallel Token Analysis Script

This script performs two types of analysis:
1. Initial Encoding Growth Analysis - measuring how much token count grows from raw input
   to the initially encoded prompt
2. Prompt Lossiness Analysis - analyzing how tokenization and detokenization affects prompts
   to determine if encoding/decoding is lossy

Both analyses use parallel processing for speed.
"""

import argparse
import os
import json
import logging
import random
import time
import statistics
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
import tqdm
from collections import defaultdict

from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def tokenize_encode_client(prompt, tokenizer, max_length):
    return tokenizer.encode(
        prompt, add_special_tokens=False, truncation=True, max_length=max_length
    )

def tokenize_decode_client(encoded_prompt, tokenizer):
    return tokenizer.decode(encoded_prompt)

def tokenize_encode_server(prompt, tokenizer, max_length, client: PromptClient):
    tokens = client.entokenize(prompt)["tokens"]
    return tokens

def tokenize_decode_server(tokens, tokenizer, client: PromptClient):
    prompt = client.detokenize(tokens, tokenizer)["prompt"]
    return prompt

def setup_argument_parser():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description="Parallel analysis of token encoding/decoding")
    parser.add_argument(
        "--server-tokenizer",
        action="store_true",
        help="Use server-side tokenizer",
    )

    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model identifier (for tokenizer)")

    parser.add_argument("--fallback-model", type=str,
                        default="gpt2",
                        help="Fallback model to use if primary model fails")

    parser.add_argument("--input-len", type=int, default=1024,
                        help="Target input sequence length")

    parser.add_argument("--max-len", type=int, default=128000,
                        help="Maximum allowed prompt length")

    parser.add_argument("--max-length-truncation", type=int, default=None,
                        help="Maximum length for truncation during tokenization")

    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts to generate")

    parser.add_argument("--distribution", type=str,
                        choices=["fixed", "uniform", "normal"],
                        default="fixed",
                        help="Token length distribution")

    parser.add_argument("--processes", type=int, default=None,
                        help="Number of parallel processes to use (default: auto)")

    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save analysis results")

    parser.add_argument("--template", type=str, default=None,
                        help="Optional template to apply to prompts")

    return parser

def get_tokenizer(model_name, fallback_model="None"):
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


def analyze_initial_encoding_growth(prompt_data, tokenizer_model, input_length, max_length_truncation=None,
                                    template=None, server_tokenizer=False, client=None):
    """
    Analyze the growth in token count during initial encoding of the prompt.

    Args:
        prompt_data: Tuple of (prompt_id, prompt_text)
        tokenizer_model: Model name to load tokenizer
        input_length: Target input length
        max_length_truncation: Maximum length for truncation during tokenization
        template: Optional template to apply
        server_tokenizer: Whether to use server-side tokenization
        client: PromptClient instance for server tokenization

    Returns:
        Dictionary with analysis results for this prompt's initial encoding
    """
    prompt_id, prompt = prompt_data

    # Load tokenizer if using client-side tokenization
    if not server_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    else:
        tokenizer = tokenizer_model  # Just pass the model name for server tokenization

    # Raw encoding (without template)
    if server_tokenizer:
        raw_encoded = tokenize_encode_server(prompt, tokenizer, max_length_truncation, client)
    else:
        raw_encoded = tokenize_encode_client(prompt, tokenizer, max_length=max_length_truncation)

    raw_token_count = len(raw_encoded)

    # Apply template if provided
    if template:
        templated_prompt = template.replace("{prompt}", prompt)
    else:
        templated_prompt = prompt

    # Encode with template
    if server_tokenizer:
        templated_encoded = tokenize_encode_server(templated_prompt, tokenizer, max_length_truncation, client)
    else:
        templated_encoded = tokenize_encode_client(templated_prompt, tokenizer, max_length=max_length_truncation)

    templated_token_count = len(templated_encoded)

    # Calculate growth
    token_growth = templated_token_count - raw_token_count
    pct_growth = (token_growth / raw_token_count) * 100 if raw_token_count > 0 else 0

    # Calculate delta from target input length
    delta_from_target = templated_token_count - input_length

    return {
        "prompt_id": prompt_id,
        "raw_token_count": raw_token_count,
        "templated_token_count": templated_token_count,
        "token_growth": token_growth,
        "pct_growth": pct_growth,
        "delta_from_target": delta_from_target,
        "template_applied": template is not None,
    }


def analyze_prompt_lossiness(prompt_data, tokenizer_model, input_length, max_length_truncation=None,
                             server_tokenizer=False, client=None):
    """
    Analyze token lossiness for a single prompt (for parallel processing)
    as well as token growth and shrinkage during roundtrip encoding/decoding.

    Args:
        prompt_data: Tuple of (prompt_id, prompt_text)
        tokenizer_model: Model name to load tokenizer
        input_length: Target input length
        max_length_truncation: Maximum length for truncation during tokenization
        server_tokenizer: Whether to use server-side tokenization
        client: PromptClient instance for server tokenization

    Returns:
        Dictionary with analysis results for this prompt
    """
    prompt_id, prompt = prompt_data

    # Load tokenizer if using client-side tokenization
    if not server_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = tokenizer_model  # Just pass the model name for server tokenization
        # Estimate vocab size - could be retrieved from server if available
        vocab_size = 32000  # Default estimate for LLM models

    # First encoding
    if server_tokenizer:
        encoded = tokenize_encode_server(prompt, tokenizer, None, client)
    else:
        encoded = tokenize_encode_client(prompt, tokenizer, max_length=None)

    original_length = len(encoded)

    # If the encoded prompt is shorter than the target length, append random tokens
    if original_length < input_length:
        # Append random tokens until we reach the target length
        while len(encoded) < input_length:
            # Generate a random token ID between 0 and vocab_size-1
            random_token = np.random.randint(0, vocab_size)
            encoded.append(random_token)

    # Decode back to text
    if server_tokenizer:
        decoded = tokenize_decode_server(encoded, tokenizer, client)
    else:
        decoded = tokenizer.decode(encoded)

    # Re-encode to see if we lose information
    if server_tokenizer:
        re_encoded = tokenize_encode_server(decoded, tokenizer, None, client)
    else:
        re_encoded = tokenize_encode_client(decoded, tokenizer, max_length=None)

    re_encoded_length = len(re_encoded)

    # Compare token sequences
    tokens_match = encoded == re_encoded
    token_diff = set(encoded) - set(re_encoded)
    missing_tokens = len(token_diff)

    # Determine if tokens grew or shrank
    token_count_delta = re_encoded_length - original_length
    tokens_grew = token_count_delta > 0
    tokens_shrank = token_count_delta < 0
    tokens_unchanged = token_count_delta == 0

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
        if server_tokenizer:
            encoded = tokenize_encode_server(current_text, tokenizer, max_length_truncation, client)
        else:
            encoded = tokenize_encode_client(current_text, tokenizer, max_length=max_length_truncation)

        # If the encoded prompt is shorter than the target length, append random tokens
        if len(encoded) < input_length:
            # Append random tokens until we reach the target length
            while len(encoded) < input_length:
                # Generate a random token ID between 0 and vocab_size-1
                random_token = np.random.randint(0, vocab_size)
                encoded.append(random_token)

        # Decode back to text
        if server_tokenizer:
            decoded = tokenize_decode_server(encoded, tokenizer, client)
        else:
            decoded = tokenizer.decode(encoded, add_special_tokens=False)

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

    # Calculate lossiness metric (higher = more lossy)
    lossiness = 0
    if not tokens_match:
        lossiness += missing_tokens * 0.5
    if not text_match:
        lossiness += len(char_diff) * 0.5

    # Calculate normalized lossiness (0-1 scale)
    normalized_lossiness = min(1.0, lossiness / 100) if lossiness > 0 else 0

    # Store results
    return {
        "prompt_id": prompt_id,
        "input_length": input_length,
        "original_tokens": original_length,
        "re_encoded_tokens": re_encoded_length,
        "token_count_delta": token_count_delta,
        "tokens_grew": tokens_grew,
        "tokens_shrank": tokens_shrank,
        "tokens_unchanged": tokens_unchanged,
        "tokens_match": tokens_match,
        "missing_tokens": missing_tokens,
        "text_match": text_match,
        "char_changes": len(char_diff),
        "text_similarity": text_similarity,
        "lossiness": normalized_lossiness,
        "token_counts": token_counts,
        "text_similarities": text_similarities,
        "token_count_stable": token_count_stable,
        "text_stabilized": text_stabilized,
        "stabilization_round": stabilization_round
    }

def generate_prompts_chunk(chunk_size, chunk_index, max_length, input_seq_len, distribution, model_name, server_tokenizer,
                           client, tokenizer):
    """Generate a chunk of random prompts for parallel processing"""
    vocab_size = 128000
    max_length = max_length
    input_seq_len = input_seq_len
    torch.manual_seed(random.randint(0, 128000))
    # torch.manual_seed(42)

    if distribution == "fixed":
        prompt_lengths = [input_seq_len] * chunk_size
    elif distribution == "uniform":
        prompt_lengths = torch.randint(1, input_seq_len, (chunk_size,)).tolist()
    elif distribution == "normal":
        prompt_lengths = (
            torch.normal(mean=input_seq_len, std=input_seq_len / 4, size=(chunk_size,))
            .clamp(1, max_length)
            .round()
            .to(torch.int32)
            .tolist()
        )
    else:
        raise ValueError(
            f"Invalid distribution method: '{distribution}'. Must be 'fixed', 'uniform', or 'normal'."
        )

    torch.manual_seed(random.randint(0, 128000))
    # torch.manual_seed(42)

    # Generate random tokens for all prompts
    token_ids_list = [
        torch.randint(0, vocab_size, (length,)).tolist() for length in prompt_lengths
    ]

    if server_tokenizer:
        prompts = [
            tokenize_decode_server(token_ids, model_name, client)
            for token_ids in token_ids_list
        ]
    else:
        prompts = [
            tokenize_decode_client(token_ids, tokenizer)
            for token_ids in token_ids_list
        ]

    # prompts = [12800, 25601]
    return prompts



def create_summary_report(initial_results, lossiness_results, args, output_dir):
    """
    Create a comprehensive summary report of both analyses.

    Args:
        initial_results: Results from initial encoding growth analysis
        lossiness_results: Results from prompt lossiness analysis
        args: Command line arguments
        output_dir: Directory to save the report
    """
    # Create a pandas DataFrame for easier analysis
    df_initial = pd.DataFrame(initial_results)
    df_lossiness = pd.DataFrame(lossiness_results)

    # Group by distribution and input length for aggregation
    config_pairs = []
    for result in lossiness_results:
        input_length = result.get('input_length')
        # You would need to track distribution somewhere in your results
        # For this example, we're using the one from args
        distribution = args.distribution
        config_pairs.append((distribution, input_length))

    df_lossiness['config'] = config_pairs

    # Add config to initial results as well
    config_pairs_initial = []
    for result in initial_results:
        config_pairs_initial.append((args.distribution, args.input_len))
    df_initial['config'] = config_pairs_initial

    # Group and aggregate
    initial_grouped = df_initial.groupby('config').agg({
        'raw_token_count': ['count', 'mean', 'std', 'min', 'median', 'max'],
        'templated_token_count': ['mean', 'std', 'min', 'median', 'max'],
        'token_growth': ['mean', 'std', 'min', 'median', 'max'],
        'delta_from_target': ['mean', 'std', 'min', 'median', 'max'],
    })

    lossiness_grouped = df_lossiness.groupby('config').agg({
        'input_length': ['count', 'mean', 'std', 'min', 'median', 'max'],
        'lossiness': ['mean', 'std', 'min', 'median', 'max'],
        'token_count_delta': ['mean', 'std', 'min', 'median', 'max']
    })

    # Calculate token growth and shrinkage statistics
    tokens_grew_count = sum(1 for r in lossiness_results if r["tokens_grew"])
    tokens_grew_pct = (tokens_grew_count / len(lossiness_results)) * 100 if lossiness_results else 0

    tokens_shrank_count = sum(1 for r in lossiness_results if r["tokens_shrank"])
    tokens_shrank_pct = (tokens_shrank_count / len(lossiness_results)) * 100 if lossiness_results else 0

    tokens_unchanged_count = sum(1 for r in lossiness_results if r["tokens_unchanged"])
    tokens_unchanged_pct = (tokens_unchanged_count / len(lossiness_results)) * 100 if lossiness_results else 0

    # Create markdown report
    markdown_report = f"""# Token Analysis Report for Model: {args.model}

*Total prompts analyzed: {len(lossiness_results)}*
*Target Input Lengths Tested: [{args.input_len}]*
*Distributions Tested: ['{args.distribution}']*
*Max Model Length: {args.max_len}*
*Max Length Truncation: {args.max_length_truncation if args.max_length_truncation is not None else 'None (No truncation)'}*

## 1. Initial Encoding Growth Analysis
*Analysis of token count growth during initial encoding*

### Token Growth from Template Application:
*(Number of tokens added by applying the template)*

|                 |   Count |   Mean Growth |   Std Dev |   Min Growth |   Median Growth |   Max Growth |
|:----------------|--------:|--------------:|----------:|-------------:|----------------:|-------------:|
"""

    # Add rows for token growth
    for config, group in initial_grouped.iterrows():
        count = int(group[('raw_token_count', 'count')])
        mean = group[('token_growth', 'mean')]
        std = group[('token_growth', 'std')]
        min_val = group[('token_growth', 'min')]
        median = group[('token_growth', 'median')]
        max_val = group[('token_growth', 'max')]
        markdown_report += f"| {config} | {count:8d} | {mean:14.0f} | {std:10.0f} | {min_val:13.0f} | {median:16.0f} | {max_val:13.0f} |\n"

    markdown_report += """
### Percentage Growth from Template Application:
*(Percentage increase in tokens from raw to templated)*

|                 |   Mean % |   Std Dev % |   Min % |   Median % |   Max % |
|:----------------|---------:|------------:|--------:|-----------:|--------:|
"""

    # Add rows for percentage growth
    for config, group in initial_grouped.iterrows():
        df_config = df_initial[df_initial['config'] == config]
        mean_pct = df_config['pct_growth'].mean()
        std_pct = df_config['pct_growth'].std()
        min_pct = df_config['pct_growth'].min()
        median_pct = df_config['pct_growth'].median()
        max_pct = df_config['pct_growth'].max()
        markdown_report += f"| {config} | {mean_pct:9.0f} | {std_pct:12.0f} | {min_pct:8.0f} | {median_pct:11.0f} | {max_pct:8.0f} |\n"

    markdown_report += """
### Delta from Target Length:
*(Actual Tokens after Template/Encoding - Target Input Length)*

|                 |   Count |   Mean Δ |   Std Dev Δ |   Min Δ |   Median Δ |   Max Δ |
|:----------------|--------:|---------:|------------:|--------:|-----------:|--------:|
"""

    # Add rows for delta from target
    for config, group in initial_grouped.iterrows():
        count = int(group[('raw_token_count', 'count')])
        mean = group[('delta_from_target', 'mean')]
        std = group[('delta_from_target', 'std')]
        min_val = group[('delta_from_target', 'min')]
        median = group[('delta_from_target', 'median')]
        max_val = group[('delta_from_target', 'max')]
        markdown_report += f"| {config} | {count:8d} | {mean:9.0f} | {std:12.0f} | {min_val:8.0f} | {median:11.0f} | {max_val:8.0f} |\n"

    markdown_report += """
## 2. Prompt Lossiness and Roundtrip Analysis
*(Based on processing the initially encoded prompt)*

### Token Growth/Shrinkage Analysis:
*(How token count changes during encode-decode roundtrip)*

|                                      |   Count |   Percentage |
|:-------------------------------------|--------:|-------------:|
| Tokens Grew During Roundtrip         | {0:8d} | {1:13.1f}% |
| Tokens Shrank During Roundtrip       | {2:8d} | {3:13.1f}% |
| Tokens Unchanged During Roundtrip    | {4:8d} | {5:13.1f}% |
""".format(
        tokens_grew_count, tokens_grew_pct,
        tokens_shrank_count, tokens_shrank_pct,
        tokens_unchanged_count, tokens_unchanged_pct
    )

    markdown_report += """
### Token Count Delta Statistics:
*(Change in token count during encode-decode roundtrip)*

|                 |   Mean Δ |   Std Dev Δ |   Min Δ |   Median Δ |   Max Δ |
|:----------------|---------:|------------:|--------:|-----------:|--------:|
"""

    # Add rows for token count delta
    for config, group in lossiness_grouped.iterrows():
        mean = group[('token_count_delta', 'mean')]
        std = group[('token_count_delta', 'std')]
        min_val = group[('token_count_delta', 'min')]
        median = group[('token_count_delta', 'median')]
        max_val = group[('token_count_delta', 'max')]
        markdown_report += f"| {config} | {mean:9.0f} | {std:12.0f} | {min_val:8.0f} | {median:11.0f} | {max_val:8.0f} |\n"

    markdown_report += """
### Descriptive Statistics for Lossiness Metric:
*(Higher value = more information lost due to tokenization)*

|                 |   Count |   Mean Lossiness |   Std Dev |   Min Lossiness |   Median Lossiness |   Max Lossiness |
|:----------------|--------:|-----------------:|----------:|----------------:|-------------------:|----------------:|
"""

    # Add rows for lossiness metrics
    for config, group in lossiness_grouped.iterrows():
        count = int(group[('input_length', 'count')])
        mean = group[('lossiness', 'mean')]
        std = group[('lossiness', 'std')]
        min_val = group[('lossiness', 'min')]
        median = group[('lossiness', 'median')]
        max_val = group[('lossiness', 'max')]
        markdown_report += f"| {config} | {count:8d} | {mean:17.0f} | {std:10.0f} | {min_val:16.0f} | {median:19.0f} | {max_val:16.0f} |\n"

    # Add stability section
    text_stabilized_count = sum(1 for r in lossiness_results if r["text_stabilized"])
    text_stabilized_pct = (text_stabilized_count / len(lossiness_results)) * 100 if lossiness_results else 0

    token_count_stable_count = sum(1 for r in lossiness_results if r["token_count_stable"])
    token_count_stable_pct = (token_count_stable_count / len(lossiness_results)) * 100 if lossiness_results else 0

    stabilized_rounds = [r["stabilization_round"] for r in lossiness_results if r["stabilization_round"] >= 0]
    avg_stabilization_round = sum(stabilized_rounds) / len(stabilized_rounds) if stabilized_rounds else -1

    markdown_report += """
### Multi-roundtrip Stability Analysis:
*(How token count and text content stabilize over multiple roundtrips)*

|                                      |   Count |   Percentage |
|:-------------------------------------|--------:|-------------:|
| Token Count Stabilized               | {0:8d} | {1:13.1f}% |
| Text Content Stabilized              | {2:8d} | {3:13.1f}% |
""".format(
        token_count_stable_count, token_count_stable_pct,
        text_stabilized_count, text_stabilized_pct
    )

    if avg_stabilization_round >= 0:
        markdown_report += f"\nAverage Stabilization Round: {avg_stabilization_round:.2f}\n"

    # Save the report
    report_path = os.path.join(output_dir, "token_analysis_summary.md")
    with open(report_path, 'w') as f:
        f.write(markdown_report)

    logger.info(f"Summary report saved to: {report_path}")
    return report_path

def main():
    """Main function to analyze token encoding/decoding"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate a unique run identifier
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.model.replace('/', '_')}_{args.distribution}_inlen{args.input_len}_maxlen{args.max_len}_prompts{args.num_prompts}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load tokenizer locally or PromptClient with Tokenizer
    if args.server_tokenizer:
        env_config = EnvironmentConfig(
            deploy_url="http://127.0.0.1",
            service_port='8000',
            authorization=None,
            jwt_secret="test1234",
            vllm_model="meta-llama/Llama-3.1-8B-Instruct",
            mesh_device="n300",
            cache_root='/home/user/tt-inference-server'
        )

        # # Set up the client
        client = PromptClient(env_config)
        actual_model = env_config.vllm_model
        tokenizer=actual_model
    else:
        try:
            tokenizer, actual_model = get_tokenizer(args.model, args.fallback_model)
            client=None
        except Exception as e:
            logger.error(f"Could not load any tokenizer: {e}")
            return

    logger.info("Generating random prompts...")
    logger.info(f"  Using tokenizer: {actual_model}")
    logger.info(f"  Distribution: {args.distribution}")
    logger.info(f"  Target length: {args.input_len} tokens")
    logger.info(f"  Number of prompts: {args.num_prompts}")
    if args.max_length_truncation is not None:
        logger.info(f"  Max length truncation: {args.max_length_truncation} tokens")


    # Parallelize prompt generation across 20 processors
    num_processors = 38
    prompts_per_processor = args.num_prompts // num_processors
    remainder = args.num_prompts % num_processors

    # Create a partial function with all parameters except the chunk size and index
    generate_chunk = partial(
        generate_prompts_chunk,
        max_length=args.max_len,
        input_seq_len=args.input_len,
        distribution=args.distribution,
        model_name=actual_model,
        server_tokenizer=args.server_tokenizer,
        client=client,
        tokenizer=tokenizer,
    )


    # Prepare arguments for each processor
    chunk_sizes = [prompts_per_processor + (1 if i < remainder else 0) for i in range(num_processors)]
    chunk_args = [(chunk_sizes[i], i) for i in range(num_processors)]

    # Create a process pool and map the function to the arguments
    logger.info(f"Generating prompts in parallel using {num_processors} processors...")
    with multiprocessing.Pool(processes=num_processors) as pool:
        prompt_chunks = pool.starmap(generate_chunk, chunk_args)

    # Merge all prompt chunks into a single list
    prompts = [prompt for chunk in prompt_chunks for prompt in chunk]
    logger.info(f"Generated {len(prompts)} prompts in total")

    # Prepare data for parallel processing
    prompt_data = [(i, prompt) for i, prompt in enumerate(prompts)]

    # Set number of processes
    num_processes = args.processes or min(multiprocessing.cpu_count(), len(prompts))
    logger.info(f"Using {num_processes} parallel processes for analysis")

    # Step 1: Analyze initial encoding growth
    logger.info("Analyzing initial encoding growth...")

    # Create partial function with the tokenizer model and input length
    analyze_initial_func = partial(
        analyze_initial_encoding_growth,
        tokenizer_model=actual_model,
        input_length=args.input_len,
        max_length_truncation=args.max_length_truncation,
        template=args.template,
        server_tokenizer=args.server_tokenizer,
        client=client
    )

    # Run analysis in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        initial_results = list(tqdm.tqdm(
            pool.imap(analyze_initial_func, prompt_data),
            total=len(prompt_data),
            desc="Analyzing initial encoding"
        ))

    # Step 2: Analyze prompt lossiness
    logger.info("Analyzing prompt lossiness...")

    # Create partial function with the tokenizer model and max length truncation
    analyze_lossiness_func = partial(
        analyze_prompt_lossiness,
        tokenizer_model=actual_model,
        input_length=args.input_len,
        max_length_truncation=args.max_length_truncation,
        server_tokenizer=args.server_tokenizer,
        client=client
    )

    # Run analysis in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        lossiness_results = list(tqdm.tqdm(
            pool.imap(analyze_lossiness_func, prompt_data),
            total=len(prompt_data),
            desc="Analyzing prompt lossiness"
        ))

    # Save raw results
    initial_results_file = os.path.join(run_dir, "initial_encoding_results.json")
    with open(initial_results_file, 'w') as f:
        json.dump(initial_results, f, indent=2)

    lossiness_results_file = os.path.join(run_dir, "prompt_lossiness_results.json")
    with open(lossiness_results_file, 'w') as f:
        json.dump(lossiness_results, f, indent=2)

    # Create summary report
    report_path = create_summary_report(initial_results, lossiness_results, args, run_dir)

    # Create visualizations

    # Initial encoding growth visualization
    df_initial = pd.DataFrame(initial_results)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_initial, x='raw_token_count', y='token_growth')
    plt.axhline(y=0, color='r', linestyle='--', label='No Growth')
    plt.xlabel("Raw Token Count")
    plt.ylabel("Token Growth")
    plt.title("Token Growth from Initial Encoding")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(run_dir, "initial_encoding_growth.png"))

    # Delta from target length visualization
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_initial, x='delta_from_target', bins=30)
    plt.axvline(x=0, color='r', linestyle='--', label='Target Length')
    plt.xlabel("Delta from Target Length (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of Deltas from Target Length")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(run_dir, "delta_from_target.png"))

    # Lossiness visualization
    df_lossiness = pd.DataFrame(lossiness_results)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_lossiness, x='input_length', y='lossiness')
    plt.xlabel("Input Length (tokens)")
    plt.ylabel("Lossiness")
    plt.title("Lossiness by Input Length")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "lossiness_by_input_length.png"))

    # Text similarity visualization
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_lossiness, x='input_length', y='text_similarity')
    plt.xlabel("Input Length (tokens)")
    plt.ylabel("Text Similarity (%)")
    plt.title("Text Similarity by Input Length")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "text_similarity_by_input_length.png"))

    # Token count delta visualization (added for growth/shrinkage analysis)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_lossiness, x='token_count_delta', bins=30)
    plt.axvline(x=0, color='r', linestyle='--', label='No Change')
    plt.xlabel("Token Count Delta (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of Token Count Changes During Roundtrip")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(run_dir, "token_count_delta.png"))

    # Growth vs Shrinkage visualization (added)
    plt.figure(figsize=(10, 6))
    counts = [
        sum(1 for r in lossiness_results if r['tokens_grew']),
        sum(1 for r in lossiness_results if r['tokens_shrank']),
        sum(1 for r in lossiness_results if r['tokens_unchanged'])
    ]
    labels = ['Grew', 'Shrank', 'Unchanged']
    colors = ['green', 'red', 'blue']
    plt.bar(labels, counts, color=colors)
    plt.ylabel('Number of Prompts')
    plt.title('Token Count Changes During Roundtrip')
    plt.savefig(os.path.join(run_dir, "token_growth_shrinkage.png"))

    # Multi-round stability visualization (added)
    plt.figure(figsize=(12, 6))
    # Find a typical prompt with multiple rounds of changes
    example_prompt = next((r for r in lossiness_results if len(set(r['token_counts'])) > 1), None)
    if example_prompt:
        rounds = list(range(len(example_prompt['token_counts'])))
        plt.plot(rounds, example_prompt['token_counts'], marker='o', label='Token Count')
        plt.ylabel('Number of Tokens')
        plt.xlabel('Roundtrip Number')
        plt.title(f'Token Count Stability Example (Prompt ID: {example_prompt["prompt_id"]})')
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, "token_stability_example.png"))

    # Display summary
    execution_time = time.time() - start_time
    logger.info(f"\nAnalysis completed in {execution_time:.2f} seconds")
    logger.info(f"Results saved to directory: {run_dir}")
    logger.info(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()