#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Cleaned Benchmark Serving Script

This script benchmarks the serving performance using cleaned random prompts generated
via the CleanedPromptGenerator with server-side tokenization.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

import aiohttp
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# Import our cleaned prompt generator
from cleaned_prompt_generator import CleanedPromptGenerator
from prompt_client import PromptClient
from prompt_configs import EnvironmentConfig

# Import necessary benchmark functions (these will be imported from benchmark_serving.py)
# For now, we'll copy the necessary parts since we can't import cross-directory easily

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    extra_body: Optional[dict] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_tokens: Optional[int] = None
    error: str = ""


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def remove_prefix(text: str, prefix: str) -> str:
    """Remove prefix from text if present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[atqdm] = None,
    auth_headers: Optional[Dict[str, str]] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        
        if request_func_input.best_of is not None and request_func_input.best_of > 1:
            payload["best_of"] = request_func_input.best_of
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.logprobs is not None:
            payload["logprobs"] = request_func_input.logprobs

        # Use provided auth headers or fallback to OpenAI-style headers
        if auth_headers:
            headers = auth_headers.copy()
        else:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"
            }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output_tokens = 0
        
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # Check if we have token data
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "text" in choice and choice["text"]:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft
                                    # Decoding phase
                                    else:
                                        output.itl.append(timestamp -
                                                          most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += choice["text"]
                                    output_tokens += 1
                            
                            # Check for usage stats
                            if "usage" in data:
                                output_tokens = data["usage"].get("completion_tokens", output_tokens)

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_tokens = output_tokens
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def setup_server_client(args, model_config):
    """Set up the client for server-side tokenization"""
    
    # Create environment config with defaults from environment variables
    env_config = EnvironmentConfig()
    
    # Override with command-line arguments and model config
    env_config.jwt_secret = args.jwt_secret if hasattr(args, 'jwt_secret') else os.getenv("JWT_SECRET", "")
    env_config.service_port = str(args.port)
    env_config.vllm_model = model_config
    
    # Create prompt client
    client = PromptClient(env_config)
    
    # Wait for server to be healthy (optional)
    print("Checking server health...")
    if client.wait_for_healthy(timeout=30):
        print("Server is healthy and ready!")
        return client
    else:
        print("Server health check failed!")
        return None


def generate_cleaned_random_requests(
    num_prompts: int,
    input_len: int,
    output_len: int,
    model_name: str,
    client: PromptClient,
) -> List[Tuple[str, int, int, None]]:
    """
    Generate cleaned random prompts using CleanedPromptGenerator with server-side tokenization.
    
    Returns list of tuples: (prompt_text, prompt_len, output_len, multi_modal_content)
    """
    print(f"Generating {num_prompts} cleaned random prompts...")
    
    # Initialize the generator with server-side tokenization
    generator = CleanedPromptGenerator(
        model_name=model_name,
        server_tokenizer=True,
        client=client,
        seed=42  # Fixed seed for reproducibility
    )
    
    input_requests = []
    
    # Generate prompts with a progress bar
    for i in tqdm(range(num_prompts), desc="Generating prompts"):
        # Generate stable tokens
        tokens = generator.generate_stable_tokens(
            input_length=input_len,
            max_length=input_len,  # Keep it at the exact requested length
            seed=42 + i  # Different seed for each prompt
        )
        
        # Decode tokens back to text for the prompt
        prompt_text = generator._decode(tokens)
        
        # The actual token count might be slightly different due to cleaning
        prompt_len = len(tokens)
        
        # Add to requests list
        input_requests.append((prompt_text, prompt_len, output_len, None))
    
    print(f"Generated {len(input_requests)} cleaned prompts")
    avg_prompt_len = sum(r[1] for r in input_requests) / len(input_requests)
    print(f"Average prompt length: {avg_prompt_len:.1f} tokens")
    
    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int, None]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int, None], None]:
    """Generate requests at the specified rate."""
    for request in input_requests:
        yield request
        
        if request_rate != float("inf"):
            interval = 1.0 / request_rate
            await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int, None]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    """Calculate benchmark metrics from outputs."""
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens
            if output_len is None:
                # Estimate from generated text length
                output_len = len(outputs[i].generated_text.split())
            
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)
    
    if completed == 0:
        print("Warning: All requests failed!")
    
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=completed / dur_s,  # Simplified for this version
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )
    
    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    model_name: str,
    input_requests: List[Tuple[str, int, int, None]],
    request_rate: float,
    max_concurrency: Optional[int],
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    auth_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the benchmark."""
    
    if backend != "vllm":
        raise ValueError(f"Only 'vllm' backend is supported, got: {backend}")
    
    print("Starting benchmark...")
    print(f"Request rate: {request_rate}")
    print(f"Maximum request concurrency: {max_concurrency}")
    
    pbar = atqdm(total=len(input_requests))
    
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)
    
    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await async_request_openai_completions(
                request_func_input=request_func_input, pbar=pbar, auth_headers=auth_headers)
        async with semaphore:
            return await async_request_openai_completions(
                request_func_input=request_func_input, pbar=pbar, auth_headers=auth_headers)
    
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    
    async for prompt, prompt_len, output_len, mm_content in get_request(
            input_requests, request_rate):
        request_func_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    
    pbar.close()
    
    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )
    
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))
    
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [r[1] for r in input_requests],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    
    # Add percentile metrics
    for metric_name in ["ttft", "tpot", "itl", "e2el"]:
        if metric_name in selected_percentile_metrics:
            print("{s:{c}^{n}}".format(s=f' {metric_name.upper()} ', n=50, c='-'))
            print("{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_name}_ms")))
            print("{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_name}_ms")))
            
            result[f"mean_{metric_name}_ms"] = getattr(
                metrics, f"mean_{metric_name}_ms")
            result[f"median_{metric_name}_ms"] = getattr(
                metrics, f"median_{metric_name}_ms")
            result[f"std_{metric_name}_ms"] = getattr(
                metrics, f"std_{metric_name}_ms")
            
            for p, value in getattr(metrics, f"percentiles_{metric_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                                value))
                result[f"p{p_word}_{metric_name}_ms"] = value
    
    print("=" * 50)
    
    return result


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: Dict[str, Any],
                                     file_name: str) -> None:
    """Save results in PyTorch benchmark format."""
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    
    pt_records = {
        "name": "cleaned_benchmark_serving",
        "model": args.model,
        "backend": args.backend,
        "metrics": {k: results.get(k, 0) for k in metrics},
        "extra_info": {
            "num_prompts": args.num_prompts,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "request_rate": "inf" if args.request_rate == float("inf") else args.request_rate,
            "completed": results["completed"],
            "duration": results["duration"],
        }
    }
    
    pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
    with open(pt_file, "w") as f:
        json.dump(pt_records, f, indent=2)
    print(f"PyTorch benchmark format saved to: {pt_file}")


def main(args: argparse.Namespace):
    """Main function to run the cleaned benchmark."""
    print(args)
    
    # Set up server client for tokenization
    client = setup_server_client(args, args.model)
    if client is None:
        print("Failed to connect to server. Exiting.")
        return
    
    # API setup
    api_url = f"http://127.0.0.1:{args.port}/v1/completions"
    
    # Generate cleaned random prompts
    input_requests = generate_cleaned_random_requests(
        num_prompts=args.num_prompts,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        model_name=args.model,
        client=client,
    )
    
    # Capture traces for the input/output length combination being used (unless disabled)
    if not args.disable_trace_capture:
        print("Capturing traces for input/output length combination...")
        context_lens = [(args.random_input_len, args.random_output_len)]
        client.capture_traces(context_lens=context_lens, timeout=1200.0)
    else:
        print("Trace capture disabled, skipping...")
    
    # Run benchmark
    benchmark_result = asyncio.run(
        benchmark(
            backend=args.backend,
            api_url=api_url,
            model_id=args.model,
            model_name=args.model,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            auth_headers=client.headers,
        ))
    
    # Save results if requested
    if args.save_result:
        from datetime import datetime
        
        result_json: Dict[str, Any] = {
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "backend": args.backend,
            "model_id": args.model,
            "num_prompts": args.num_prompts,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "request_rate": (args.request_rate if args.request_rate < float("inf") 
                             else "inf"),
            "max_concurrency": args.max_concurrency,
        }
        
        # Merge with benchmark result
        result_json.update(benchmark_result)
        
        # Determine filename
        if args.result_filename:
            file_name = args.result_filename
        else:
            base_model_id = args.model.split("/")[-1]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"cleaned-{args.backend}-{base_model_id}-{timestamp}.json"
        
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        
        # Save JSON results
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile, indent=2)
        print(f"Results saved to: {file_name}")
        
        # Save PyTorch format
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark serving with cleaned random prompts.")
    
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Backend to use (only vllm supported)."
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the serving API."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model."
    )
    
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process."
    )
    
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If inf, all requests sent at once."
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests."
    )
    
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request."
    )
    
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request."
    )
    
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to JSON file."
    )
    
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save benchmark results."
    )
    
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Filename for benchmark results."
    )
    
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-separated list of metrics to report percentiles."
    )
    
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles to report."
    )
    
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already running and traces captured."
    )
    
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    
    args = parser.parse_args()
    main(args) 