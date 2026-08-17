#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""
Test script for the C++ embedding endpoint.
Usage: python test_embedding.py [--host HOST] [--port PORT]
"""

import argparse
import json
import os
import time

import requests

DEFAULT_API_KEY = "your-secret-key"


def _auth_headers() -> dict:
    token = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    return {"Authorization": f"Bearer {token}"}


def test_health_check(base_url: str) -> bool:
    """Test the health endpoint."""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_liveness_check(base_url: str) -> bool:
    """Test the liveness endpoint."""
    print("\n=== Testing Liveness Check ===")
    try:
        response = requests.get(f"{base_url}/tt-liveness", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_embedding(base_url: str, model: str = "BAAI/bge-large-en-v1.5") -> bool:
    """Test a single embedding request."""
    print("\n=== Testing Single Embedding ===")

    payload = {"model": model, "input": "The quick brown fox jumps over the lazy dog."}

    print(f"Request: {json.dumps(payload, indent=2)}")

    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json=payload,
            headers={"Content-Type": "application/json", **_auth_headers()},
            timeout=60,
        )
        elapsed = time.time() - start_time

        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.3f}s")

        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data.get('model', 'N/A')}")
            print(f"Object: {data.get('object', 'N/A')}")

            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0]
                emb_vector = embedding.get("embedding", [])
                print(f"Embedding dimension: {len(emb_vector)}")
                print(f"First 5 values: {emb_vector[:5]}")
                print(f"Last 5 values: {emb_vector[-5:]}")

            if "usage" in data:
                print(f"Usage: {data['usage']}")

            return True
        else:
            print(f"Error response: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_embedding(base_url: str, model: str = "BAAI/bge-large-en-v1.5") -> bool:
    """Test batch embedding with multiple inputs."""
    print("\n=== Testing Batch Embedding ===")

    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language.",
        "Climate change is affecting global weather patterns.",
    ]

    # Note: OpenAI API supports array input, but our implementation may need single strings
    # Test with single requests in sequence
    print(f"Testing {len(texts)} texts sequentially...")

    total_time = 0
    success_count = 0

    for i, text in enumerate(texts):
        payload = {"model": model, "input": text}

        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json", **_auth_headers()},
                timeout=60,
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            if response.status_code == 200:
                data = response.json()
                emb_dim = len(data["data"][0]["embedding"]) if data.get("data") else 0
                print(f"  [{i + 1}] OK - {elapsed:.3f}s - dim={emb_dim}")
                success_count += 1
            else:
                print(f"  [{i + 1}] FAILED - {response.status_code}")

        except Exception as e:
            print(f"  [{i + 1}] ERROR - {e}")

    print(f"\nBatch Results: {success_count}/{len(texts)} successful")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time: {total_time / len(texts):.3f}s per request")

    return success_count == len(texts)


def test_throughput(
    base_url: str, num_requests: int = 10, model: str = "BAAI/bge-large-en-v1.5"
) -> bool:
    """Test throughput with multiple sequential requests."""
    print(f"\n=== Testing Throughput ({num_requests} requests) ===")

    text = "This is a test sentence for measuring embedding throughput performance."
    payload = {"model": model, "input": text}

    times = []
    success_count = 0

    start_total = time.time()

    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json", **_auth_headers()},
                timeout=60,
            )
            elapsed = time.time() - start_time
            times.append(elapsed)

            if response.status_code == 200:
                success_count += 1

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_requests}")

        except Exception as e:
            print(f"  Request {i + 1} failed: {e}")

    total_time = time.time() - start_total

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = success_count / total_time

        print("\nResults:")
        print(f"  Successful: {success_count}/{num_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} requests/sec")
        print(f"  Avg latency: {avg_time * 1000:.1f}ms")
        print(f"  Min latency: {min_time * 1000:.1f}ms")
        print(f"  Max latency: {max_time * 1000:.1f}ms")

    return success_count == num_requests


def test_error_handling(base_url: str) -> bool:
    """Test error handling with invalid requests."""
    print("\n=== Testing Error Handling ===")

    test_cases = [
        ("Empty input", {"model": "BAAI/bge-large-en-v1.5", "input": ""}),
        ("Missing model", {"input": "test"}),
        ("Missing input", {"model": "BAAI/bge-large-en-v1.5"}),
        ("Empty payload", {}),
    ]

    for name, payload in test_cases:
        try:
            response = requests.post(
                f"{base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json", **_auth_headers()},
                timeout=10,
            )
            print(f"  {name}: Status {response.status_code}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    return True  # Just informational


def main():
    parser = argparse.ArgumentParser(description="Test C++ embedding endpoint")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Model name")
    parser.add_argument(
        "--throughput-requests",
        type=int,
        default=10,
        help="Number of requests for throughput test",
    )
    parser.add_argument(
        "--skip-throughput", action="store_true", help="Skip throughput test"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token (defaults to OPENAI_API_KEY env or 'your-secret-key')",
    )
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    base_url = f"http://{args.host}:{args.port}"
    print(f"Testing embedding endpoint at {base_url}")
    print(f"Model: {args.model}")

    results = {}

    # Run tests
    results["health"] = test_health_check(base_url)
    results["tt-liveness"] = test_liveness_check(base_url)
    results["single"] = test_single_embedding(base_url, args.model)
    results["batch"] = test_batch_embedding(base_url, args.model)

    if not args.skip_throughput:
        results["throughput"] = test_throughput(
            base_url, args.throughput_requests, args.model
        )

    test_error_handling(base_url)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
