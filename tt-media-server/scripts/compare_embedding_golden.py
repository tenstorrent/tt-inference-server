# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Validate a running embedding server against captured golden vectors.

Two phases:

1. Sequential: POST every golden prompt one at a time and compare the returned
   vector against the prompt's ``embedding_single`` golden (exact dimension,
   exact token count, cosine similarity above a threshold, optionally
   bit-exact).

2. Concurrent: fire a barrier-synchronized burst so requests share dynamic
   batches, then verify each response still matches its own prompt's golden
   (single or batched variant) above a threshold. This exercises the dynamic
   batching path under real concurrency.

Usage (server must already be running; see
cpp_server/docs/embedding_serving_guide.md for the launch recipe):

    python scripts/compare_embedding_golden.py \
        --golden scripts/goldens/bge_large_en_v1_5_n150.json

Exit code 0 = all checks passed, 1 = at least one failure.
"""

import argparse
import concurrent.futures
import json
import math
import sys
import threading
import urllib.error
import urllib.request


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def post_embedding(url: str, api_key: str, text: str, timeout: float,
                   model: str):
    body = json.dumps({"input": text, "model": model}).encode()
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp["data"][0]["embedding"], resp["usage"]["total_tokens"]


def run_sequential(prompts, args, failures):
    print("=== phase 1: sequential, vs embedding_single goldens ===")
    for p in prompts:
        try:
            vec, tokens = post_embedding(args.url, args.api_key, p["text"],
                                         args.timeout, args.model)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            failures.append(f"sequential {p['id']}: request failed: {e}")
            print(f"FAIL {p['id']:>18}: {e}")
            continue

        golden = p["embedding_single"]
        problems = []
        if len(vec) != len(golden):
            problems.append(f"dim {len(vec)} != {len(golden)}")
        if tokens != p["token_count"]:
            problems.append(f"tokens {tokens} != {p['token_count']}")
        cos = cosine(vec, golden) if len(vec) == len(golden) else 0.0
        if cos < args.threshold:
            problems.append(f"cos {cos:.6f} < {args.threshold}")
        bit_exact = vec == golden
        if args.require_bit_exact and not bit_exact:
            problems.append("not bit-exact")

        status = "FAIL" if problems else "ok"
        exact = "bit-exact" if bit_exact else f"cos={cos:.8f}"
        print(f"{status:>4} {p['id']:>18}: {exact} tokens={tokens}"
              + (f"  [{'; '.join(problems)}]" if problems else ""))
        if problems:
            failures.append(f"sequential {p['id']}: {'; '.join(problems)}")


def run_concurrent(prompts, args, failures):
    burst = args.burst if args.burst > 0 else min(8, len(prompts))
    # Cycle prompts to fill the burst if it exceeds the prompt count.
    work = [prompts[i % len(prompts)] for i in range(burst)]
    barrier = threading.Barrier(len(work))
    print(f"=== phase 2: concurrent burst of {len(work)} ===")

    def worker(p):
        barrier.wait()
        try:
            vec, _ = post_embedding(args.url, args.api_key, p["text"],
                                    args.timeout, args.model)
            return p, vec, None
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            return p, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as ex:
        results = list(ex.map(worker, work))

    for p, vec, err in results:
        if err is not None:
            failures.append(f"concurrent {p['id']}: request failed: {err}")
            print(f"FAIL {p['id']:>18}: {err}")
            continue

        # The response may match the single-inference golden (request ran in
        # its own batch) or the batched golden (shared a batch); padding
        # differences make small deviations from either legitimate.
        cos_single = cosine(vec, p["embedding_single"])
        cos_batched = cosine(vec, p["embedding_batched"]) \
            if "embedding_batched" in p else cos_single
        best_cos = max(cos_single, cos_batched)

        problems = []
        if best_cos < args.concurrent_threshold:
            problems.append(
                f"cos {best_cos:.6f} < {args.concurrent_threshold} (single "
                f"{cos_single:.6f}, batched {cos_batched:.6f})")

        status = "FAIL" if problems else "ok"
        print(f"{status:>4} {p['id']:>18}: cos_single={cos_single:.6f} "
              f"cos_batched={cos_batched:.6f}"
              + (f"  [{'; '.join(problems)}]" if problems else ""))
        if problems:
            failures.append(f"concurrent {p['id']}: {'; '.join(problems)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--golden", required=True,
                        help="Path to the golden JSON captured by "
                             "capture_embedding_golden.py")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running server")
    parser.add_argument("--api-key", default="your-secret-key",
                        help="Bearer token (OPENAI_API_KEY of the server)")
    parser.add_argument("--threshold", type=float, default=0.999,
                        help="Minimum cosine similarity, sequential phase "
                             "(single requests reproduce goldens bit-exactly, "
                             "so this only leaves headroom for future stacks)")
    parser.add_argument("--concurrent-threshold", type=float, default=0.9985,
                        help="Minimum cosine similarity, concurrent phase. "
                             "Looser than sequential because arbitrary batch "
                             "compositions pad differently than the golden "
                             "capture did; the measured single-vs-batched "
                             "floor for BGE on n150 is 0.99857")
    parser.add_argument("--require-bit-exact", action="store_true",
                        help="Sequential phase must match goldens byte for "
                             "byte (the current server achieves this)")
    parser.add_argument("--burst", type=int, default=0,
                        help="Concurrent burst size; 0 = min(8, #prompts). "
                             "Values above the model's max_batch_size "
                             "reproduce the MAX_IN_FLIGHT_COUNT overflow")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    with open(args.golden) as f:
        golden = json.load(f)
    prompts = golden["prompts"]
    meta = golden.get("metadata", {})
    args.model = meta.get("model", "unknown")
    print(f"golden: {args.golden}")
    print(f"model={meta.get('model')} device={meta.get('device')} "
          f"tt_metal_commit={meta.get('tt_metal_commit')} "
          f"prompts={len(prompts)}")

    failures: list = []
    run_sequential(prompts, args, failures)
    run_concurrent(prompts, args, failures)

    print("=== summary ===")
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("PASSED: sequential + concurrent, all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
