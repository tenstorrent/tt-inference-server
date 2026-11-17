#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Sends N identical POST /v1/videos/generations requests one after another and prints the client latency.

Usage: post_latency.py <base_url> <api_key> <n> <out.json>
"""
import json, sys, time, urllib.request, statistics
base, key, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({"prompt": "post latency", "num_inference_steps": 4}).encode()
lat, codes = [], {}
for i in range(n):
    r = urllib.request.Request(base + "/v1/videos/generations", data=body, method="POST")
    r.add_header("authorization", "Bearer " + key); r.add_header("content-type", "application/json")
    t = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            resp.read(); c = resp.status
    except urllib.error.HTTPError as e:
        e.read(); c = e.code
    lat.append(time.perf_counter_ns() - t); codes[c] = codes.get(c, 0) + 1
json.dump({"client_ns": lat, "codes": codes}, open(out, "w"))
xs = sorted(lat)
print(codes, "p50 %.3f ms  p95 %.3f ms  mean %.3f ms" % (xs[len(xs)//2]/1e6, xs[int(len(xs)*0.95)]/1e6, statistics.fmean(xs)/1e6))
