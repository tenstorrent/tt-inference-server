# ADR-001: Enable TCP_NODELAY for SSE Streaming

**Date:** 2026-02-21
**Status:** Accepted

## Context

The C++ media server uses Server-Sent Events (SSE) over HTTP chunked transfer encoding to stream LLM tokens to clients. Benchmarking with `vllm bench serve` showed ~44ms Time to First Token (TTFT) despite server-side logs indicating tokens were ready in ~1.3ms.

The root cause was the interaction between TCP Nagle's algorithm (enabled by default) and the client's delayed ACK timer (~40ms on Linux). Nagle buffers small writes when there is unacknowledged data; the client delays ACKs by 40ms hoping to piggyback on outgoing data. Together they create a fixed ~40ms stall on every small SSE chunk.

## Decision

Enable `TCP_NODELAY` on all accepted connections via Drogon's `setAfterAcceptSockOptCallback`, disabling Nagle's algorithm.

## Consequences

### TTFT (Time to First Token)


| Scenario            | Before        | After          |
| ------------------- | ------------- | -------------- |
| 1 concurrent user   | 44ms          | 0.6ms          |
| 64 concurrent users | 70ms (median) | 1.8ms (median) |


### TPOT (Time per Output Token) — tradeoff

TCP_NODELAY sends each SSE chunk as its own TCP packet instead of coalescing. This increases per-packet syscall overhead and **reduces throughput under high concurrency**:

| Scenario (64 users, 1000 requests) | Before | After |
|---|---|---|
| Output tok/s | 96,382 | 76,675 |
| TPOT | 0.08ms | 0.72ms |

This is a **~20% throughput regression** in the mock (instant) backend.

### Simulated real device validation

To verify the tradeoff under realistic conditions, we simulated blitz decode pipeline timing in the mock backend: 64 pipeline stages at 100us each, producing one token per user every 6.4ms. The 100us stage time was validated via instrumented logging (measured 104-105us actual).

| Metric | Simulated result |
|---|---|
| TPOT (64 users) | 6.9ms |
| Device time | 6.4ms (93%) |
| Server overhead | 0.5ms (7%) |

**The server overhead is 7% of total TPOT.** The device pipeline dominates; the TCP_NODELAY per-packet cost is negligible at realistic token rates. Without TCP_NODELAY, TTFT (~44ms) would exceed the entire blitz decode token interval (6.4ms), causing multi-token buffering before clients see anything.

