# Zero-Overhead Inference Server

Production-grade, zero-overhead C++ inference server for AI workloads on
Tenstorrent hardware. Supports LLM and image models; video, audio,
and text-to-speech models are on the roadmap.

## Non-Functional Requirements

These requirements drive design decisions in this codebase. They are ordered
by priority — when two requirements conflict, the higher-ranked one wins.

### 1. Performance (Zero-Overhead)

The server is the envelope around the model; that envelope must be invisible.
Every microsecond of framework overhead (serialization, scheduling, IPC,
queue management) is a design failure. The goal is not "fast" — it is
"never the bottleneck."

Solution approach:
- Prefer zero-copy and shared-memory IPC (`/dev/shm`) over serialization.
- Performance benchmarks to catch regressions before they ship.
- Tools and agent skills to help diagnose and fix performance issues.

### 2. Minimize KV cache recomputations

Computed KV cache is preserved and reused as long as possible.

Solution approach:
- Prefix caching - the same prefix is reused for the same user and across users.
- KV copy between slots - when we don't want users to share a memory slot, we copy existing KV cache to a new slot.
- KV cache migration - KV cache computed by other nodes in the cluster is migrated to the node that needs it
- Offloading KV cache to host RAM and SSD

### 3. Observability

This is a customer-facing product. Clients and field engineers must be able
to diagnose production issues (slow tokens, hung devices, degraded throughput)
without reading source code.

Solution approach:
- Structured logging with request-ID correlation across HTTP → worker → device.
- Production metrics (tokens/sec, queue depth, latency histograms, device health).
- Health/liveness endpoints that surface meaningful diagnostics, not just "alive."
- Sentry distributed tracing of Dynamo generate requests (see below).

#### Sentry distributed tracing

A request arriving over the Dynamo transport with a W3C `traceparent` header
continues that trace with a single Sentry transaction per server role:
`decode/generate` and `prefill/generate` in disaggregated mode, `generate`
otherwise. A request without a (valid) `traceparent` publishes nothing — the
server never starts a trace of its own. On the decode → prefill ZMQ hop the
decode transaction's traceparent rides inside `PrefillRequestMessage`, so
both halves of a disaggregated request appear in one trace.

Configuration (tracing is disabled unless `SENTRY_DSN` is set):

| Env var | Meaning | Default |
|---|---|---|
| `SENTRY_DSN` | Sentry project DSN; enables tracing | unset (disabled) |
| `SENTRY_ENVIRONMENT` | Sentry environment tag | `development` |
| `SENTRY_RELEASE` | Release override | server version |
| `SENTRY_TRACES_SAMPLE_RATE` | Transaction sample rate 0.0–1.0 | `1.0` |
| `SENTRY_DEBUG` | `1` logs SDK activity to stderr | off |

The Dynamo frontend only injects `traceparent` towards the worker when its
OTel layers are installed, which requires running it with
`DYN_LOGGING_JSONL=1` (default in `dynamo_frontend/Dockerfile.frontend`,
`deploy.sh`, and `scripts/start_dynamo.sh`; the frontend logs become JSON
lines). The frontend does not forward `baggage` / `sentry-trace` headers, so
the server deliberately handles only `traceparent`.

### 4. Extensibility

New model types (image, video, audio, text-to-speech) will be added with increasing frequency. The
server must make this straightforward without modifying core infrastructure.

Solution approach
- Stable core that rarely changes.
- Model-specific logic isolated in adapter layers (runner implementations).
- When hardware topology requires custom communication or scheduling, changes
are confined to the worker level — the main server remains unchanged.
