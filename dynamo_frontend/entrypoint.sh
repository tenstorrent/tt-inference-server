#!/bin/bash
set -e

# Tokenizers are baked into the image at the path each worker advertises in its
# MDC, so the frontend resolves every discovered model locally and serves
# whatever registers in etcd — no --model-path overlay or runtime download.

# Default to Dynamo's Rust-native chat processor. The HTTP server, router,
# and preprocessing all live in Rust (driven by Tokio); Python is just the
# thin supervisor that awaits the server future. To actually parallelize
# request handling on the Rust side, tune the runtime threads below — the
# vllm/sglang chat-processor + ProcessPoolExecutor route is only needed if
# you want subprocess isolation, which we don't.
DYN_CHAT_PROCESSOR="${DYN_CHAT_PROCESSOR:-dynamo}"

# Dynamo's Rust RuntimeConfig (lib/runtime/src/config.rs in upstream) reads
# DYN_RUNTIME_* and DYN_COMPUTE_* env vars at startup.
#   DYN_RUNTIME_NUM_WORKER_THREADS  — Tokio I/O runtime worker pool
#   DYN_RUNTIME_MAX_BLOCKING_THREADS — Tokio blocking-pool (used when
#                                      Tokio offloads CPU work like
#                                      tokenize + chat-template render)
#   DYN_COMPUTE_THREADS             — dedicated compute pool, separate
#                                     from the I/O runtime. The HF
#                                     tokenizer crate parks here.
# Default policy: scale all three to the container's logical CPU count so
# preprocessing actually runs in parallel across cores. Override with the
# matching env at deploy time when you want to pin them lower (e.g. to
# co-locate with other containers on a shared host).
HOST_CPUS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
DYN_RUNTIME_NUM_WORKER_THREADS="${DYN_RUNTIME_NUM_WORKER_THREADS:-$HOST_CPUS}"
DYN_RUNTIME_MAX_BLOCKING_THREADS="${DYN_RUNTIME_MAX_BLOCKING_THREADS:-$((HOST_CPUS * 4))}"
DYN_COMPUTE_THREADS="${DYN_COMPUTE_THREADS:-$HOST_CPUS}"
# HuggingFace `tokenizers` crate uses Rayon for any per-input parallelism.
# Rayon's default is num_cpus; pin it explicitly so the inside-container
# default doesn't collapse to 1 if cgroup CPU shares are misread.
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$HOST_CPUS}"
export DYN_RUNTIME_NUM_WORKER_THREADS DYN_RUNTIME_MAX_BLOCKING_THREADS \
       DYN_COMPUTE_THREADS RAYON_NUM_THREADS

# Tokenizer backend selection. The default HuggingFace tokenizer wraps each
# call in mutable state which serializes concurrent encodes across requests
# (likely culprit when --dyn-runtime-num-worker-threads >> 1 yields no speedup
# because every Tokio worker piles up on the same tokenizer lock). The
# `fastokens` backend is a Rust BPE re-implementation that's stateless per
# call and lets concurrent requests actually proceed in parallel — needs
# to be opt-in because it can differ on edge-case tokens.
DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}"
export DYN_TOKENIZER

# Optional perf tracing — only respected by vllm/sglang processors, but
# harmless to pass through.
DEBUG_PERF_FLAG=""
if [ "${DYN_DEBUG_PERF:-0}" != "0" ]; then
    DEBUG_PERF_FLAG="--dyn-debug-perf"
fi

# Optional router mode. Default (unset) leaves Dynamo's own default
# (round-robin). Set ROUTER_MODE to one of round-robin|random|power-of-two|
# kv|direct|least-loaded|device-aware-weighted. Note: per-request timing
# fields like nvext.timing.ttft_ms / prefill_time_ms are only populated on
# the kv-router push path, so ROUTER_MODE=kv is required to surface them.
ROUTER_MODE_FLAG=""
if [ -n "${ROUTER_MODE:-}" ]; then
    ROUTER_MODE_FLAG="--router-mode ${ROUTER_MODE}"
fi

# TT native prefill routing policy contract. ai-dynamo 1.2.0.post1 does not
# consume this env directly yet; C++ workers also advertise the same threshold
# in their MDC runtime_config so a Dynamo router patch has a stable input.
TT_NATIVE_PREFILL_POLICY="${TT_NATIVE_PREFILL_POLICY:-threshold}"
TT_PREFILL_ON_DECODE_MAX_TOKENS="${TT_PREFILL_ON_DECODE_MAX_TOKENS:-1000}"
export TT_NATIVE_PREFILL_POLICY TT_PREFILL_ON_DECODE_MAX_TOKENS

echo "[entrypoint] Starting Dynamo frontend on port $HTTP_PORT..."
echo "  DYN_DISCOVERY_BACKEND=$DYN_DISCOVERY_BACKEND"
if [ "$DYN_DISCOVERY_BACKEND" = "etcd" ]; then
    echo "  ETCD_ENDPOINTS=$ETCD_ENDPOINTS"
fi
echo "  DYN_REQUEST_PLANE=$DYN_REQUEST_PLANE"
echo "  DYN_EVENT_PLANE=$DYN_EVENT_PLANE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  DYN_CHAT_PROCESSOR=$DYN_CHAT_PROCESSOR"
echo "  DYN_RUNTIME_NUM_WORKER_THREADS=$DYN_RUNTIME_NUM_WORKER_THREADS"
echo "  DYN_RUNTIME_MAX_BLOCKING_THREADS=$DYN_RUNTIME_MAX_BLOCKING_THREADS"
echo "  DYN_COMPUTE_THREADS=$DYN_COMPUTE_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "  DYN_TOKENIZER=$DYN_TOKENIZER"
echo "  RUST_LOG=${RUST_LOG:-<unset>}"
echo "  DEBUG_PERF_FLAG=$DEBUG_PERF_FLAG"
echo "  ROUTER_MODE=${ROUTER_MODE:-<default>}"
echo "  TT_NATIVE_PREFILL_POLICY=$TT_NATIVE_PREFILL_POLICY"
echo "  TT_PREFILL_ON_DECODE_MAX_TOKENS=$TT_PREFILL_ON_DECODE_MAX_TOKENS"

exec python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --model-name "$MODEL_NAME" \
    --dyn-chat-processor "$DYN_CHAT_PROCESSOR" \
    --tokenizer "$DYN_TOKENIZER" \
    $ROUTER_MODE_FLAG \
    $DEBUG_PERF_FLAG
