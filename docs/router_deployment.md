# Serving gemma-4 behind the vLLM Router

How to deploy gemma-4 (and any other entry with more than one engine per box)
so that a request router can keep a conversation on the engine that holds its
prefix cache. Nothing in this layout is tt-inference-server code: the routing
is the official [vLLM Router](https://github.com/vllm-project/router), the
serving is the ordinary catalog entries, and the glue is deployment
configuration.

## Why a router, and why per-engine workers

Every vLLM engine owns its KV cache, so automatic prefix caching is per engine.
A whole-box entry such as gemma-4-31B-it on `BLACKHOLE_GALAXY` runs four
engines (`data_parallel_size: 4`) behind one API server whose load balancer
picks an engine from load counts only. A multi-turn conversation therefore
lands on a different engine each turn and warms every cache separately:
measured on that entry, one repeated prompt warmed three of the four engines
to a 48.6% hit rate instead of hitting one engine from the second request on.

An external router can only steer what it can address, and the internal-DP
endpoint is one opaque URL. So the deployable unit for routed serving is **one
engine per instance**: on a Blackhole Galaxy, four instances of the
`P150X8` (8-chip, TP=8) entry rather than one `BLACKHOLE_GALAXY` instance;
on a QuietBox 2, one `P300X2` instance per box. Each instance is a worker
the router discovers through its own URL.

## Proof of concept (2026-10-15)

Two independent gemma-4-31B-it instances (TP=4, chips 0-3 and 4-7 of one
Galaxy, prefix caching on) behind `vllm-router` 0.1.16, no server-side
change. Two 4-turn conversations (~7.2K-token prompts, history re-sent each
turn), then conversation A again from a cold client. Hit rates come from each
worker's own `vllm:prefix_cache_hits_total / queries_total`.

| policy | follow-up turns | re-sent conversation | new sessions spread |
|---|---|---|---|
| `consistent_hash` + `X-Session-ID` | one worker, 74% hit (turn 1 is the only miss) | same worker, 99% | 8 session ids -> 5 / 3 |
| `consistent_hash`, no session key | one worker | same worker | **none**: every unkeyed request went to one worker |
| `cache_aware` (defaults) | one worker, 74% | same worker, 99% | none at low load (balancing starts past a 32-request imbalance) |
| `cache_aware --balance-abs-threshold 0 --balance-rel-threshold 1.0` | one worker, 74% | same worker, 99% | 8 concurrent fresh prompts -> 4 / 4 |

## Deployment layout

1. **Workers.** One tt-inference-server instance per engine, each on its own
   port. With the Helm chart this is `replicaCount: 4` of the `P150X8` entry
   on a Galaxy: each Pod claims 8 boards through the tt-operator DRA driver
   (`deviceBoardCounts.p150x8`), and the chart's Service selects all
   replicas. The DRA placement has to hand each Pod a topologically valid
   8-chip block (a 2x4 mesh); that is a tt-operator concern, not a catalog
   one. Without Kubernetes, start the instances by hand with disjoint
   `TT_VISIBLE_DEVICES` sets, as the proof of concept did.
2. **Router.** `vllm-router` (pip wheel or Docker image) in front of the
   workers, in the cluster above the instances (Console's k8s), for example:
   ```bash
   vllm-router --host 0.0.0.0 --port 8000 \
     --policy consistent_hash \
     --service-discovery --selector app=tt-inference-server model=gemma-4-31B-it
   # or a static list: --worker-urls http://w1:8000 http://w2:8000 ...
   ```
   `consistent_hash` keys on `X-Session-ID` (then `X-User-ID`, `X-Tenant-ID`,
   `X-Request-ID`, `X-Correlation-ID`, `X-Trace-ID`, then the body's
   `session_params.session_id`, `user`, `session_id`, `user_id`). **Without a
   key the policy pins all traffic to one worker**, so the gateway in front
   of the router must set a session header. `cache_aware` needs no key (it
   keeps an approximate radix tree of prompt prefixes per worker) but with
   the default balance thresholds it fills one worker before spreading; lower
   `--balance-abs-threshold` / `--balance-rel-threshold` for small worker
   counts.
3. **Cache visibility.** The dev-catalog gemma-4 entries pass vLLM's
   `--enable-prompt-tokens-details`, so responses carry
   `usage.prompt_tokens_details.cached_tokens`. Marketplaces such as
   OpenRouter read that field to credit cache hits and to route stickily.

## What this does not cover

- Cross-worker KV reuse: a worker only hits its own cache. That is the step
  [vLLM Production Stack](https://github.com/vllm-project/production-stack)
  or llm-d adds, and it is not required for sticky sessions.
- Prefill/decode disaggregation awareness (llm-d), which the TT plugin does
  not support yet.
- In-process submesh DP (`TT_DATA_PARALLEL`) inside one instance, which the
  router cannot see.

## Verifying a deployment

Send one conversation of several turns through the router with a fixed
`X-Session-ID`, then read `vllm:prefix_cache_hits_total` on each worker's
`/metrics`: only one worker's counter should move, by the re-sent prefix
length per turn. With `--enable-prompt-tokens-details`, the same shows up
per response as `cached_tokens`.
