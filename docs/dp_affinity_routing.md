# Session-affine routing across data-parallel engines

Applies to catalog entries with `vllm_args.data_parallel_size` > 1 (for
example gemma-4 on BLACKHOLE_GALAXY: four TP=8 engines behind one API server).

## Why

Each vLLM data-parallel engine owns its KV cache, so automatic prefix caching
is per engine. The API server's load balancer picks an engine from load counts
alone, so a multi-turn conversation lands on a different engine each turn and
warms every cache separately. Measured on gemma-4-31B-it BLACKHOLE_GALAXY: one
prompt repeated seven times warmed three of the four engines to a 48.6% hit
rate instead of hitting one engine every time after the first request.

vLLM accepts a per-request engine pin on the chat and completions endpoints:
the `X-data-parallel-rank` header, used by the load balancer before any load
scoring. `vllm-tt-metal/src/dp_affinity_middleware.py` sets that header from a
stable session key so a conversation stays on the engine that holds its prefix.
The launcher registers it (vLLM `--middleware`) whenever
`data_parallel_size` > 1.

## Modes (`TT_DP_AFFINITY`)

| Mode | Pins when | Use for |
|---|---|---|
| `auto` (default) | the request carries a session signal: the `X-TT-Affinity-Key` header, else the `user` field of the body | gateways and clients that identify a session (OpenRouter-style traffic, chat apps). Requests without a signal keep vLLM's load-aware balancing, so benchmark sweeps (no `user`, per-request `cache_salt`) are unchanged |
| `prefix` | as `auto`, otherwise a hash of the first `TT_DP_AFFINITY_PREFIX_BYTES` (default 4096) bytes of the prompt text | callers that send no session signal but replay multi-turn conversations (agentic traces). Pins every request, so load can skew when few distinct prefixes are in flight |
| `off` | never | disable without touching the catalog |

Set the mode in the environment of the server process (`TT_DP_AFFINITY=prefix`
before `run.py ... --workflow server`, or in the catalog entry's `env_vars`).
The launcher publishes the engine count to the middleware as
`TT_DP_AFFINITY_DP_SIZE`.

## Interaction with vLLM's own header

A client-supplied `X-data-parallel-rank` is honoured but reduced modulo the
engine count. vLLM indexes its engine list with the raw value, so an
out-of-range rank from a client would otherwise fail the request.

## What it does not cover

- In-process submesh data parallelism (`TT_DATA_PARALLEL`, plugin
  `tt_data_parallel`) is routed by the plugin's lane scheduler and does not
  see the header.
- Multi-host deployments that run one API server per engine
  (`data_parallel_external_lb` / `data_parallel_hybrid_lb`) need the external
  router to pin the rank; the middleware only sees requests that reach one
  API server.

## Verifying

Start the server with `--enable-prompt-tokens-details` (vLLM reports
`usage.prompt_tokens_details.cached_tokens` only with that flag), send one
long prompt several times with the same `user`, and check that
`cached_tokens` is non-zero from the second request on and that only one
engine's `Prefix cache hit rate` rises in the server log.
