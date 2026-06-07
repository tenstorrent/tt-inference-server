# Qwen3-32B release benchmark exceeds the 6h CI cap — FIXED (output not capped)

Part of the forge-TP-p300x2 umbrella. Qwen3-32B release benchmark
([tt-shield release run 5204](https://github.com/tenstorrent/tt-shield/actions/runs/27054514200))
hit the **6h job cap**. **Root cause found and fix verified locally.**

## Root cause (confirmed on a live Qwen server)
`vllm bench serve --random-output-len N` is **not honored as a `max_tokens` cap** by the forge
OpenAI-compat server, so benchmark requests run **unbounded** — and **Qwen3-32B emits ~1000–1500
reasoning (`<think>`) tokens** for an osl=128 request. ~13 configs × ~1000+ tokens ≈ the 6h. Confirmed:
- Decode is **healthy** (~5 tok/s, TPOT ~185 ms) — *not* the bottleneck (see the conc sweep below).
- A direct request with explicit `max_tokens` **is** honored (probe: max_tokens=32 → 32 tokens;
  max_tokens=128 → 111). So it's a request-not-capped issue, not a server enforcement bug.
- `chat_template_kwargs={"enable_thinking": false}` did **not** disable Qwen3 thinking via this server.
- gemma is **not** affected — uncapped it stops much shorter (no long reasoning).

## Fix (implemented + verified)
`benchmarking/run_benchmarks.py`: add `"max_tokens": osl` to the text-task `--extra-body` (alongside
the existing `truncate_prompt_tokens`). The server honors `max_tokens`, so output caps at the requested
osl. Applies to all text benchmarks (a correctness improvement; only changes behavior for models that
were over-generating).

**Verification (local, live Qwen server):**
| | duration | result |
|---|---|---|
| `bench serve` conc-1, n=2, **no** `max_tokens` | **>600 s (timed out)** | ~1500 tok/req |
| `bench serve` conc-1, n=2, **+`max_tokens`** | **49 s** | ~128 tok/req |
| `run.py --workflow benchmarks` smoke (conc-1, n=8) e2e **+fix** | **210 s** (8/8) | ~128 tok/req, TPOT 185 ms |

Same config was ~24–36 min/config uncapped in CI → ~3.5 min with the fix (**~8–10×**). A full ~13-config
sweep now lands well under 6h.

## Concurrency sweep (decode is healthy — no conc-1/batch pathology)
| conc | per-request tok/s | aggregate tok/s |
|---|---|---|
| 1 | 5.04 | 5.04 |
| 4 | 5.03 | 20.1 |
| 16 | 5.05 | 80.8 |

## Notes
- **Separate follow-up (not this fix):** the benchmark *acceptance* still fails on perf
  (tput ~5 vs the placeholder thresholds 18.5/37) because the perf-reference targets are
  Qwen3-32B placeholders that are too aggressive for ~5 tok/s single-user — recalibrate from real
  numbers (umbrella checkbox). For `EXPERIMENTAL` this is waived; the **runtime** cap is what this
  issue fixes.
- Likely mechanism for why `--random-output-len` isn't honored: the openai-chat backend sends
  `max_completion_tokens` (which the forge server doesn't enforce) rather than `max_tokens`. The
  `--extra-body` `max_tokens` sidesteps it; a cleaner long-term fix is server-side honoring of
  `max_completion_tokens`.
