# Qwen3-32B release benchmark exceeds the 6h CI cap — FIXED (output not capped)

Part of the forge-TP-p300x2 umbrella. Qwen3-32B release benchmark ([tt-shield release run 5204](https://github.com/tenstorrent/tt-shield/actions/runs/27054514200)) hit the **6h job cap**. **Root cause found and fix verified locally.**

## Root cause (confirmed on a live Qwen server)
`vllm bench serve --random-output-len N` is **not honored as a `max_tokens` cap** by the forge OpenAI-compat server, so benchmark requests run **unbounded** — and **Qwen3-32B emits ~1000–1500 reasoning (`<think>`) tokens** for an osl=128 request. ~13 configs × ~1000+ tokens ≈ the 6h. Confirmed:
- Decode is **healthy** (~5 tok/s, TPOT ~185 ms) — *not* the bottleneck (see the conc sweep below).
- A direct request with explicit `max_tokens` **is** honored (probe: max_tokens=32 → 32 tokens; max_tokens=128 → 111). So it's a request-not-capped issue, not a server enforcement bug.
- `chat_template_kwargs={"enable_thinking": false}` did **not** disable Qwen3 thinking via this server.
- gemma is **not** affected — uncapped it stops much shorter (no long reasoning).

## Fix (implemented + verified)
`benchmarking/run_benchmarks.py`: add `"max_tokens": osl` to the text-task `--extra-body` (alongside the existing `truncate_prompt_tokens`). The server honors `max_tokens`, so output caps at the requested osl. Applies to all text benchmarks (a correctness improvement; only changes behavior for models that were over-generating).

**Verification (local, live Qwen server):**
| | duration | result |
|---|---|---|
| `bench serve` conc-1, n=2, **no** `max_tokens` | **>600 s (timed out)** | ~1500 tok/req |
| `bench serve` conc-1, n=2, **+`max_tokens`** | **49 s** | ~128 tok/req |
| `run.py --workflow benchmarks` smoke (conc-1, n=8) e2e **+fix** | **210 s** (8/8) | ~128 tok/req, TPOT 185 ms |

Same config was ~24–36 min/config uncapped in CI → ~3.5 min with the fix (**~8–10×**). A full ~13-config sweep now lands well under 6h.

## Concurrency sweep (decode is healthy — no conc-1/batch pathology)
| conc | per-request tok/s | aggregate tok/s |
|---|---|---|
| 1 | 5.04 | 5.04 |
| 4 | 5.03 | 20.1 |
| 16 | 5.05 | 80.8 |

## Notes
- **Acceptance verdict is *not* gated by perf (corrected):** for `EXPERIMENTAL`, `required_target_tiers` is empty, so `acceptance_criteria.py` moves **all** benchmark perf tiers (functional/complete/target) to *informational* — they're reported but don't gate. The per-stream `functional` tput target (0.1× = **3.7 tok/s**) even **passes** (actual ~5 > 3.7). The 18.5/37 numbers in the report are the `complete`/`target` tiers, informational here. The only thing that actually gates is **eval accuracy** (`r1_aime24`), and in *smoke* mode that's a ~3-sample run vs the full-set reference (80) → a spurious 0.33; a real nightly (`CI_NIGHTLY` limit) runs a meaningful sample. So: runtime is what this issue fixes; perf is informational and even passes at the EXPERIMENTAL tier; the smoke eval "failure" is a sample-size artifact.
- **Recalibration caution (re #3995):** when we replace the placeholder perf-reference with real numbers, set `theoretical.tput` to the **single-stream** value at conc=1 (≈ `tput_user`), not an aggregate — otherwise `functional.tput_check` compares conc-1 actual against an aggregate-derived threshold (exactly #3995). The current placeholder is safe (`tput == tput_user == 37`).
- **TTFT is genuinely high (~2.9 s):** the `functional` ttft target (460 ms) "fails" at 2907 ms, but it's informational. Worth a look later (prefill at 4K / dynamic batcher) when recalibrating.
- Likely mechanism for why `--random-output-len` isn't honored: the openai-chat backend sends `max_completion_tokens` (which the forge server doesn't enforce) rather than `max_tokens`. The `--extra-body` `max_tokens` sidesteps it; a cleaner long-term fix is server-side honoring of `max_completion_tokens`.
