# SDXL Perf Benchmark — methodology for issue #3479

Captures Blackhole perf numbers for the three SDXL configs Nikola asked about
(2026-05-12): trace runner, Forge UNet-only, Forge full-on-device.

## Prerequisites

- A running media server with SDXL loaded for the config under test.
- Per-stage logging in place (issue #3478 / branch `feat/sdxl-perf-logging`).
- Full-on-device path available (issue #3477 / branch `feat/sdxl-forge-full-on-device`).

## Configs to capture

| Label                  | Runner | Env                                 | Device      |
| ---------------------- | ------ | ----------------------------------- | ----------- |
| `trace`                | Metal pipeline (trace runner)       | (defaults)                          | N300 / T3K (reference upper bound) |
| `forge-unet-only`      | Forge runner | `TTXLA_SDXL_FULL_ON_DEVICE=false` (or unset) | p150x4 / p150x8 / p300x2 |
| `forge-full-on-device` | Forge runner | `TTXLA_SDXL_FULL_ON_DEVICE=true`              | p150x4 / p150x8 / p300x2 |

## Procedure (per config)

1. Start the server with the target runner + env vars. Capture the server log
   to a file (e.g. `tee /tmp/sdxl-<label>.log`).
2. Wait for warmup to complete (look for `Warmup completed` in the log).
3. Run the benchmark:

   ```bash
   python tt-media-server/scripts/sdxl_perf_benchmark.py \
       --host localhost --port 8000 \
       --config-label <label> \
       --steps 20 --warmup 1 --runs 5 \
       --output json > /tmp/sdxl-<label>.json
   ```

4. Capture per-stage timings from the server log:

   ```bash
   grep -E "Text encoding took|UNet diffusion|UNet\+VAE|VAE decode" \
       /tmp/sdxl-<label>.log \
       | tail -n +N   # skip warmup, take the last 5 runs' worth of lines
   ```

   For Forge, per-stage log lines are: `Text encoding`, `UNet diffusion (N steps)`,
   `VAE decode (full_on_device=…)`. For trace, after #3478 lands: `Text encoding`,
   `UNet+VAE inference`.

## Report format

Paste in issue #3479 as a Markdown table:

```
| Config | Device | n | Total mean (s) | TE (s) | UNet (s) | VAE (s) | Compile/warmup (s) |
|---|---|---|---|---|---|---|---|
| trace | t3k | 5 | … | … | … | … | … |
| forge-unet-only | p150x8 | 5 | … | … | … | … | … |
| forge-full-on-device | p150x8 | 5 | … | … | … | … | … |
```

Include a one-sentence recommendation: should `TTXLA_SDXL_FULL_ON_DEVICE` default
to `true` based on the numbers? Note any compile/warmup time regressions for the
full-on-device config — that hits cold-start, not steady-state inference.

## Fixed run parameters

- `prompt`: "A beautiful sunset over a mountain landscape with vibrant colors"
- `negative_prompt`: "blurry, low quality, distorted"
- `seed`: 42
- `num_inference_steps`: 20
- `guidance_scale`: 7.5
- `number_of_images`: 1
- Resolution: 1024 (default). Run a separate 512 sweep only if explicitly asked.

Keeping these fixed across all configs lets us compare like-for-like; the only
variable is the runner config / env.
