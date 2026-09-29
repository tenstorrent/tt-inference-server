# Monitoring Stack

Prometheus + Grafana Docker Compose stack for the TT Media Server.

This stack covers **both** the C++ server and the (transitional) Python
server. Same Prometheus, same Grafana, two dashboards; you select which
server to scrape at start time via `SERVER_SERVICE`. Lives at the
`tt-media-server/` top level, alongside [`telemetry/`](../telemetry)
(which is the Python instrumentation that emits `/metrics`) and
[`cpp_server/`](../cpp_server) (which contains the C++ instrumentation).

## Quick Start

The inference server must be on the shared `tt_net` Docker network so Prometheus can reach it by container name:

```bash
# One-time: create the network and attach the inference container
docker network create tt_net 2>/dev/null || true
docker network connect tt_net <your-inference-container-name>

# From the tt-media-server/ directory, start Prometheus + Grafana pointing
# at the inference container. Pick SERVER_SERVICE=cpp or python. If
# PrefillGateway is running, set GATEWAY_TARGET to its --metrics-port endpoint.
SERVER_TARGET=<your-inference-container-name>:8000 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<your-inference-container-name>:9091 \
  docker compose -f monitoring/docker-compose.yml up -d
```

If you're already inside `tt-media-server/monitoring/`, pass the file as
`./docker-compose.yml` (the leading `./` is required — a bare
`docker-compose.yml` is not resolved as a path):

```bash
SERVER_TARGET=<your-inference-container-name>:8000 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<your-inference-container-name>:9091 \
  docker compose -f ./docker-compose.yml up -d
```

`SERVER_TARGET` defaults to `tt-inference-server:8000` if omitted.
`GATEWAY_TARGET` defaults to `prefill-gateway:9091` and should point at the
PrefillGateway `--metrics-port` endpoint when the gateway is enabled.
`SERVER_SERVICE` defaults to `python` (kept for backwards compatibility
with the original setup).

### Disaggregated prefill / decode

When you run the C++ server split into a prefill node and a decode node:

```bash
# Prefill server on :8001, decode server on :8000 (defaults shown in the prompt)
TT_LOG_LEVEL=debug LLM_MODE=prefill ./build/tt_media_server_cpp -p 8001
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 TT_LOG_LEVEL=debug LLM_MODE=decode ./build/tt_media_server_cpp
```

point Prometheus at **both** servers — they expose identical metric names and
are told apart only by a `role` label that the scrape config attaches:

```bash
PREFILL_TARGET=<server-container-name>:8001 \
DECODE_TARGET=<server-container-name>:8000 \
  docker compose -f monitoring/docker-compose.yml up -d
```

`PREFILL_TARGET` defaults to `tt-prefill-server:8001` and `DECODE_TARGET` to
`tt-decode-server:8000`. Targets that aren't running are simply marked *down*
by Prometheus and emit no series, so these jobs are harmless to leave
configured in single-server mode (and vice versa — the `tt_media_server`
regular job is harmless when only prefill/decode are up).

Two dedicated dashboards are provisioned for this setup:

| Dashboard (Grafana title)                     | uid                          | filters on   | focus |
|-----------------------------------------------|------------------------------|--------------|-------|
| TT Inference Server — Prefill (disaggregated) | `tt-inference-server-prefill`| `role="prefill"` | prompt-token throughput, prefill (E2E) latency, queue depth, per-slot ISL |
| TT Inference Server — Decode (disaggregated)  | `tt-inference-server-decode` | `role="decode"`  | TSU / TPOT / ITL, decoding users, output throughput, per-slot decode perf |

> **Note (current iteration):** only the dashboards and scrape labels are
> disaggregation-aware. The metrics emitted by `metrics.cpp` are still the
> shared regular-mode set, so a handful of panels are mode-irrelevant by
> construction (e.g. TPOT on the prefill dashboard, were it shown). Tailoring
> the *exposed* metrics per mode is the next iteration.

### TTS server

A TTS deployment (`MODEL_SERVICE=tts`) is scraped by the plain
`tt_media_server` job — point `SERVER_TARGET` at it like any other single
server, and open the **TT Media Server — TTS (decode + vocoder + conditioning)**
dashboard (uid `tt-media-server-tts`):

```bash
SERVER_TARGET=<tts-container-name>:8000 SERVER_SERVICE=cpp \
  docker compose -f monitoring/docker-compose.yml up -d
```

Speech generation is a pipeline, and the dashboard measures its stages
separately so a slowdown can be pinned to one of them.

**Stage 0 — conditioning.**
`tt_tts_conditioning_seconds` is a summary (exact quantiles, 60 s window) of
time spent *preparing* a request rather than synthesizing it, labelled by
`stage`:

| `stage` | Process | Runs when |
| --- | --- | --- |
| `text_conditioning` | main | request has **no** voice sample — tokenizer lookup + prompt compilation |
| `voice_normalization` | main | request **has** a voice sample — validation, downmix to mono, resample |
| `voice_encode` | worker | voice-sample requests, encoding the reference WAV into speech IDs on device |
| `prompt_compile` | worker | voice-sample requests, prompt compilation once speech IDs exist |

The stages are named for the conditioning they build, not for the individual
steps inside them, and the two paths are disjoint: a request either has a voice
sample or it does not. `text_conditioning` is the whole text-only path — prompt
compilation included — which is why prompt compilation is broken out as its own
`prompt_compile` stage only on the voice path, where it happens later, in the
worker, and only once the speech IDs exist. So the two never co-occur on one
request, and `prompt_compile` timings are always worker-side and always
post-encode, never blended with the cheap main-process compile.

p50/p99 per stage come straight off the summary
(`tt_tts_conditioning_seconds{stage="...", quantile="0.99"}`). The headline is
its **share of engine time**:

```promql
sum(rate(tt_tts_conditioning_seconds_sum[$__rate_interval]))
  / sum(rate(tt_tts_request_duration_seconds_sum[$__rate_interval]))
```

Short utterances can be dominated by preprocessing rather than synthesis: the
fixed cost of normalizing and conditioning stops being amortized once there is
little audio to generate, and this ratio is where that shows up. Use the
`_sum` series for it — quantiles cannot be summed or averaged. Mean per-request
cost is `rate(_sum) / rate(_count)`.

Both sides have to be aggregated the same way or the division silently returns
nothing: the conditioning series carries a `stage` label that the request-duration
series does not, and Prometheus only pairs samples whose label sets match
exactly. Collapsing both to a label-less scalar with `sum()` is what makes them
divisible. Per stage, keep `stage` on the left and reduce the denominator to a
true scalar instead:

```promql
sum by (stage) (rate(tt_tts_conditioning_seconds_sum[$__rate_interval]))
  / scalar(sum(rate(tt_tts_request_duration_seconds_sum[$__rate_interval])))
```

Two things to know when reading it. **A stage that did not run is not
observed**, rather than observed as zero — so `voice_encode` goes silent on a
voice-sample cache hit instead of dragging its own quantiles toward zero, and
`rate(tt_tts_conditioning_seconds_count{stage="voice_encode"})` climbing toward
the request rate means the cache has stopped absorbing repeats. **Only requests
that reached a terminal event are counted** in either the numerator or the
denominator; a client cancellation ends a request at an arbitrary point, so it
is excluded from both rather than skewing the share.

**Warmup pollutes `text_conditioning`.** The tokenizer is cached in a
`thread_local` map (`tokenizerForPath`), and the cache fill happens *inside* the
measured window — the clock in `TtsService::generate` starts before
`prepareTask`. So the first request on each Drogon IO thread pays a full
tokenizer load from disk, and with N IO threads (`--threads`, defaulting to
`hardware_concurrency`) you get N outliers of hundreds of milliseconds against a
steady state that is sub-millisecond. Because the summary window is 60 s, those
outliers own the stage's p95/p99 for the first minute of any deployment: a
`text_conditioning` p99 of 400 ms read off a freshly started server is the
tokenizer load, not the cost of preparing a text-only request. Give the server a minute and
more requests than it has IO threads before trusting that stage's quantiles, or
warm the tokenizer per thread at startup to keep the load out of the window
entirely. The worker-side `prompt_compile` stage fills the same per-thread cache
and carries the same one-time cost on its first compile.

The worker-process stages are timed in `BlazeTtsRunner` and travel to the main
process as microsecond fields on the request's terminal audio IPC message —
which is the one message per request the main process sees exactly once, and is
therefore what lets cross-process work land in a real quantile summary rather
than in bucketed shared-memory counters.

**Stage 1 — codec-token decode.** `tt_tts_codec_tokens_total` is the cumulative
count of acoustic / codec tokens emitted by the TTS decoder, labelled by
`worker_id`, `device` (the worker's `DEVICE_IDS` group), `model_name` and
`voice_source`. Throughput is
`rate(tt_tts_codec_tokens_total[$__rate_interval])`, i.e. the autoregressive
decode capacity that has to stay ahead of playback.

**Stage 2 — vocoder / waveform reconstruction.**
`tt_tts_audio_frames_total` is the cumulative count of PCM frames (samples per
channel) the vocoder reconstructed from those tokens, labelled by `worker_id`,
`device`, `model_name` and `batch`. Both units come from that one counter, so
nothing can drift apart:

| Quantity | Query |
| --- | --- |
| samples/s | `rate(tt_tts_audio_frames_total[$__rate_interval])` |
| audio seconds/s (RTF) | `rate(tt_tts_audio_frames_total[$__rate_interval]) / scalar(max(tt_tts_audio_sample_rate_hz))` |

The second is the real-time factor: **below 1.0 the vocoder cannot keep up with
playback** even when the decoder can. `tt_tts_vocoder_chunks_total` (same
labels) pairs with the frame counter to give mean frames per chunk, which tells
apart "fewer chunks" from "shorter chunks" when audio throughput drops.

**Separating decode from vocoding.** Tokens per second of audio is a constant
of the codec,
so `rate(codec_tokens) / (rate(audio_frames) / sample_rate)` is flat while both
stages keep pace — it rises when the vocoder falls behind a healthy decoder and
falls when the decoder is starving. The two staleness clocks say the same thing
at a glance: `tt_tts_last_vocode_age_seconds` rising while
`tt_worker_last_output_age_seconds` stays flat is a vocoder stall; both rising
together points upstream at token generation.

Per-replica granularity comes from the scrape `instance` label. All of these are
published by the TTS worker process into its worker-metrics shared-memory slot
(`MetricsLayout::TTS_RUNNER`, see
`cpp_server/include/runtime/worker/tts_metrics_layout.hpp`) and rendered on the
main process's `/metrics` by `TtsWorkerMetricsRenderer`.

#### Label caveats

There is deliberately **no `voice` or `language` label**: the TTS API accepts
only `text`, a free-form `description` and an optional voice WAV, so neither
dimension exists to label by. `voice_source`
(`default` / `description` / `voice_sample`) is the bounded stand-in until
those fields are added to the request.

`batch` (`1`, `2`, `3-4`, `5-8`, `9-16`, `17+`) is **derived, not reported**.
The engine does not expose the batch its vocoder formed, so the runner counts
the distinct streams whose chunks came out of one `drainAudioOutputs()` sweep —
the engine vocodes a batch and pushes one chunk per stream in it, so a sweep
observes one batch's worth. The proxy errs in both directions. Under load a
single batch can straddle two sweeps, which shows up as two smaller buckets
rather than one larger one; conversely the runner sleeps 1 ms between sweeps, so
two batches that finish inside that gap are drained together and counted as one
larger batch. Over-counting is the flattering error — frames credited to a batch
bigger than the engine actually formed make reconstruction look like it scales
with batch size better than it does, which is the one question the label exists
to answer. Read a bucket as approximate rather than as a bound in either
direction, and trust a shift in the distribution over any single bucket. Buckets
rather than a raw count keep the label at 6 values instead of the 128 that
`PM_MAX_USERS` would admit. Replace this with the real batch size if the engine
starts reporting it.

### Docker Scrape Targets

Prometheus runs in Docker, so `localhost` inside Prometheus refers to the
Prometheus container, not the host or dev container running the server. Make
sure the server or gateway container is attached to `tt_net`, then use that
container name in `SERVER_TARGET` and `GATEWAY_TARGET`:

```bash
docker network create tt_net 2>/dev/null || true
docker network connect tt_net <server-container-name> 2>/dev/null || true

SERVER_TARGET=<server-container-name>:8001 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<server-container-name>:9091 \
  docker compose -f monitoring/docker-compose.yml up -d
```

Verify Prometheus can see all targets:

```bash
docker exec tt_prometheus wget -qO- http://localhost:9090/api/v1/targets
```

The cpp dashboard is the default Grafana home. To make the python
dashboard the default home instead, set
`GF_HOME_DASHBOARD=/etc/grafana/provisioning/dashboards/tt_media_server_python.json`.

Open Grafana at **http://localhost:3000** (admin / admin). The dashboard loads
automatically. PrefillGateway panels are available in the `TT Prefill Gateway`
dashboard.

## Video generation metrics

The Python dashboard has a **Video Generation** row driven by the
`tt_media_server_video_*` family (emitted by `telemetry/telemetry_client.py`,
recorded in `model_services/video_service.py`). The generic request metrics
answer *how long*; these answer *how much video, how fast*:

| Metric | What it tells you |
|--------|-------------------|
| `video_generation_total{request_type,status}` | throughput and outcome, split t2v / i2v; `status` is `success` / `failure` / `cancelled` |
| `video_generation_duration_seconds{resolution,status}` | end-to-end latency; filter `status="success"` for latency, `"failure"` for time-to-failure |
| `video_frames_generated_total`, `video_content_seconds_total` | fleet output rate (frames/sec, video-seconds/sec) |
| `video_denoise_steps_total` | denoise steps **executed** (success only) — Distill/Lightning run 4, AniSora 8, even when the client asked for 20 |
| `video_frames_per_second`, `video_pixels_per_second` | per-generation throughput; pixels/sec makes 480p and 720p comparable |
| `video_step_duration_seconds` | mean wall-clock seconds per **executed** denoise step |
| `video_realtime_factor` | wall seconds per second of playable video (1.0 = realtime) |
| `video_output_size_bytes`, `video_output_frames` | what the client actually got back |
| `video_requested_inference_steps`, `video_conditioning_images` | what clients are asking for |
| `video_generations_in_progress` | live concurrency, split by request type |
| `video_last_generation_timestamp` | freshness — Since Last Success is 0 while a generation is in flight, then `time() - this` once idle |
| `video_encode_*` | ffmpeg mp4 encode cost |

Three things to know when reading these:

* **`cancelled` is not `failure`.** `POST /generations/{id}/cancel` is a normal
  client action, so it gets its own `status` value and is excluded from both
  sides of the Success Rate panel. Executed steps, frames, and throughput are
  recorded for successful generations only — a request that timed out or was
  cancelled after 6s of a 300s budget did not run its steps that fast.
  Distill, Lightning, and AniSora ignore `num_inference_steps`; executed
  steps come from those pipelines, not from the request. The Success Rate
  panel is empty (not 0% or 100%) when no success/failure completed in the
  last hour.

* **Frame count and resolution come from probing the produced mp4** (PyAV, in
  `utils.video_manager.probe_video`). On a multihost `sp_runner` deployment the
  server only receives a file path from its MPI peer, so this is the only source
  of shape truth. If the probe fails, the resolution label is `unknown` and the
  shape-derived series are skipped rather than recorded as zero.

* **`video_encode_*` is absent on multihost `sp_runner` deployments.** The mp4 is
  encoded inside the external runner peer, which serves no `/metrics`. It *is*
  populated for in-process runners and the CPU postprocessing workers — including
  a small `resolution="64x64"` series from the postprocessing warmup task.

### Stage timings: denoise and VAE decode

The family above measures whole requests. These measure the two *stages* inside
one, timed in the device worker from the `denoising` and `vae` sections the
tt_dit pipelines emit on their `on_event` stream
(`telemetry/video_stage_metrics.py`, wired at the `run()` sites in
`tt_model_runners/dit_runners.py`). They answer a question the request-level
metrics cannot: **the VAE can be the limiter even when the denoise loop is
keeping the device busy.**

| Metric | Labels | What it tells you |
|--------|--------|-------------------|
| `video_vae_frames_total` | `model_type`, `device_id`, `resolution` | frames converted from latents to pixels |
| `video_vae_pixels_total` | same | pixels converted — makes 480p and 720p comparable |
| `video_vae_decode_duration_seconds` | same | how long one video's decode took, as a histogram |
| `video_denoise_duration_seconds` | same | how long the sampling loop that fed it took |

**Divide the counter by the duration, not by wall-clock time.** The two useful
queries:

```promql
# how fast the VAE is: frames (and megapixels) per second of VAE-busy time
sum(rate(tt_media_server_video_vae_frames_total[$__rate_interval]))
  / clamp_min(sum(rate(tt_media_server_video_vae_decode_duration_seconds_sum[$__rate_interval])), 1e-9)

sum(rate(tt_media_server_video_vae_pixels_total[$__rate_interval]))
  / clamp_min(sum(rate(tt_media_server_video_vae_decode_duration_seconds_sum[$__rate_interval])), 1e-9) / 1e6

# how much of the device time it is eating: VAE's share of measured stage time
sum(rate(tt_media_server_video_vae_decode_duration_seconds_sum[5m]))
  / clamp_min(
      sum(rate(tt_media_server_video_denoise_duration_seconds_sum[5m]))
        + sum(rate(tt_media_server_video_vae_decode_duration_seconds_sum[5m])),
      1e-9)
```

A bare `rate(video_vae_frames_total[…])` is **not** VAE throughput. Each
generation increments it once, after denoise *and* decode have finished, so
over a window it is the fleet's frame output rate: slow denoising makes it read
low even on an idle VAE, and an idle server makes it read zero on a fast one.
`tt_media_server_video_frames_generated_total` already reports that number.
Normalising by `_duration_seconds_sum` is what removes the load signal and
leaves the decode rate.

Use these rather than `video_step_duration_seconds` for the bottleneck
question: that one is derived from the request-level duration, which
deliberately includes queue wait, so it inflates under backlog. These are
device-side spans and do not.

Six things to know when reading these:

* **Coverage is per pipeline, and uneven by construction.** Only pipelines that
  emit a `vae` section can be measured:

  | Runner | VAE metrics |
  |--------|-------------|
  | Wan2.2 T2V / I2V / AniSora / Distill / LoRA / Lightning | yes — the I2V variants inherit `WanPipeline.__call__` |
  | Mochi-1 | yes |
  | Wan2.2 Prodia T2V / I2V | no — external `pipelines.pipeline`, takes no `on_event` |
  | LTX-2.3-distilled, MiniMax-H3 | no — no `on_event` parameter |

  A run site must not pass `on_event` to a pipeline that does not accept it;
  that is a `TypeError`, not a silently ignored kwarg.

* **One observation is a whole video, not a frame.** A 81-frame decode is a
  single `_count`. Per-frame cost is `_sum / video_vae_frames_total`, not
  `_sum / _count`.

* **Read the mean for stage time; treat the quantiles as tail-only.** At a fixed
  shape these stages are close to deterministic, so every observation lands in
  one histogram bucket and `histogram_quantile` returns that bucket's
  interpolated midpoint — a value that does not move with the data and can sit
  well above the truth. A perfectly flat p50 or p95 is reporting the bucket, not
  the device. `rate(_sum) / rate(_count)` is not bucketed and stays exact, which
  is what the decode-time panel plots. The buckets are sized so a real
  regression crosses an edge rather than being swallowed, but no bucket set
  makes a quantile exact on a narrow distribution.

* **There is no per-step denoise latency for video.** Neither video pipeline
  emits the `denoising_step_<i>` sections the image ones do, so the loop is a
  single span. The two together cover the measured device time and nothing
  else — text encoding and latent preparation are emitted but not recorded, so
  the split panel is the split between these two, not of the whole request.

* **The `vae` span is not the same work on Wan and Mochi.** Wan closes it after
  the host readback, so D2H is inside the number. Mochi closes it before
  `postprocess_video` but opens it around a `_reshape_vae()` device remesh, so
  mesh reconfiguration is inside the number and the PIL conversion is not.
  Frames per VAE-second compares fine across the two; raw decode latency does
  not — that is why the latency panel says so in its description.

* **Shape comes from probing the produced frames**, the same rule the
  request-level metrics follow, with the runner's configured resolution as the
  fallback label. If neither is readable the duration is still recorded under
  `resolution="unknown"` and the frame and pixel counters are skipped rather
  than credited a guessed zero.

* **Warmup is excluded, and only spans that closed are exported.** The recorder
  is not created while `_warming_up` is set. Every run site flushes in a
  `finally`, so a generation that dies inside the decode still reports the
  denoise loop that completed before it — while recording no decode latency and
  crediting no frames, since that span never closed. Every frame counted has a
  timed decode behind it, and the two series stay dividable.

## Directory layout

```
monitoring/
├── docker-compose.yml                        # Prometheus + Grafana + process-exporter services
├── prometheus.yml                            # scrape config (server, gateway + process metrics)
├── prometheus/rules/prefill_gateway.yml      # PrefillGateway alert rules
├── process-exporter.yml                      # which host processes to expose CPU/memory/threads for
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml        # auto-registers Prometheus datasource
    │   └── dashboards/default.yml            # tells Grafana where to load dashboards from
    └── dashboards/
        ├── tt_media_server_cpp.json          # C++ server dashboard, regular mode (latency, throughput, queue)
        ├── tt_media_server_cpp_prefill.json  # C++ disaggregated prefill node (role="prefill")
        ├── tt_media_server_cpp_decode.json   # C++ disaggregated decode node (role="decode")
        ├── tt_media_server_cpp_tts.json      # C++ TTS server (MODEL_SERVICE=tts): conditioning, codec-token + vocoder throughput
        ├── tt_media_server_python.json       # Python server dashboard (legacy, sunsetting)
        └── tt_prefill_gateway.json           # PrefillGateway routing, latency, registration-age dashboard
```

## Ports

| Service          | Port |
|------------------|------|
| Grafana          | 3000 |
| Prometheus       | 9090 |
| PrefillGateway   | 9091 by default (`--metrics-port`) |
| process-exporter | internal only (9256 on `monitoring` net) |

## Process metrics (CPU / memory / threads per binary)

`process-exporter` runs with `pid: host` and a read-only `/proc` mount so it
sees every process on the host without any change on the server side. It
groups processes by binary so you get per-server CPU/memory/threads even
when both the C++ and Python servers run on the same host.

For the C++ server, main and worker are the same binary
(`tt_media_server_cpp`); workers are distinguished by a `--worker N` argv.
[process-exporter.yml](./process-exporter.yml) splits them into two named
groups (via cmdline regex, worker rule first since first-match-wins). The
Python server (uvicorn) gets its own group:

| `groupname`                      | matches                                                  |
|----------------------------------|----------------------------------------------------------|
| `tt_media_server_cpp_worker`     | `...tt_media_server_cpp --worker N` processes            |
| `tt_media_server_cpp_main`       | main C++ server (comm `tt_media_server`, truncated)      |
| `prefill_gateway`                | PrefillGateway binary                                    |
| `tt_media_server_python`         | uvicorn / `python tt-media-server/main.py` processes     |

## PrefillGateway Alerts

Prometheus loads gateway rules from
[`prometheus/rules/prefill_gateway.yml`](./prometheus/rules/prefill_gateway.yml).
The initial rules cover stale prefill registrations, low prefix-match rate, high
prefill latency, and observed request timeouts. They appear in Prometheus and
Grafana as long as the gateway scrape target is reachable.

These are rendered by the "Infrastructure" row at the bottom of the
relevant Grafana dashboard: CPU %, memory RSS (MB), threads, open fds,
process count, and page-fault rate per binary.

Example PromQL if you want ad-hoc queries:

```promql
sum by (groupname) (rate(namedprocess_namegroup_cpu_seconds_total[1m])) * 100
sum by (groupname) (namedprocess_namegroup_memory_bytes{memtype="resident"}) / 1024 / 1024
sum by (groupname) (namedprocess_namegroup_num_threads)
```

After editing [process-exporter.yml](./process-exporter.yml) (e.g. a
renamed binary or a new worker kind), reload:

```bash
docker compose -f monitoring/docker-compose.yml restart process-exporter
```

## Stopping

```bash
docker compose -f monitoring/docker-compose.yml down
```

Add `-v` to also delete stored metrics and Grafana state.
