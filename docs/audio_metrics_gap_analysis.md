# Audio HTTP Metrics — Gap Analysis

Comparison of the implemented request-level audio metrics (`tt-media-server/telemetry/audio_metrics.py`)
against the required TTS and STT metrics tables.

**Short version:** the request-level metrics are covered (RTF, latency, durations, text length,
voice mix, language mix, streaming mix, first-partial latency and cadence for STT), and the
metrics the server cannot observe directly have exported proxies: TTS time-to-first-audio and
chunk cadence (runner-level mel generation timings), STT finalization (last-partial→final wait),
and WER / noise condition (Whisper's own confidence signals as drift and difficulty proxies).
What remains blocked needs more than a metric: the client-observed TTS streaming timings need a
streaming endpoint, true finalization needs an end-of-audio protocol marker, true WER needs
reference transcripts (offline evals), and noise/accent attribution needs acoustic classification.

## TTS coverage

| Required metric | Status | Notes |
|---|---|---|
| Time to first audio | **Covered (proxy)** | `tts_first_chunk_seconds`: time until the first text chunk's mel spectrogram completes inside `speecht5_runner` — a lower bound on achievable streaming TTFA (a streaming path would add one per-chunk vocoder pass). Client-observed TTFA today ≡ total request duration, because the endpoint has no streaming mode. |
| Audio chunk cadence | **Covered (proxy)** | `tts_chunk_generation_seconds`: per-chunk mel generation time — the achievable streaming cadence. True client-side cadence needs a streaming TTS endpoint to exist first. |
| Real-time factor | **Covered** | `tts_realtime_factor{voice}` histogram (p50/p99 via `histogram_quantile`), per utterance and per voice. |
| End-to-end utterance latency | **Covered** | `tts_request_duration_seconds`. "By utterance length" cross-dimension isn't labeled — you can correlate via the separate character histogram but not slice latency by length. |
| Text length | **Covered** | Characters and whitespace words per request + totals (handler); exact encoder input tokens per generation + total (`tts_input_tokens_*`, recorded in the runner, which already tokenizes every chunk — the model's true input size, not an approximation). |
| Utterance duration | **Covered** | `tts_output_audio_duration_seconds` + `tts_output_audio_seconds_total{voice}` (produced audio; "requested" duration isn't a concept in this API). |
| Language mix | **N/A today** | Nothing varies: no language/locale field on the request, the model reports none, text normalization is hardcoded English, and SpeechT5 is English-only. Becomes doable (and easy — the STT label pattern applies) with a multilingual model or a request-level language field. |
| Voice mix | **Covered** | `voice` label on `tts_requests_total` and `tts_output_audio_seconds_total`: the speaker id the runner reports (falling back to the requested id), `custom` for raw client embeddings, `default` otherwise; truncated to 64 chars. Cardinality = the deployment's speaker catalog. |
| Speaker conditioning | **Covered (share) / N/A today (size)** | Share of conditioning types is the `voice` label (`custom` = user-supplied embedding, catalog id, or `default`) — redundant with voice mix. Size ("reference seconds or tokens") is degenerate: SpeechT5 conditioning is always a fixed 512-dim x-vector, never reference audio or style prompts; becomes meaningful with a voice-cloning model. |
| Streaming mix | **N/A today** | All TTS requests are batch (no `stream` field, complete-utterance responses), so a `streaming` label would have exactly one value. The runner already speaks the internal streaming protocol and generates mel chunks incrementally — the endpoint is the missing piece. |

The N/A-today rows are contingent on the current model and endpoint, not intrinsically
unmeasurable — re-check them whenever a new TTS model is onboarded or a streaming TTS
endpoint lands.

## STT coverage

| Required metric | Status | Notes |
|---|---|---|
| First partial latency | **Covered** | `stt_first_partial_seconds`: request arrival at the handler until the first transcript update is emitted to the client. The pre-existing `audio_chunk_first_token_seconds` still gives the per-chunk view inside the runner. |
| Partial-result cadence | **Covered** | `stt_partial_interval_seconds`: wall time between successive transcript updates, as the client experiences it (a slow consumer shows up by design). Note "per stream" granularity isn't something Prometheus gives you — histograms aggregate across streams; true per-stream needs traces/logs. |
| Finalization latency | **Covered (proxy)** | `stt_finalization_seconds`: the wait between the last partial and the final transcript (segment assembly). The true metric — end-of-*speech* to final — needs live audio input with an end-of-audio marker, which the protocol doesn't have; for file uploads end-of-input = request arrival, so it collapses into total request duration (covered). |
| Real-time factor | **Covered** | `stt_realtime_factor` per request; time-window RTF is also derivable as `rate(request_duration_sum) / rate(input_audio_seconds_total)`. |
| Word error rate | **Proxy exported; true WER out of scope** | WER needs reference transcripts, so it can't be an HTTP metric — offline evals in `test_module/eval_tests/whisper_eval_test.py` stay the source of truth. Live drift proxies now exported from the whisper runner, once per generation: `stt_avg_logprob`, `stt_no_speech_probability`, `stt_compression_ratio` (the same signals Whisper's temperature-fallback gates on). Optional next step: canary WER on known audio via the existing `CanaryMonitor`. |
| Audio duration | **Covered** | `stt_input_audio_duration_seconds` + total. |
| Chunk size | **Partial (pre-existing)** | Existing `audio_chunks_per_request` + the new input-seconds total give average chunk size; per-chunk size distribution and the configured value aren't exported. |
| Sample rate / channel count | **Covered (pre-existing)** | Already labels on `audio_feature_extraction_input_seconds_total{sample_rate, channels}` — not in the new module, but on `/metrics` today. |
| Language / accent mix | **Partial** | `language` label (from `settings.audio_language`, one value per deployment — same source the encoder metrics use) on all STT request and usage metrics, so language mix by requests and audio seconds works across deployments. Accent isn't detectable at all. |
| Noise condition | **Partial (proxy)** | The exported confidence signals (`stt_avg_logprob` down, `stt_no_speech_probability` up) track acoustic *difficulty*, of which noise is one cause — but they're cause-agnostic (accent and domain mismatch move them too). A truly acoustic, model-independent measure (energy-based SNR from the VAD's speech/non-speech split) is the cheap next step; attributing conditions (reverberant, far-field, overlapping) needs a classifier in preprocessing. |
| Streaming mix | **Covered** | `streaming` label on the request counter and on all usage metrics (`stt_input_audio_seconds_total`, `stt_output_characters_total`), so the mix is computable by requests, audio seconds, and characters. |

## Extras added that aren't in the required list

- **Success/error split** (`status` label on both request counters and duration histograms) — error rates and time-to-failure.
- **STT output characters** (`stt_output_characters_total`) — transcript volume produced.
- **TTS response-format mix** (`response_format` label: wav/mp3/ogg/json).
- **Task label** on STT (transcription vs. translation mix).

## Remaining gaps (all blocked on more than a metric)

1. **True TTS time-to-first-audio and chunk cadence** — the runner-level mel-generation proxies are exported; the client-observed metrics need a streaming TTS endpoint (which would also make a TTS `streaming` mix label meaningful).
2. **True STT finalization latency** — the last-partial→final proxy is exported; the end-of-speech-anchored measurement needs live audio input with an end-of-audio marker.
3. **True word error rate** — confidence-signal drift proxies are exported; real WER stays in offline evals (`test_module/eval_tests/`), optionally plus a canary WER probe on known audio.
4. **Noise-condition attribution / accent** — difficulty proxies are exported; attributing the cause needs an acoustic SNR estimate (cheap, via VAD energies) or a classifier in preprocessing.
5. **Per-request language mix** — on STT the label plumbing exists (per-deployment value today), but non-trivial values need the API to accept a `language` field *and* the runner to honor it, otherwise the label would report what clients asked for rather than what the model did. On TTS there is nothing to label at all until a multilingual model or a request-level language field exists.
6. **TTS speaker-conditioning size (reference seconds/tokens)** — the share half is covered by the `voice` label; the size half is degenerate while conditioning is a fixed 512-dim x-vector, and becomes measurable only if a voice-cloning model (reference audio / style prompts) lands.

