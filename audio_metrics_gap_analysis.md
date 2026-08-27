# Audio HTTP Metrics — Gap Analysis

Comparison of the implemented request-level audio metrics (`tt-media-server/telemetry/audio_metrics.py`)
against the required TTS and STT metrics tables.

**Short version:** the request-level metrics are covered (RTF, latency, durations, text length,
voice mix, language mix, streaming mix, first-partial latency and cadence for STT). The remaining
gaps are TTS intra-stream timing (blocked on a streaming TTS endpoint existing at all) and the
content-classification metrics the server cannot observe at the HTTP layer (WER, accent, noise
condition).

## TTS coverage

| Required metric | Status | Notes |
|---|---|---|
| Time to first audio | **Missing** | The `/v1/audio/speech` endpoint has no streaming mode — the response is the complete utterance, so "first playable chunk" = full request today. Measuring true TTFA means instrumenting inside `speecht5_runner` (which does chunk text internally) or adding a streaming TTS endpoint first. |
| Audio chunk cadence | **Missing** | Same root cause: no streaming TTS path exists to have cadence on. Runner-level chunk emission could be instrumented as a proxy. |
| Real-time factor | **Covered** | `tts_realtime_factor{voice}` histogram (p50/p99 via `histogram_quantile`), per utterance and per voice. |
| End-to-end utterance latency | **Covered** | `tts_request_duration_seconds`. "By utterance length" cross-dimension isn't labeled — you can correlate via the separate character histogram but not slice latency by length. |
| Text length | **Partial** | Characters: yes (`tts_input_characters_per_request` + total). **Words and tokens: no.** |
| Utterance duration | **Covered** | `tts_output_audio_duration_seconds` + `tts_output_audio_seconds_total{voice}` (produced audio; "requested" duration isn't a concept in this API). |
| Voice mix | **Covered** | `voice` label on `tts_requests_total` and `tts_output_audio_seconds_total`: the speaker id the runner reports (falling back to the requested id), `custom` for raw client embeddings, `default` otherwise; truncated to 64 chars. Cardinality = the deployment's speaker catalog. |
| Streaming mix | **N/A today** | All TTS requests are batch; there's no streaming label because there's nothing to mix. Worth adding the label pre-emptively if streaming TTS is on the roadmap. |

## STT coverage

| Required metric | Status | Notes |
|---|---|---|
| First partial latency | **Covered** | `stt_first_partial_seconds`: request arrival at the handler until the first transcript update is emitted to the client. The pre-existing `audio_chunk_first_token_seconds` still gives the per-chunk view inside the runner. |
| Partial-result cadence | **Covered** | `stt_partial_interval_seconds`: wall time between successive transcript updates, as the client experiences it (a slow consumer shows up by design). Note "per stream" granularity isn't something Prometheus gives you — histograms aggregate across streams; true per-stream needs traces/logs. |
| Finalization latency | **Missing** | Time from end-of-input to final transcript. For file uploads this ≈ total duration (already covered); for streaming it needs an end-of-audio marker the current protocol doesn't expose. |
| Real-time factor | **Covered** | `stt_realtime_factor` per request; time-window RTF is also derivable as `rate(request_duration_sum) / rate(input_audio_seconds_total)`. |
| Word error rate | **Out of scope for live metrics** | WER needs reference transcripts, so it can't be an HTTP metric. The repo already computes it offline in `test_module/eval_tests/whisper_eval_test.py` — that's the right home. |
| Audio duration | **Covered** | `stt_input_audio_duration_seconds` + total. |
| Chunk size | **Partial (pre-existing)** | Existing `audio_chunks_per_request` + the new input-seconds total give average chunk size; per-chunk size distribution and the configured value aren't exported. |
| Sample rate / channel count | **Covered (pre-existing)** | Already labels on `audio_feature_extraction_input_seconds_total{sample_rate, channels}` — not in the new module, but on `/metrics` today. |
| Language / accent mix | **Partial** | `language` label (from `settings.audio_language`, one value per deployment — same source the encoder metrics use) on all STT request and usage metrics, so language mix by requests and audio seconds works across deployments. Accent isn't detectable at all. |
| Noise condition | **Missing** | Requires acoustic classification (SNR/reverb estimation) in preprocessing — a new capability, not just a metric. The VAD pipeline would be the natural hook. |
| Streaming mix | **Covered** | `streaming` label on the request counter and on all usage metrics (`stt_input_audio_seconds_total`, `stt_output_characters_total`), so the mix is computable by requests, audio seconds, and characters. |

## Extras added that aren't in the required list

- **Success/error split** (`status` label on both request counters and duration histograms) — error rates and time-to-failure.
- **STT output characters** (`stt_output_characters_total`) — transcript volume produced.
- **TTS response-format mix** (`response_format` label: wav/mp3/ogg/json).
- **Task label** on STT (transcription vs. translation mix).

## Remaining gaps (all blocked on more than a metric)

1. **TTS time-to-first-audio and chunk cadence** — needs a streaming TTS endpoint (or runner-level instrumentation in `speecht5_runner` as a proxy).
2. **STT finalization latency** — needs an end-of-audio marker in the streaming protocol.
3. **Word error rate** — offline eval concern; already computed in `test_module/eval_tests/`.
4. **Noise condition / accent** — needs acoustic classification in preprocessing.
5. **TTS text length in words/tokens** — needs a tokenizer choice; characters are exported today.
