# Verifying TTS Metrics on `/metrics`

Manual procedure for checking that the TTS codec-token, vocoder and
conditioning instrumentation actually reaches the server's `/metrics`
endpoint. What each series means and how to query it lives in
[`monitoring/README.md`](../../monitoring/README.md); this document only covers
producing and reading them on a running server.

There are two lanes. Run the mock lane first regardless of hardware
availability: `MockTtsScheduler` emits token and audio outputs shaped like the
real engine, so every metric below is exercised through the same
`SingleProcessWorkerMetrics` calls, the same shared-memory cells and the same
`TtsWorkerMetricsRenderer`. The only thing it does not validate is the real
engine's call pattern.

## Prerequisite: a TTS tokenizer

Both lanes need `TTS_TOKENIZER_PATH` pointing at a `tokenizer.json` whose vocab
contains the speech tokens `validateRequiredTokens` demands (`<|speech_start|>`,
`<|speech_end|>`, `<|s_0|>`, `<|s_1|>`, the audio/voice prompt delimiters and
`<|bot|>`). A stock LLM tokenizer does not have them, and the default —
`tokenizerPath(ModelType::LLAMA_3_1_8B_INSTRUCT)` — resolves to an empty string
unless that model directory exists under `cpp_server/tokenizers/`. Without it,
every request that compiles a prompt fails with:

```
{"error":{"message":"TTS tokenizer path is empty; set TTS_TOKENIZER_PATH", ...}}
```

Keep `tokenizer_config.json` in the same directory as the `tokenizer.json`:
`ttsEngineConfig()` reads `bos_token` / `add_bos_token` from it to build the
prompt-leading BOS, and logs a "TTS prompt has no leading BOS" warning if it
cannot.

[Section 4a](#4a-partial-pass-without-a-tokenizer) covers what can still be
verified when no tokenizer is available.

### Synthetic tokenizer for the mock lane

The mock scheduler never reads the prompt — `compilePromptTokens` only has to
return a non-empty vector for `allocateTask` to reach SUBMIT. That makes a
throwaway fixture enough to unblock every counter on the mock lane. Build one
with the Python `tokenizers` package (same Rust core that `tokenizers-cpp`
wraps), as a byte-level BPE with no merges plus the nine required tokens and a
pre-seeded `<|s_N|>` range so voice-clone prompts stay atomic:

```python
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
for token in ["<|speech_start|>", "<|speech_end|>", "<|audio_prompt_start|>",
              "<|audio_prompt_end|>", "<|voice_prompt_start|>",
              "<|voice_prompt_end|>", "<|bot|>"] + [f"<|s_{i}|>" for i in range(1024)]:
    vocab.setdefault(token, len(vocab))

tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.add_special_tokens([AddedToken(t, normalized=False, special=True)
                        for t in vocab if t.startswith("<|")])
tok.save("/tmp/tt_tts_tokenizer/tokenizer.json")
```

Keeping the specials in the model vocab (rather than only in `added_tokens`)
matters: `getEncodedVocab()` walks ids `0..GetVocabSize()` through
`IdToToken`, and `validateRequiredTokens` searches that list by content.

Write it under `/tmp`, never into `cpp_server/tokenizers/` — prompts compiled
with it are meaningless to a real model, and it must not be mistaken for the
real SpeechLM tokenizer. Omit `tokenizer_config.json` for the fixture; the
missing BOS only produces a warning. Swapping in the real tokenizer later is a
one-variable change.

## 1. Build

```bash
cd tt-media-server/cpp_server
env -u TT_METAL_HOME ./build.sh --blaze
```

Clear stale IPC state before every run. Worker shared memory is
kernel-persistent and survives `SIGKILL`, so a leftover segment makes a fresh
worker attach to a corrupt queue and hang:

```bash
rm -f /dev/shm/tt_*     # lowercase only — TT_UMD_LOCK.* belongs to tt-metal
```

## 2. Start the server

### Mock backend (no hardware)

```bash
MODEL_SERVICE=tts \
MODEL_RUNNER_TYPE=mock_tts \
DEVICE_IDS='(0)' \
TT_LOG_LEVEL=debug \
./build/tt_media_server_cpp -p 8000 2>&1 | tee /tmp/tts_metrics_run.log
```

`DEVICE_IDS` sets the worker count — one parenthesized group per worker, so
`'(0),(1)'` gives two workers and is how you check that the `worker_id` label
splits. Set `TT_WORKER_METRICS_SHM` to a unique name if another server on the
host might use the default segment.

### Real engine

```bash
export TT_METAL_HOME=$PWD/tt-llm-engine/tt-metal
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$PWD/tt-llm-engine/build-full:$LD_LIBRARY_PATH

MODEL_SERVICE=tts \
MODEL_RUNNER_TYPE=tt_tts \
DEVICE_IDS='(0,1,4,5,24,25,28,29)' \
LD_PRELOAD=$PWD/tt-llm-engine/build-full/libtt_llm_engine.so.0 \
TT_LOG_LEVEL=info \
./build/tt_media_server_cpp -p 8000
```

Two preconditions, both worth checking before spending device time:

1. The binary must link `TtLlmEngine::Full`. `makeTtsScheduler()` builds the
   real scheduler only under `TT_MEDIA_SERVER_TTS_FULL_PIPELINES`, which CMake
   defines only when that target exists; otherwise startup throws
   *"TTS scheduler is not linked with socket-capable TtLlmEngine::Full"*.
2. The model launcher must have published its socket descriptors —
   `tts2_encoder`, `tts2_speechlm`, `tts2_decoder` by default. Check with
   `ls /dev/shm | grep tts2`; if they are absent the scheduler sits in its
   connect timeout.

### Confirm the worker attached

```bash
grep 'SingleProcessWorkerMetrics] Worker 0 attached to shm' /tmp/tts_metrics_run.log
```

A `failed to attach to shm` line instead means metrics are disabled for that
worker and every series will read zero — a startup problem, not a metrics bug.

## 3. Scrape once before sending traffic

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'age_seconds|sample_rate_hz'
```

This is the seeding check. The three age gauges must start near zero, not near
1.7 billion. `SCRATCH_LAST_VOCODE_EPOCH_MS` in particular is its own cell and
needs its own seed in `SingleProcessWorkerMetrics::initialize()`; an unseeded
cell makes `ageSeconds` report time since the Unix epoch.

## 4. Generate traffic

`/metrics` is exempt from the bearer filter, `/v1/audio/speech` is not. The key
defaults to `your-secret-key` unless `OPENAI_API_KEY` is set.

Text-only — `voice_source="default"`:

```bash
for i in $(seq 1 5); do
  curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Authorization: Bearer your-secret-key" \
    -H "Content-Type: application/json" \
    -d '{"text":"the quick brown fox jumps over the lazy dog"}' \
    -o /tmp/speech_$i.wav
done
```

With a description — `voice_source="description"`:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text":"calm narration check","description":"a calm low voice"}' \
  -o /tmp/speech_desc.wav
```

With a voice sample — `voice_source="voice_sample"` plus the
`voice_normalization` / `voice_encode` / `prompt_compile` conditioning stages.
Send it twice: the second request is a cache hit, where `voice_encode` must
stop being observed while `prompt_compile` keeps being observed.

```bash
python3 -c "
import wave, math, struct
w = wave.open('/tmp/voice.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
w.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(i/12.0))) for i in range(24000)))
w.close()"

for i in 1 2; do
  curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Authorization: Bearer your-secret-key" \
    -F 'text=cloned voice check' \
    -F 'file=@/tmp/voice.wav' -o /tmp/speech_voice_$i.wav
done
```

## 4a. Partial pass without a tokenizer

The voice-sample requests above are worth sending even with no tokenizer on
disk. That path normalizes PCM in the main process and encodes the reference
WAV in the worker before it ever compiles a prompt, so three series still move
over real HTTP:

| Series | After two identical voice-sample requests |
| --- | --- |
| `tt_tts_conditioning_seconds{stage="voice_normalization"}` | `_count` = 2 |
| `tt_tts_conditioning_seconds{stage="voice_encode"}` | `_count` = 1 — silent on the cache hit |
| `tt_tts_request_duration_seconds` | `_count` = 2 |

That covers the main-process conditioning half end to end, including the
cross-process `voice_encode` timing that travels back on the terminal IPC
message and the "a stage that did not run is not observed" rule that the
cache-hit case exists to prove.

It cannot reach the synthesis half. `prompt_compile` throws before its timing is
recorded, `text_conditioning` covers a path that cannot be entered at all, and
the worker shared-memory counters need audio that is never produced.

Two things to expect while reading this. The failing request returns **HTTP 200
with a 44-byte header-only WAV**, because the streaming writer commits the
response before the worker reports the error — payload size, not status code,
tells success from failure on this endpoint. And the error still counts as a
terminal event, so `tt_tts_request_duration_seconds_count` advances for requests
that produced no audio at all.

Everything in the synthesis half is meanwhile covered by `ctest`, which needs no
tokenizer and no hardware:

```bash
cd build && ctest -R 'Tts|TTS' --output-on-failure
```

## 5. Read `/metrics`

```bash
curl -s http://127.0.0.1:8000/metrics > /tmp/metrics.txt
grep -E '^tt_tts_|^tt_worker_' /tmp/metrics.txt
```

The worker families are rendered lazily — `MetricsController` calls
`WorkerMetricsAggregator::refresh()` on every scrape — so a stale read is not
possible, but an empty one means the aggregator was never initialized for the
TTS layout.

| Series | Labels | Expected after the five text requests |
| --- | --- | --- |
| `tt_tts_codec_tokens_total` | `worker_id`, `device`, `model_name`, `voice_source` | 3 series; `default` = 125 |
| `tt_tts_audio_frames_total` | `worker_id`, `device`, `model_name`, `batch` | 6 series; sum = 14400 |
| `tt_tts_vocoder_chunks_total` | `worker_id`, `device`, `model_name`, `batch` | 6 series; sum = 15 |
| `tt_tts_audio_sample_rate_hz` | `worker_id` | 48000 |
| `tt_tts_last_vocode_age_seconds` | `worker_id` | small, rises once traffic stops |
| `tt_worker_last_output_age_seconds` | `worker_id` | small |
| `tt_worker_heartbeat_age_seconds` | `worker_id` | small |
| `tt_tts_conditioning_seconds` | `model`, `stage` | 4 stage series, each with `_sum` / `_count` / quantiles |
| `tt_tts_request_duration_seconds` | `model` | `_count` = finished requests |

The expected column assumes the mock backend and one worker. Its arithmetic is
25 codec tokens per request (three chunks of eight, plus the terminal token
that carries `is_complete` on a real token rather than a synthetic sentinel),
2880 PCM frames (three chunks of 960 samples at one channel) and 3 chunks.

With one request in flight at a time everything lands in `batch="1"`. Under
concurrency the distribution should shift right, but a single engine batch can
straddle two 1 ms drain sweeps and land as two smaller buckets, so read a shift
in the distribution rather than any single bucket.

## 6. Readings that look plausible but are wrong

| Reading | What it means |
| --- | --- |
| An age gauge reads ~1.7e9 | That cell was never seeded, so `ageSeconds` is measuring since the Unix epoch |
| Every series is zero while `tt_worker_alive` is 1 | The worker did not attach to shared memory, or its id fell outside the slot range — the critical log line at startup says which |
| Codec tokens come out at 24N rather than 25N | The terminal token stopped being counted |
| No `tt_tts_*` families at all | The aggregator was not initialized, or `TtsWorkerMetricsRenderer` was not registered for `MetricsLayout::TTS_RUNNER` in `main()` |
| `voice_encode` still observed on the second identical WAV | The voice-sample cache did not hit; a stage that did not run must be absent, not observed as zero |
| Frames climb while chunks stay flat | Per-sweep chunk accounting in `drainAudioOutputs()` is off, making mean frames per chunk meaningless |

On hardware the numbers additionally have to be plausible: the real-time factor
`rate(tt_tts_audio_frames_total[1m]) / 48000` should exceed 1.0, and
`rate(codec_tokens) / (rate(audio_frames) / 48000)` should sit flat at the
codec's tokens-per-audio-second constant. If that ratio drifts under steady
load, one of the two counters is credited at the wrong point.

## 7. Dashboard pass (optional)

A panel querying a label that does not exist renders empty and looks exactly
like a metrics bug, so the Grafana dashboard is worth checking separately once
`/metrics` reads correctly:

```bash
cd tt-media-server
SERVER_TARGET=<container-or-host>:8000 SERVER_SERVICE=cpp \
  docker compose -f monitoring/docker-compose.yml up -d
```

Open the `tt-media-server-tts` dashboard at <http://localhost:3000>. Prometheus
runs in Docker, so the target must be reachable by container name on `tt_net`.
Give the server a minute of traffic before judging the `text_conditioning`
quantiles — the tokenizer cache fills inside the measured window, once per
Drogon IO thread, so the first minute's p99 on that stage is a disk load rather
than a conditioning cost.

### Without a Docker daemon

Reservation shells run in an unprivileged container with no `docker.sock`, so
compose is unavailable there. Run both as plain binaries against the same
provisioning tree:

```bash
mkdir -p /tmp/tt_monitoring && cd /tmp/tt_monitoring
curl -sSLO https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
curl -sSL https://dl.grafana.com/oss/release/grafana-10.4.2.linux-amd64.tar.gz -o grafana.tar.gz
tar xzf prometheus-2.51.0.linux-amd64.tar.gz && tar xzf grafana.tar.gz

# prometheus.yml: one static target at 127.0.0.1:8000 carrying the labels the
# dashboard expects — job=tt_media_server, service=cpp, role=regular,
# language=cpp.
./prometheus-2.51.0.linux-amd64/prometheus --config.file=prometheus.yml \
  --storage.tsdb.path=prom-data --web.listen-address=127.0.0.1:9090 &

GF_PATHS_PROVISIONING=<repo>/tt-media-server/monitoring/grafana/provisioning \
  ./grafana-v10.4.2/bin/grafana server --homepath $PWD/grafana-v10.4.2 &
```

Point the provisioned datasource at `http://127.0.0.1:9090` instead of the
compose service name, and reach Grafana over an SSH tunnel. Leave the dashboard
template variables on `All`: with one worker and one device, a panel that looks
empty is usually a selected variable value rather than a missing series.

## 8. Teardown

```bash
pkill -f tt_media_server_cpp
rm -f /dev/shm/tt_*
```
