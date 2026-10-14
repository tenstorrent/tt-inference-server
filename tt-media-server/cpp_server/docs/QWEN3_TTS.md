# Qwen3-TTS serving

`MODEL_SERVICE=tts MODEL_RUNNER_TYPE=tt_qwen3_tts` serves
[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) on
`POST /v1/audio/speech`. Each worker process embeds a Python interpreter and
drives tt-metal's `models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline` on one
chip, one utterance at a time. The response is a streamed mono PCM16 WAV at
24 kHz.

## One release per deployment

Qwen3-TTS ships as three releases that choose the voice differently. A
deployment serves exactly one, chosen by its checkpoint, never by the request:

| release (`TTS_QWEN3_RELEASE`) | hub ids | voice comes from |
|---|---|---|
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | a WAV clip to clone |
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | one of nine built-in speakers |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (there is no 0.6B) | a sentence describing the voice |

The server resolves the release and size before any worker starts. For each,
the first source that answers wins:

1. `TTS_QWEN3_RELEASE` / `TTS_QWEN3_MODEL_SIZE`;
2. the checkpoint's `config.json` (`tts_model_type` / `tts_model_size`), if it
   is on local disk: in `QWEN3_TTS_CKPT`, in `HF_MODEL` when that is a
   directory, or in the HF hub cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else
   `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`);
3. the `HF_MODEL` hub id's name, or `Qwen/Qwen3-TTS-12Hz-1.7B-Base` when neither
   `HF_MODEL` nor `QWEN3_TTS_CKPT` is set.

Startup fails when `TTS_QWEN3_RELEASE` or `TTS_QWEN3_MODEL_SIZE` contradicts a
local `config.json`, when no source names the release or size (for example
`QWEN3_TTS_CKPT` without a `config.json` and no `TTS_QWEN3_*`), and for 0.6B
VoiceDesign. At warmup each worker loads the checkpoint and refuses to serve if
it is not the configured release and size.

Knowing the release lets the server reject a request it cannot serve with an
HTTP 400 before streaming starts. Once the 200 and the WAV header have gone out,
a failure can only end the stream early (see [Limitations](#limitations)).

## Request fields

JSON (`Content-Type: application/json`) or `multipart/form-data`. The voice clip
can only travel as a multipart file part: PCM16 WAV, any rate, any channel
count; the server downmixes and resamples it to 24 kHz mono. The file part's
field name does not matter; the first file in the form is the clip.

| field | meaning for Qwen3-TTS |
|---|---|
| `text` | required, not blank; what to say |
| `description` | VoiceDesign: the voice description (required). CustomVoice: an optional delivery instruction (1.7B only). Base: an optional instruction for a voice-only clone (1.7B only) |
| `speaker` | CustomVoice only, required: a built-in speaker name (case-insensitive) |
| `language` | optional: `Auto` or a language the checkpoint lists (case-insensitive) |
| `reference_text` | Base only, optional: the clip's transcript |
| WAV file part | Base only, required: the voice to clone |

What each release accepts (anything else is a 400 that names the release):

| release | required | rejected | `language` default |
|---|---|---|---|
| `custom_voice` | `speaker` | WAV, `reference_text`; `description` on 0.6B | `English` |
| `voice_design` | `description` | `speaker`, WAV, `reference_text` | `Auto` |
| `base` | WAV | `speaker`; `description` together with `reference_text` or on 0.6B; a clip longer than `TTS_QWEN3_MAX_REFERENCE_SECONDS` | `Auto` |

Base clones in one of two ways. With `reference_text` the clone is in context:
closer to the voice, but the transcript must match the whole clip. Without it
the clone uses the voice alone (speaker embedding only). 3 to 10 s of clean
speech is the useful clip length.

Fields that are present but blank count as absent. Speakers and languages are
checked against the checkpoint's `config.json` once the server can read it. For
a hub checkpoint that is not downloaded yet, the first worker downloads it and
the server rereads the lists at most every 10 s until it finds them; until then
an unknown speaker or language is rejected by the worker, after the 200.

Other responses: 429 when `min(TTS_MAX_USERS, TTS_TASK_QUEUE_CAPACITY)`
requests are already accepted; 503 until a worker has finished warmup.

## Environment

| variable | meaning | default |
|---|---|---|
| `MODEL_SERVICE` | `tts` | |
| `MODEL_RUNNER_TYPE` | `tt_qwen3_tts` | `tt_tts` |
| `DEVICE_IDS` | one worker per group, e.g. `(0),(1)`; see [Limitations](#limitations) for multi-chip groups | `(0)` |
| `HF_MODEL` | hub id or local directory of the checkpoint | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `QWEN3_TTS_CKPT` | local checkpoint directory; wins over `HF_MODEL` | |
| `TTS_QWEN3_RELEASE` | `base`, `custom_voice` or `voice_design` | resolved, see above |
| `TTS_QWEN3_MODEL_SIZE` | `1b7` or `0b6` (`1.7b` / `0.6b` also accepted) | resolved, see above |
| `TTS_QWEN3_MAX_FRAMES` | frame budget per utterance at 12.5 frames/s; sizes the KV cache; at least 1 | `400` (32 s) |
| `TTS_QWEN3_SEED` | non-negative integer; reseeds sampling before every request, so the same request gives the same audio | unset (unseeded) |
| `TTS_QWEN3_MAX_REFERENCE_SECONDS` | longest clip accepted for cloning, 1 to 40 | `30` |
| `TTS_MAX_USERS` | requests accepted at once (running plus queued); more get 429 | number of workers |
| `TTS_TASK_QUEUE_CAPACITY` | task queue slots (2 MiB each in `/dev/shm`); also caps accepted requests | `1000` |
| `TTS_AUDIO_QUEUE_CAPACITY` | audio queue slots per worker (512 KiB each) | `1024` |
| `TTS_CANCEL_QUEUE_CAPACITY` | cancel queue slots per worker | `1024` |
| `TT_METAL_HOME` | tt-metal checkout: `models.demos` is imported from here, kernels compile here | required |
| `TT_PYTHON_PATH` | extra directory put first on the embedded interpreter's `sys.path` | unset |

Startup also fails if `TTS_AUDIO_SAMPLE_RATE_HZ` or `TTS_VOICE_SAMPLE_RATE_HZ`
is set to anything but 24000, or `TTS_AUDIO_CHANNELS` / `TTS_VOICE_CHANNELS` to
anything but 1. `TTS_MAX_BATCH_SIZE` is ignored with a warning (batch 1).

`TTS_TOKENIZER_PATH` is not used. The TTS-2 tokenizer lookup still runs at
startup, so the log may show `TTS BOS lookup failed` and
`TTS prompt has no leading BOS` warnings; both are harmless for Qwen3-TTS.

Each worker sets `TT_VISIBLE_DEVICES` to its `DEVICE_IDS` group, uses
`$TT_METAL_HOME/built/<group>` (commas as underscores) as `TT_METAL_CACHE`, runs
with `TT_METAL_HOME` as its working directory, and defaults `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` to 2.

Python: the embedded interpreter must resolve the tt-metal venv that matches
`TT_METAL_HOME` (`ttnn`, `torch`, `librosa`, `soundfile`, `safetensors`,
`huggingface_hub`). Launch with that venv activated so its `python3` is first on
`PATH`, and scrub any ambient `PYTHONPATH` that points at another tt-metal
checkout; [`embedding_serving_guide.md`](embedding_serving_guide.md) explains
both traps. Hub checkpoints (codec included) download on first use into the HF
cache, a few GB each.

### Shared memory

The TTS queues live in `/dev/shm` under fixed names (`tt_tts_task_queue`,
`tt_tts_audio_queue_<worker>`, `tt_tts_cancel_queue_<worker>`). With the default
capacities they reserve about 2.5 GB for one worker (2 GB of task queue plus
0.5 GB of audio queue per worker). On a host with a small `/dev/shm` (4 GB, for
example) startup then crashes with a Boost `interprocess_exception`
(`library_error`). Qwen3-TTS runs batch 1, so small queues are enough: the
launch recipes below and the docker workflow use 16 / 64 / 64.

The names are fixed, so run one TTS server per host. Queue files left by a
crashed server are normally replaced at startup; one the server cannot remove
(owned by another user, for example) breaks startup. Delete them only when no
server is running: `rm -f /dev/shm/tt_tts_*`. Never delete `tt_device_*`.

## Launch

```bash
source /path/to/tt-metal/python_env/bin/activate   # the venv matching TT_METAL_HOME
export TTS_TASK_QUEUE_CAPACITY=16 TTS_AUDIO_QUEUE_CAPACITY=64 TTS_CANCEL_QUEUE_CAPACITY=64

# CustomVoice 1.7B, one worker per chip on a two-chip host
env PYTHONPATH= \
  MODEL_SERVICE=tts MODEL_RUNNER_TYPE=tt_qwen3_tts \
  DEVICE_IDS='(0),(1)' \
  HF_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  TT_METAL_HOME=/path/to/tt-metal \
  ./build/tt_media_server_cpp -p 8000

# Base 0.6B (voice cloning) from a local checkpoint directory
env PYTHONPATH= \
  MODEL_SERVICE=tts MODEL_RUNNER_TYPE=tt_qwen3_tts \
  DEVICE_IDS='(0)' \
  QWEN3_TTS_CKPT=/models/Qwen3-TTS-12Hz-0.6B-Base \
  TT_METAL_HOME=/path/to/tt-metal \
  ./build/tt_media_server_cpp -p 8000

# VoiceDesign 1.7B
env PYTHONPATH= \
  MODEL_SERVICE=tts MODEL_RUNNER_TYPE=tt_qwen3_tts \
  DEVICE_IDS='(0)' \
  HF_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  TT_METAL_HOME=/path/to/tt-metal \
  ./build/tt_media_server_cpp -p 8000
```

Each worker reports ready after a short warmup utterance (at most 32 frames)
that compiles the kernels; `/health` returns 200 from then on. The first request
at a new prompt or audio length still compiles a little more. A worker whose
warmup fails logs `[Qwen3TtsRunner] Warmup failed` and exits. `/tt-liveness`
reports `runner_in_use: tt_qwen3_tts`.

### Docker workflow

`workflows/run_docker_server.py` runs the dev-catalog specs
`Qwen/Qwen3-TTS-12Hz-1.7B-{Base,CustomVoice,VoiceDesign}` and
`Qwen/Qwen3-TTS-12Hz-0.6B-{Base,CustomVoice}` on N150 and P150 with this server. It sets
`SERVER_MODE=cpp`, `MODEL_SERVICE=tts`,
`MODEL_RUNNER_TYPE=tt_qwen3_tts`, `HF_MODEL`, `DEVICE_IDS=(0)`,
`TTS_QWEN3_SEED=0` and the 16 / 64 / 64 queues. The specs pin no image, so pass
one that contains this server and a tt-metal with the Qwen3-TTS demo:

```bash
MODEL_SPECS_ENV=dev python3 run.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --tt-device n150 --workflow server --docker-server --override-docker-image <image>
```

## Requests

Requests carry the `OPENAI_API_KEY` bearer token (`your-secret-key` when unset).

```bash
# CustomVoice: a built-in speaker, optionally with a delivery instruction (1.7B)
curl -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
  -d '{"text": "Hello from Tenstorrent.", "speaker": "ryan", "language": "English",
       "description": "Speak slowly and warmly."}' \
  http://localhost:8000/v1/audio/speech -o ryan.wav

# VoiceDesign: describe the voice
curl -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
  -d '{"text": "Hello from Tenstorrent.",
       "description": "A calm older man speaking slowly, with a slight rasp."}' \
  http://localhost:8000/v1/audio/speech -o designed.wav

# Base, in-context clone: the clip and exactly what it says
curl -H "Authorization: Bearer your-secret-key" \
  -F text="Hello from Tenstorrent." \
  -F reference_text="exactly what my_voice.wav says" \
  -F voice=@my_voice.wav \
  http://localhost:8000/v1/audio/speech -o clone.wav

# Base, voice-only clone, with an instruction (1.7B)
curl -H "Authorization: Bearer your-secret-key" \
  -F text="Hello from Tenstorrent." \
  -F description="Whisper." \
  -F voice=@my_voice.wav \
  http://localhost:8000/v1/audio/speech -o clone_xvector.wav
```

## Metrics

Qwen3-TTS feeds the same `/metrics` series as TTS-2, under the same names and
labels; [`TTS_METRICS_TESTING.md`](TTS_METRICS_TESTING.md) is the procedure for
checking them. What the numbers mean for this runner:

| series | Qwen3-TTS |
|---|---|
| `tt_tts_codec_tokens_total` | one per codebook per generated frame, 16 per frame, so 200 per second of audio. `voice_source`: `voice_sample` for a clone (Base), `description` when the request has one (VoiceDesign, an instructed CustomVoice), else `default` |
| `tt_tts_audio_frames_total`, `tt_tts_vocoder_chunks_total` | always `batch="1"`; one chunk per 4 s (96000-sample) IPC message |
| `tt_tts_audio_sample_rate_hz` | 24000 |
| `tt_tts_conditioning_seconds{stage="voice_normalization"}` | Base requests: release validation plus the WAV downmix and resample in the server process |
| `tt_tts_conditioning_seconds{stage="text_conditioning"}` | other requests: release validation only, since the prompt is templated in the worker |
| `tt_tts_conditioning_seconds{stage="voice_encode"}` | Base requests: building the clone reference on the worker (speaker encoder, plus the codec encoder for an in-context clone). There is no reference cache, so every clone reports it |
| `tt_tts_conditioning_seconds{stage="prompt_compile"}` | never observed: prompt building happens inside generation and is not timed apart |

Codec tokens are counted as frames are generated, audio frames only once the
codec has decoded the whole utterance, so the two rates move together only over
windows longer than an utterance.

## Evals

Two evals score a live Qwen3-TTS server through the `qwen3_cpp` adapter
(`tts_api: qwen3_cpp`) in `test_module/eval_tests/tts_client.py`. Both use 8
LibriTTS-R `test.clean` speakers with 3 to 12 s clips and request
`language: English`.

| eval (`task_name`) | measures |
|---|---|
| `tts_speaker_similarity` | SECS: cosine similarity of `microsoft/unispeech-sat-base-plus-sv` embeddings between each reference clip and a voice-only clone of it reading another speaker's sentence. Baseline (`cross_speaker`): the same sentence cloned from a different speaker's clip; margin = SECS minus baseline |
| `tts_naturalness` | predicted MOS (UTMOS22-strong) over 8 sentences: voice-only clones on Base, speaker `ryan` on CustomVoice, the fixed description "A calm adult male voice, speaking clearly at a moderate pace." on VoiceDesign. No 0.6B request carries a description |

Each release passes when its means reach these thresholds. Calibration is on
N150 with `TTS_QWEN3_SEED=0`; the same thresholds apply on P150, where they have
not been measured yet. Upstream is Qwen3-TTS on CPU (fp32) for the same requests:

| checkpoint | evals | thresholds | N150 | upstream CPU |
|---|---|---|---|---|
| `1.7B-Base` | SECS, naturalness | SECS >= 0.90, margin >= 0.06, UTMOS >= 4.2 | 0.962, +0.128, 4.505 | 0.952, +0.125, 4.510 |
| `1.7B-CustomVoice` | naturalness | UTMOS >= 3.85 | 4.145 | 4.239 |
| `1.7B-VoiceDesign` | naturalness | UTMOS >= 4.1 | 4.420 | 4.445 |
| `0.6B-Base` | SECS, naturalness | SECS >= 0.88, margin >= 0.07, UTMOS >= 4.2 | 0.954, +0.138, 4.511 | 0.954, +0.145, 4.497 |
| `0.6B-CustomVoice` | naturalness | UTMOS >= 3.9 | 4.197 | 4.218 |

The tasks and thresholds are registered in
`reference_config/evals/eval_config.py`.
The adapter sends no seed, so reproducible scores need the server started with
`TTS_QWEN3_SEED` (the docker workflow sets 0).

The WER eval (`tts_generation`) is not registered for Qwen3-TTS: it speaks only
the Python media server's `response_format: json` contract. The `benchmarks`
workflow takes the adapter settings from these eval tasks; on Base it clones the
audio benchmark's 5.3 s LibriSpeech clip. Its TTFT is the time to the first
audio byte after the WAV header, which this server sends only once the whole
utterance is synthesized.

The specs are in the dev catalog, so set `MODEL_SPECS_ENV=dev`. Against a server
already listening on port 8000:

```bash
MODEL_SPECS_ENV=dev python3 run.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --tt-device n150 --workflow evals --server-url http://127.0.0.1 --service-port 8000
```

Use `--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` for a CustomVoice server, or
replace `--server-url ... --service-port ...` with
`--docker-server --override-docker-image <image>` to start the server as well.
The eval client sends `OPENAI_API_KEY`, else `API_KEY`, else `your-secret-key`,
the same key `--docker-server` gives the server. The adapter and the gates have
device-free tests:
`pytest tests/test_module/test_tts_client.py tests/test_module/test_tts_eval.py`.

## Tests

The C++ tests need no device:

| binary | covers |
|---|---|
| `qwen3_tts_config_test` | release and size resolution, `config.json` parsing |
| `qwen3_tts_settings_test` | `ttsEngineConfig()` in Qwen3 mode: defaults, per-worker devices, startup failures |
| `qwen3_tts_request_test` | request fields, per-release validation, the preprocessor's Qwen3 path |
| `tts_audio_ipc_test` | Qwen3 task fields, PCM16 audio chunks, chunking, WAV bytes |
| `qwen3_tts_runner_test` | `Qwen3TtsRunner` against a fake tt-metal Python package |

Run one from `build/` (e.g. `./qwen3_tts_runner_test`) or all with
`ctest --output-on-failure`. `qwen3_tts_runner_test` embeds Python and skips
unless the `python3` on `PATH` can import `numpy`.

## Limitations

- **Audio arrives only after the whole utterance.** The codec decodes all
  frames at once, so the response sends the WAV header, then nothing until
  generation finishes, then all the audio in 4 s chunks. Time to first audio is
  the full generation time.
- **Failures after the 200 end the stream.** The client sees a short or empty
  WAV; the worker logs the cause. Examples: a prompt too long for the cache
  (about 500 tokens of text, less for an in-context clone, whose clip and
  transcript are part of the prompt), the model ending before its first frame,
  a speaker or language the worker rejects before the server has the
  checkpoint's lists.
  Speech longer than `TTS_QWEN3_MAX_FRAMES` is cut at the budget without an
  error.
- **A dropped client does not stop generation.** The server notices a closed
  connection only when a write fails, and nothing is written during generation.
- **The server's own WAV cannot be sent back as a clip.** Its header carries the
  streaming size `0xFFFFFFFF`, which the clip parser rejects with a 400; re-save
  the file with an audio tool first.
- One release per deployment and batch 1 per worker; scale with `DEVICE_IDS`
  groups.
- One chip per worker: a multi-chip group such as `(0,1)` makes all its chips
  visible but opens only the first.
- The clone reference (speaker encoder, plus the codec encoder for an in-context
  clone) is rebuilt on every request; there is no reference cache.
