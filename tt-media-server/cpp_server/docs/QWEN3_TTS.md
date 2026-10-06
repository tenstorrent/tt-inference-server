# Qwen3-TTS serving

`MODEL_SERVICE=tts MODEL_RUNNER_TYPE=tt_qwen3_tts` serves
[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) on
`POST /v1/audio/speech`. Each worker process embeds a Python interpreter and
drives tt-metal's `models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline` on one
device group, one utterance at a time. Audio is mono PCM16 WAV at 24 kHz.

## One release per deployment

Qwen3-TTS ships as three releases that choose the voice differently. A
deployment serves exactly one, chosen by its checkpoint, never by the request:

| release (`TTS_QWEN3_RELEASE`) | hub ids | voice comes from |
|---|---|---|
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | a WAV clip to clone |
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | one of nine built-in speakers |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (there is no 0.6B) | a sentence describing the voice |

The server works out the release and size before any worker starts, in this
order: `TTS_QWEN3_RELEASE` / `TTS_QWEN3_MODEL_SIZE` if set; else the
checkpoint's `config.json` if it is on local disk (`QWEN3_TTS_CKPT`, `HF_MODEL`
as a directory, or the HF hub cache); else the `HF_MODEL` hub id's name; else
tt-metal's default, 1.7B Base. Startup fails when they disagree or when none
of them says. At warmup each worker loads the checkpoint and refuses to serve
if it is not the release the server was configured for.

Knowing the release lets the server reject a request it cannot serve with an
HTTP 400 before streaming starts. Once the WAV header has gone out, a failure
can only end the stream early, so the client sees 200 and a short or empty WAV.

## Request fields

JSON (`Content-Type: application/json`) or `multipart/form-data`; the voice clip
can only travel as a multipart file part (PCM16 WAV, any rate, any channel
count; it is downmixed and resampled to 24 kHz mono).

| field | meaning for Qwen3-TTS |
|---|---|
| `text` | required; what to say |
| `description` | VoiceDesign: the voice description (required). CustomVoice: an optional delivery instruction (1.7B only). Base: an optional instruction for a voice-only clone (1.7B only) |
| `speaker` | CustomVoice only, required: a built-in speaker name (case-insensitive) |
| `language` | optional everywhere: `Auto` or a language the checkpoint lists (case-insensitive) |
| `reference_text` | Base only, optional: the clip's transcript |
| WAV file part | Base only, required: the voice to clone |

What each release accepts (anything else is a 400):

- **custom_voice**: `speaker` required; no WAV, no `reference_text`;
  `description` only on 1.7B. `language` defaults to `English`, as in the
  tt-metal demos.
- **voice_design**: `description` required; no `speaker`, WAV or
  `reference_text`. `language` defaults to `Auto`.
- **base**: the WAV is required; no `speaker`. With `reference_text` the clone
  is in context (closer, but the transcript must match the whole clip), without
  it the clone uses the voice alone. `description` only for a voice-only clone,
  only on 1.7B. `language` defaults to `Auto`. Clips longer than
  `TTS_QWEN3_MAX_REFERENCE_SECONDS` are refused; 3 to 10 s of clean speech is
  the useful range.

Fields that are present but blank count as absent. Unknown speakers and
languages are rejected up front once the server can read the checkpoint's
`config.json`; for a hub checkpoint that is not downloaded yet, the first worker
downloads it and the server picks the lists up within seconds after that.

## Environment

| variable | meaning | default |
|---|---|---|
| `MODEL_SERVICE` | `tts` | |
| `MODEL_RUNNER_TYPE` | `tt_qwen3_tts` | `tt_tts` |
| `DEVICE_IDS` | one worker per group, e.g. `(0),(1)`; each opens its group as device 0 | `(0)` |
| `HF_MODEL` | hub id or local directory of the checkpoint | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `QWEN3_TTS_CKPT` | local checkpoint directory; wins over `HF_MODEL` | |
| `TTS_QWEN3_RELEASE` | `base`, `custom_voice` or `voice_design` | derived, see above |
| `TTS_QWEN3_MODEL_SIZE` | `1b7` or `0b6` | derived, see above |
| `TTS_QWEN3_MAX_FRAMES` | frame budget per utterance, 12.5 frames a second; sizes the KV cache | `400` (32 s) |
| `TTS_QWEN3_SEED` | reseed sampling before every request (reproducible output); unset = unseeded | unset |
| `TTS_QWEN3_MAX_REFERENCE_SECONDS` | longest clip accepted for cloning, at most 40 | `30` |
| `TTS_MAX_USERS` | requests accepted at once (running plus queued); more get 429 | number of workers |
| `TT_METAL_HOME` | tt-metal checkout: `models.demos` is imported from here, kernels compile here | required |
| `TT_PYTHON_PATH` | extra directory put first on the embedded interpreter's `sys.path` | unset |

`TTS_AUDIO_SAMPLE_RATE_HZ` and `TTS_VOICE_SAMPLE_RATE_HZ` default to 24000 for
this runner and startup fails if either is set to anything else. The TTS-2
tokenizer (`TTS_TOKENIZER_PATH`) is not used. Each worker sets its own
`TT_VISIBLE_DEVICES`, uses `$TT_METAL_HOME/built/<device ids>` as
`TT_METAL_CACHE` and runs with `TT_METAL_HOME` as its working directory.

Python: the embedded interpreter must resolve the tt-metal venv that matches
`TT_METAL_HOME` (`ttnn`, `torch`, `librosa`, `soundfile`, `safetensors`,
`huggingface_hub`). Launch with that venv activated so its `python3` is first on
`PATH`, and scrub any ambient `PYTHONPATH` that points at another tt-metal
checkout: `docs/embedding_serving_guide.md` explains both traps. Hub
checkpoints download on first use into the HF cache (`HF_HOME` /
`HF_HUB_CACHE`); 3.6 GB at 1.7B, 1.8 GB at 0.6B.

## Launch

```bash
source /path/to/tt-metal/python_env/bin/activate   # the venv matching TT_METAL_HOME

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

`/health` reports `runner_in_use: tt-qwen3-tts`. Each worker reports ready
after its warmup utterance, which compiles the kernels; the first requests of a
new prompt length or audio length still compile a little more.

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

The file part's field name does not matter; the first file in the form is the
clip.

## Limitations

- **Audio arrives only after the whole utterance.** The codec decodes all
  frames of an utterance at once, so the response sends the WAV header, then
  nothing until generation finishes, then all the audio (in 4 s IPC chunks).
  Time to first audio is the full generation time.
- One utterance at a time per worker (batch 1); scale with `DEVICE_IDS` groups.
- Failures the server cannot foresee (text too long for the prompt cache, the
  model ending before its first frame) happen after the 200 and end the stream
  with whatever audio exists, usually none. They are logged by the worker.
- A clone reference is rebuilt (speaker encoder, plus the codec encoder for an
  in-context clone) on every request; there is no reference cache yet.
