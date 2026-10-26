# Implementation requirements

Behavior tt-media-server must have that `openapi.yaml` deliberately leaves out.
The spec says what clients can rely on; this file says how the runner gets
there.

## All models

- **No cold compiles after ready.** Once tt-media-server reports ready, every
  resolution, shape and input combination the spec allows runs at full speed
  on its first request: either it was warmed up at startup, or the model
  ensures internally that no input triggers a compile (e.g. padding to
  compiled shapes). This covers output sizes, input image counts, and first
  and last frame conditioning.
- **`count` is fixed at 1 for now.** Allowing more needs model work so it
  makes product sense (DP > 1 or batched inference), and the console has to
  learn to route `count > 1` requests to runners that batch. The field stays
  so this can change without a schema change.
- **Image URLs are fetched at submit through `utils/media_downloader.py`**
  (#4974) for every model, including the image models that don't use it yet:
  allowlist or public-host mode per deployment, SSRF checks on every redirect,
  one deadline, 30 MB cap, and its 400 / 413 / 422 mapping, as the spec states.
- **Inline images are data URIs.** `Media` takes an `https://` URL or a data
  URI (`data:image/png;base64,…`); bare base64 gets a 422 that names the
  expected form. Today's runner takes bare base64 (or an http(s) URL), so this
  changes its parsing. The console converts if a customer sends bare base64.
- **Retention and `expires_at`.** Keep today's behavior, deleting a finished
  job and its files `job_retention_seconds` (24 hours by default) after
  `completed_at`, and report it as `expires_at`. The console copies outputs
  into its own storage when a job completes, so this only has to outlast its
  download and retries.
- **Video and audio inputs** (`VideoMedia`, `AudioMedia`) go through the same
  downloader and data-URI parsing as images, with their own caps: 50 MB per
  video, 15 MB per audio clip (images stay at 30 MB). Today's downloader has
  one cap; `MediaTooLarge` covers all three.
- **Variants share a request model.** Deployments of the same checkpoint with
  different sampling (e.g. `minimaxai/minimax-h3-fl2va` and
  `tenstorrent/minimax-h3-fl2va-lightx2v`) use one Pydantic request model and
  separate routes; `Job.model` reports the route's `{org}/{model}`. Further
  distillations follow the same pattern, as `tenstorrent/{base}-{variant}`.
- **Unknown fields are rejected** (`extra="forbid"` on every request model),
  with FastAPI's default 422 body.
- **One router and one Pydantic request model per model**, mounted by
  `model_runner`, so each deployment's `/openapi.json` shows only its own
  model's submit endpoint plus `/v1/jobs`.

## Qwen-Image-Edit-2511

- **Long prompts.** The spec sets no prompt length limit, and diffusers'
  `QwenImageEditPlusPipeline` does not truncate (its `max_sequence_length=512`
  is not applied). The runner must handle the longest prompts customers send,
  together with 3 input images (each about 188 image tokens), without failing
  on device: pad to a supported length, or truncate, with a clear 422 only as
  a last resort. Measure the largest sequence the compiled pipeline accepts.
- **`webp` output.** Today's image path only produces JPEG and PNG.

## FLUX.2 klein 4B

- **14 output shapes**: 7 aspect ratios × `1K`/`2K`, listed in the spec. All
  must meet "no cold compiles after ready", together with 0–4 input images.
- **Default shape with input images**: snap the first image's ratio to the
  closest supported one, at the requested tier. Input images themselves are
  resized as diffusers does (to at most 1 MP, sides multiples of 16).

## FLUX.3 Image klein

- **Prompt upsampler, always on.** Bring up `flux-3-image-klein-pup` (a 9B
  vision-language model fine-tuned from Qwen3.5-9B) on TT hardware and run
  every prompt through it, as the reference `generate()` does: greedy
  decoding, reasoning off, up to 3,584 new tokens, the references as image
  inputs (downscaled per `MAX_PIXELS_BY_REFS` in `src/flux3/pup.py`). BFL says
  not to run the model without it. Its decode time adds to every request.
- **Boxes in the prompt.** Use the reference `flux3.bbox` handling: `detect`
  finds a trailing box list (0–1000 grid), `plan` switches the upsampler to
  its layout prompts, `reconcile` keeps the supplied rows exact. Skip the
  upsampler only where the reference does (`is_finished`: a prompt that is
  already an upsampler caption with its boxes).
- **21 output shapes** (7 ratios × `1K`/`2K`/`4K`) × 0–10 references, under
  "no cold compiles after ready". References change the sequence length, so
  expect padding to a few fixed lengths. Prepare references as the reference
  pipeline does: the first at most 4 MP, each further one at most 1 MP.
- **Long prompts.** The upsampler's caption can be long; the text encoder
  takes up to 4,096 tokens (padded to multiples of 80 in the reference).
- **Moderation is the console's job**, as for Ideogram: the reference code
  refuses flagged prompts, references and outputs; the runner doesn't.

## MiniMax-H3 (FL2VA and Ref2VA)

- **Checkpoints and variants.** FL2VA runs the `FL2VA` checkpoint of
  `MiniMaxAI/MiniMax-H3` (`transformer/` in the diffusers layout; text-only
  requests run as t2va), Ref2VA the `Ref2VA` checkpoint (`transformer_ref/`).
  Base deployments sample 50 steps; the LightX2V ones load lightx2v's per-task
  Turbo adapter (`MINIMAX_H3_LORA_PATH`) and sample 5. The Ref2VA adapter was
  trained at 544p and is served at 768p.
- **`adaptive`.** FL2VA already sizes the canvas from the keyframe (first,
  else last). Ref2VA's `adaptive` is new: snap the first image's ratio (else
  the first video's) to the closest of the 6 listed ratios. Reject an explicit
  ratio with keyframes, and `adaptive` without them, with 422.
- **Input limits** (422): keyframes 256–5760 px per side, ratio 1:4 to 4:1;
  references at most 9 images, 3 videos, 3 audio clips, 12 total, not audio
  alone; video and audio clips 2–15 s each, at most 15 s combined per kind;
  reference videos must report a frame rate; audio mono or stereo. All as
  `minimax_h3_policy` / tt-metal's `minimax_h3/policy.py` enforce today.
- **Long prompts.** Up to `MINIMAX_H3_MAX_TEXT_TOKENS` (3,008 by default)
  text tokens; longer gets 422.
- **No cold compiles after ready** across 6 ratios plus adaptive keyframe
  canvases (any ratio from 1:4 to 4:1), 12 durations (4–15 s, 107–362
  frames), 0–2 keyframes, and the Ref2VA reference mixes.
- **Output audio**: AAC, 32 kHz stereo, reported in `VideoOutput.audio`.

## LTX-2.3 22B distilled

- **Output exactly 1920×1080.** The two-stage pipeline needs sides divisible
  by 64, so it runs at 1920×1088; the tt-metal model implementation must pad
  and crop internally so callers (and tt-media-server) see 1920×1080. Today
  the runner serves and reports 1088.
- **One served shape**: 153 frames at 25 fps (6.12 s), the only shape the
  traces are captured for. fps sets the audio latent length and the A/V
  cross-PE, so it cannot vary per request: the spec's `fps` allows only 25.
  The old API's `num_frames`, `height` and `width` (accepted only at the
  served values) have no equivalent here.
- **Output audio**: AAC, reported in `VideoOutput.audio` with its actual
  sample rate.

## Ideogram 4

- **Caption serialization.** Write `structured_prompt` out in the model's key
  order (`reorder_caption_keys` in `ideogram-oss/ideogram4`) as compact JSON
  (`separators=(",", ":")`, `ensure_ascii=False`), whatever order the client
  sent. Also reject a `bbox` with `ymin > ymax` or `xmin > xmax` (422); the
  schema can't express it.
- **Presets** map to the reference `PRESETS`: `turbo` → `V4_TURBO_12`,
  `default` → `V4_DEFAULT_20`, `quality` → `V4_QUALITY_48`.
- **14 output shapes** (7 ratios × `1K`/`2K`) × 3 presets, under "no cold
  compiles after ready".
- **Long captions.** The spec sets no caption length limit; the reference
  pipeline rejects captions over 2048 text tokens (`max_text_tokens`). The
  runner must handle any caption up to 2048 tokens without a cold compile
  (e.g. padding to a few fixed lengths), and reject longer ones with 422.
  Benchmarks so far use about 238 tokens.
- **Moderation is the console's job, from day one.** The runner does no
  safety filtering. Ideogram's `docs/safety.md` expects anyone serving the
  model to screen every prompt and every output image (the reference uses
  Hive's moderation APIs) and calls running without that unsupported, so the
  console must have prompt and output moderation in place before it exposes
  Ideogram 4.

## Wan2.2 I2V A14B

- **4 output shapes**: `16:9` and `9:16` at 480p and 720p, under "no cold
  compiles after ready". Today's runner serves one size per deployment,
  1280×720 on Galaxy-class meshes and 832×480 otherwise
  (`wan22_target_resolution` in `config/constants.py`), landscape only.
- **Frames are fitted to the output size**: `first_frame` and `last_frame` are
  resized and center-cropped to it.
- **Default negative prompt** is the reference `sample_neg_prompt`, as the
  spec states; today's runner passes none.

- **`last_frame` quality.** Wan2.2 I2V-A14B was not trained for last-frame
  conditioning (the reference repo conditions only on frame 0); it works
  through the frame-position input diffusers and tt_dit both accept. Check
  first+last-frame quality against diffusers on a few pairs before the console
  exposes it.
- **Last-frame cost.** Conditioning the last frame makes the VAE encode all 81
  frames and, the first time, recompiles. Under "no cold compiles after
  ready", startup must cover it (e.g. a last-frame warm-up request).

## Follow-ups

Not in the first version of the spec; add once the need is real.

- **`Idempotency-Key` on submit.** Lets the console retry a submit after a
  timeout without starting a second job and spending device time twice.
  Neither tt-media-server nor, as far as we know, the console supports it
  today. Proposed behavior: an optional `Idempotency-Key` header (at most 255
  characters) on every `POST /v1/generations/…`; retrying with the same key
  returns the original job; a key is remembered as long as its job; reusing a
  key with a different body is rejected (409 or 422, to decide). Keys share
  one space per deployment, so they must be random (e.g. a V4 UUID). Runway is
  the only video provider found that has it; none of the providers serving
  these four models do. The text from the earlier spec draft is in git
  history.
- **Plain-text prompts for Ideogram 4 (prompt enhancement).** Ideogram 4 only
  works with a structured caption; plain text needs an LLM to expand it first
  ("magic prompt"). Not in the first version because a performant LLM on TT
  hardware would push timelines back too far. When added: a `prompt` string
  field alongside `structured_prompt`, always expanded, each output reporting
  the caption used. Ideogram open-sourced the instructions
  (`src/ideogram4/magic_prompt_system_prompts/v1.txt` in
  `ideogram-oss/ideogram4`, about 4,000 words, any LLM) but tested them only
  with Claude Opus/Sonnet, and says they differ from its production magic
  prompt. Options:
  - Ideogram's hosted magic-prompt API (`ideogram-4-v1`): free, closest to
    Ideogram's results; sends customer prompts to Ideogram, external
    dependency on the request path.
  - The open system prompt with Claude through an API: tested quality; sends
    customer prompts to a third party, adds latency and cost per request.
  - The open system prompt with an open-weight LLM served on TT hardware: no
    third party; untested quality, needs a model that follows long
    instructions and reliably emits strict JSON.

  Whichever is chosen, validate the LLM's output against `Ideogram4Caption`
  (retry once on invalid JSON) and keep the LLM behind an OpenAI-compatible
  endpoint so it can be swapped by configuration.
- **LTX-2.3: image-to-video and more shapes.** The distilled pipeline can
  condition on a first frame (`images` with frame 0); the runner turns it off
  today. Also portrait (`9:16`), last frame, and other durations or frame
  rates, each needing its own traces.
- **MiniMax-H3: other size tiers** (`480p`, and `2K`, which isn't in the open
  release).
