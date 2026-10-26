# Image and video generation API

Design of the tt-media-server image and video generation API: the runner API
the console calls, not the public one. Each model has one submit endpoint,
`POST /v1/generations/{org}/{model}`, and every job then goes through a common
`/v1/jobs` API (status with outputs, content, cancel, delete).

`{org}/{model}` is the lowercase Hugging Face ID of the upstream model, without
format or quantization suffixes. Models covered:

| Model | Endpoint |
| ----- | -------- |
| Wan2.2 I2V A14B | `POST /v1/generations/wan-ai/wan2.2-i2v-a14b` |
| Qwen-Image-Edit-2511 | `POST /v1/generations/qwen/qwen-image-edit-2511` |
| FLUX.2 klein 4B | `POST /v1/generations/black-forest-labs/flux.2-klein-4b` |
| FLUX.3 Image klein | `POST /v1/generations/black-forest-labs/flux-3-image-klein` |
| Ideogram 4 | `POST /v1/generations/ideogram-ai/ideogram-4` |
| MiniMax-H3 FL2VA | `POST /v1/generations/minimaxai/minimax-h3-fl2va` |
| MiniMax-H3 FL2VA, LightX2V | `POST /v1/generations/tenstorrent/minimax-h3-fl2va-lightx2v` |
| MiniMax-H3 Ref2VA | `POST /v1/generations/minimaxai/minimax-h3-ref2va` |
| MiniMax-H3 Ref2VA, LightX2V | `POST /v1/generations/tenstorrent/minimax-h3-ref2va-lightx2v` |
| LTX-2.3 22B distilled | `POST /v1/generations/lightricks/ltx-2.3-22b-distilled` |

When one upstream repo holds several checkpoints, the checkpoint's name is
appended (`minimax-h3-fl2va`, `ltx-2.3-22b-distilled`). Variants we derive,
such as distillations, live under `tenstorrent/` and share their base
model's request schema: the same API, a different route.

## Files

- `common.yaml`: the `/v1/jobs` API and everything the models share: schemas
  (`Media`, `VideoMedia`, `AudioMedia`, `Seed`, `ImageFormat`, `Job`,
  outputs, errors), responses and
  auth. Its description covers the API as a whole.
- `models/{org}/{model}.yaml`: one complete OpenAPI file per model, describing
  what that model's deployment serves: its submit endpoint and request schema,
  plus `/v1/jobs` by reference to `common.yaml`.
- `REQUIREMENTS.md`: behavior the runner must have that the spec deliberately
  leaves out, and follow-ups.

## Lifecycle

These files are the design. Each model is implemented on its own:

1. Write the model's Pydantic request model and router (and, with the first
   model, the shared job and output models).
2. Generate that deployment's spec (its router plus `/v1/jobs`) and compare it
   by meaning against `models/{org}/{model}.yaml`, with `common.yaml`
   references followed. A text diff won't work: FastAPI lays out OpenAPI
   differently.
3. Replace the YAML with the generated `models/{org}/{model}.openapi.json` and
   have CI regenerate and compare it on every PR, as `cpp_server` does for its
   `resources/openapi.json`.

Until the last model is implemented, a change to a shared schema goes both into
the Pydantic models and into `common.yaml`. Delete `common.yaml` with the last
model.

## Preview

```sh
npx @scalar/cli document serve --watch models/qwen/qwen-image-edit-2511.yaml
```

Use `npx @scalar/cli`, not `scalar`: git ships its own `scalar`. Model files
refer to `common.yaml`; whether the preview follows those references hasn't
been checked.

## Check

```sh
pip install openapi-spec-validator
for f in common.yaml models/*/*.yaml; do openapi-spec-validator "$f"; done
```
