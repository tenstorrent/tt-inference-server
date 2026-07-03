# comfyui-media-server

Standalone FastAPI media-inference server that backs the Tenstorrent **ComfyUI**
custom nodes (`ComfyUI/custom_nodes/tenstorrent_nodes`). It serves diffusion
image (SDXL, SD3.5) and video (Wan2.2 T2V) generation on Tenstorrent hardware,
and — unlike the OpenAI-API `tt-media-server/` in this repo — exposes the
**staged / streaming pipeline contract** the ComfyUI nodes drive.

This is a drop-in relocation of the server that previously lived at the **root**
of a tt-metal working branch (PR #47510 content, ground truth:
`tt-metal` branch `samt/standalone-media-20260703`). It consumes tt-metal/ttnn as
an **installed dependency** — it does not vendor them.

> Status: **runnable locally, Docker path drafted.** The launcher is fixed for
> the relocated layout (external `TT_METAL_HOME`), so it runs directly against a
> built host tt-metal today — see **[`LOCAL_TESTING.md`](./LOCAL_TESTING.md)**
> (Path 1). A `Dockerfile` + `docker-entrypoint.sh` for the `run.py --docker-server`
> path exist but the image hasn't been built/tested on hardware (Path 2). Not yet
> wired into `model_spec.json`, and `server_manager.py` still auto-launches the
> old in-tt-metal layout.

## Why this is separate from `tt-media-server/`

They are different products that overlap only on one-shot generation:

| | this server | `tt-media-server/` |
|---|---|---|
| Consumer | ComfyUI custom nodes | OpenAI-style API clients |
| One-shot | `/image/generations`, `/video/generations` | `/generations`, `/generations/i2v` |
| Staged ops | `/latent/denoise`, `/vae/encode`, `/vae/decode`, `/video/denoise`, `/video/vae_decode` | — none — |
| Streaming | per-step `DenoiseStep` on `/latent/denoise_stream` (NDJSON) | LLM/audio only |

The ComfyUI nodes call the staged `latent`/`vae` endpoints directly (the
"Milestone 2" path); `tt-media-server/` implements none of them. Strategy 1
(chosen) keeps this server intact so the node contract is preserved verbatim;
converging it into `tt-media-server/`'s runner/service/route pattern is a
possible follow-up, not a prerequisite.

## Layout (flat, drop-in)

```
comfyui-media-server/
  server.py               FastAPI app: /health, /metrics, /image|/video/generations,
                          staged /latent/* and /vae/* + /video/* ops
  worker.py               multiprocessing worker; defers ALL ttnn imports to the child
  sdxl_runner.py          } tt-metal-coupled runners — the ONLY files that import
  wan_runner.py           }   ttnn / models.* (the PR content, consumed as a dep)
  sd35_runner.py          }
  sdxl_config.py          } per-model config dataclasses (import only device_specs)
  wan_config.py           }
  sd35_config.py          }
  device_specs.py         server-side board + (model,board) deployment matrix
                          (its own DeviceClass; NOT the tt-metal #48616 primitive)
  utils/                  cache / image / logger / validation helpers (no ttnn)
  image_test.py           HTTP client used for smoke/accuracy testing
  launch_server.sh        launcher (NEEDS relocation fixups — see below)
  stop_server.sh          graceful shutdown (SIGTERM root uvicorn -> workers close device)
  requirements-server.txt fastapi + uvicorn (installed into tt-metal's python_env)
```

Imports are flat (`from device_specs import ...`, `from worker import ...`,
`from utils.logger import ...`), so run with this directory on `PYTHONPATH`.

## tt-metal dependency boundary

Only `sdxl_runner.py` (`models.demos.stable_diffusion_xl_base.*`),
`wan_runner.py` / `sd35_runner.py` (`models.tt_dit.*`), and one deferred
`DenoiseStep` import in `worker.py` touch tt-metal. Those symbols are the
in-flight tt-metal PRs (#47509 SDXL per-component LoRA, #47265 Wan per-request
guidance, #47715 DenoiseStep, #48616 DeviceClass, #46519 LoRA). This server must
run against a tt-metal build that contains them — pin that SHA when the Docker
image is added (mirrors `tt-media-server/`'s `tt_metal_commit`).

## How it gets launched (target: `run.py --docker-server`)

The ComfyUI nodes (`ComfyUI/custom_nodes/tenstorrent_nodes/server_manager.py`)
supervise the server as a subprocess. Today they spawn
`$TT_METAL_DIR/launch_server.sh` directly (the makeshift local path). The target
is to have them instead drive this repo's standard entrypoint, in **dev mode**:

```
python run.py --model <sdxl|wan> --workflow server \
    --tt-device p300x2 --docker-server --dev-mode \
    --override-docker-image <locally-built comfyui-media-server image>
```

`run.py` enforces that `--dev-mode --docker-server` **requires**
`--override-docker-image` (`run.py:572`), so the dev loop points at a locally
built image without any `model_spec.json` change. `run.py` then pulls/starts the
container (`workflows/run_docker_server.py`), publishing the service port and
passing `MODEL=<name> DEVICE=<device>` env into the container's entrypoint.

### Two things this requires

1. **A container entrypoint that matches `run.py`'s contract.** `run.py` starts
   the image detached, publishes a port, and sets `MODEL`/`DEVICE` env — it does
   **not** call `launch_server.sh` directly. The image needs an entrypoint that
   reads those env vars and starts `server.py` on the service port (adapt
   `launch_server.sh` into that entrypoint — see relocation fixups below).
2. **Routing away from `tt-media-server`.** `sdxl`/`wan` model names already
   resolve to the OpenAI-API image (`inference_engine: "media"` →
   `tt-media-inference-server`, via `generate_default_docker_link` in
   `workflows/model_spec.py`). That image has none of the staged ComfyUI ops.
   `--override-docker-image` sidesteps this for dev; a permanent path needs a
   distinct `inference_engine` (e.g. `comfyui`) → its own image repo, or distinct
   model-spec entries.

## Relocation fixups feeding the Docker entrypoint

`launch_server.sh` was written for the old layout where it lived at the tt-metal
root (`TT_METAL_HOME = <script dir>`). When it becomes the image entrypoint:

- `launch_server.sh:45` — `TT_METAL_HOME="$SCRIPT_DIR"` must point at the
  **tt-metal built into the image**, not this directory.
- `launch_server.sh:66` — `requirements-server.txt` is now in this dir.
- `launch_server.sh:81` — `PYTHONPATH` must include **both** tt-metal (for
  `ttnn` / `models.*`) **and** this dir (for `server`, `device_specs`, `utils`).
- `launch_server.sh:303` — runs `$TT_METAL_HOME/server.py`; `server.py` now lives
  here.
- Map `run.py`'s `MODEL`/`DEVICE` env → `server.py`'s `--model`/`--board` args.

## Status of the work

Done:
- ✅ Relocated launcher (`launch_server.sh`) — external `TT_METAL_HOME`, dual
  `PYTHONPATH`, runs `server.py` from this dir. Runnable locally now (Path 1).
- ✅ `Dockerfile` (pinned tt-metal build) + `docker-entrypoint.sh`
  (`MODEL`/`DEVICE` env → launcher) + `build_image.sh` for the Docker path.
- ✅ Legacy SDXL-only set excluded from the relocation.

Not done — next steps:
1. **Build + prove the image**: `./build_image.sh`, then `run.py …
   --docker-server --dev-mode --override-docker-image comfyui-media-server:dev`
   healthy on the published port. (Long first build; untested on hardware.)
2. **Update `ComfyUI/.../server_manager.py`** to spawn the `run.py …
   --docker-server` command instead of `launch_server.sh`, keeping the `/health`
   poll.
3. **Decide permanent routing** (`inference_engine: comfyui` + image repo) vs.
   staying on the `--override-docker-image` dev path (avoids colliding with
   `tt-media-server`'s `sdxl`/`wan` names).
4. **Add `model_spec.json` entries** for the ComfyUI models once routing is
   decided.
