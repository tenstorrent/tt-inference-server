# LTX 2.3 Fast Local Media Server Setup

This runbook starts the `tt-media-server` branch that serves LTX 2.3 Fast on
the local Blackhole Galaxy machine and shows how to submit a text-to-video job.

## Required Checkouts

Use these exact branches and commits:


| Repo                  | Path                        | Branch                             | Commit                                     |
| --------------------- | --------------------------- | ---------------------------------- | ------------------------------------------ |
| `tt-inference-server` | `$HOME/tt-inference-server` | `rsalman/ltx-2.3-distilled-runner` | `9f1ea55b5bb8a15ed9452884960d358b7f2647ea` |
| `tt-metal`            | `$HOME/tt-metal`            | `ltx-2.3-fast-25fps`               | `93f4f14d2b8b14fabfc6a151f24b5c0dbbd8a0c4` |


The server uses `TT_METAL_HOME=$HOME/tt-metal`. The `tt-metal`
checkout must already be built for this branch.

The served shape is fixed by the runner: 153 frames at 25 fps, 1088x1920
generation, exported as MP4. Requests should only provide prompt parameters
such as `prompt`, `negative_prompt`, `num_inference_steps`, and `seed`; do not
try to override frame count, resolution, or fps per request.

## Create the Media Server Python Env

The `tt-metal/python_env` already has the TT stack (`ttnn`, LTX dependencies,
and the local `tt-metal` package paths), but it does not include the web server
packages. Copy it into `tt-media-server/python_env`, then install the media
server requirements into the copy.

```bash
cd $HOME/tt-inference-server/tt-media-server

# Create the media server env from the TT env if it does not already exist.
test -d ./python_env || cp -a $HOME/tt-metal/python_env ./python_env

# Install the FastAPI/uvicorn/media-server deps into the copied env.
/usr/local/bin/uv pip install \
  --python "$PWD/python_env/bin/python" \
  --index-strategy unsafe-best-match \
  --overrides "$PWD/uv-overrides.txt" \
  -r "$PWD/requirements.txt"

# Pin diffusers to the LTX branch expectation without letting it disturb
# the rest of the TT env's dependency set.
/usr/local/bin/uv pip install \
  --python "$PWD/python_env/bin/python" \
  --no-deps diffusers==0.38.0
```

Sanity check the env:

```bash
cd $HOME/tt-inference-server/tt-media-server

./python_env/bin/python - <<'PY'
for name in ["ttnn", "uvicorn", "fastapi", "diffusers", "av"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "OK"))
PY
```



## Hugging Face Auth

The LTX pipeline needs access to gated Hugging Face repos, including:

- `google/gemma-3-12b-it-qat-q4_0-unquantized`
- `Lightricks/LTX-2.3`

Log in with the same env that will run the server:

```bash
cd $HOME/tt-inference-server/tt-media-server
./python_env/bin/hf auth login
```

Then verify that the token can see both repos:

```bash
./python_env/bin/hf auth whoami

./python_env/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
for repo in [
    "google/gemma-3-12b-it-qat-q4_0-unquantized",
    "Lightricks/LTX-2.3",
]:
    info = api.model_info(repo)
    print(repo, "OK", info.sha)
PY
```

For a one-off launch, `HF_TOKEN` can be exported instead of writing credentials
to disk:

```bash
export HF_TOKEN='hf_...'
```

Do not commit or paste the real token into files.

## Launch the Server

The branch provides `start_ltx_fast.sh`, which bakes in the local LTX serving
configuration:

- `TT_METAL_HOME=$HOME/tt-metal`
- `TT_DIT_CACHE_DIR=$HOME/tt-metal/tt_dit_cache`
- `TT_VIDEO_OUTPUT_DIR=$HOME/tt-media-videos`
- `LTX_YUV_EXPORT=1`
- `DEVICE=galaxy`
- `MODEL=LTX-2.3-distilled`
- `SERVED_MODEL_NAME=Lightricks/LTX-2.3-Fast`
- `ENABLE_JOB_PERSISTENCE=true`
- `JOB_DATABASE_PATH=$HOME/ltx-media-server-jobs.db`

Start it manually:

```bash
cd $HOME/tt-inference-server/tt-media-server
./start_ltx_fast.sh
```

It binds to port `8000` by default. To use another port:

```bash
PORT=8082 ./start_ltx_fast.sh
```

Startup performs model loading and a traced warmup before `/health` returns
`200`, so cold start can take several minutes.

Check readiness from another shell:

```bash
curl -i http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | jq
```

`/health` and `/v1/models` do not require an API key. Generation endpoints do.
Unless `API_KEY` or `API_KEYS` is set, the default key is `your-secret-key`.

## Submit a Video Job

```bash
AUTH='Authorization: Bearer your-secret-key'

JOB=$(
  curl -s -X POST http://127.0.0.1:8000/v1/videos/generations \
    -H "$AUTH" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a cinematic shot of a futuristic city at sunset","seed":10}' \
    | jq -r .id
)

echo "$JOB"
```

Poll until `status` becomes `completed`:

```bash
curl -s -H "$AUTH" \
  "http://127.0.0.1:8000/v1/videos/generations/$JOB" | jq
```

Download the result:

```bash
curl -L -H "$AUTH" \
  -o out.mp4 \
  "http://127.0.0.1:8000/v1/videos/generations/$JOB/download"
```



## Useful Troubleshooting

If model load fails with a Hugging Face `401 Unauthorized` or `GatedRepoError`,
the server process cannot see a token with access to the gated repo. Run the
Hugging Face verification commands above from `tt-media-server/python_env`.

If startup fails because `./python_env/bin/uvicorn` is missing, recreate or
repair the media env with the setup commands above.

If `/health` stays non-200 during initial startup, wait for warmup to finish.
If it remains unhealthy after warmup, check the server log/terminal for the
first model-loading exception.

Only one process can own the TT chips at a time. Stop any other server or test
using the devices before launching this server.