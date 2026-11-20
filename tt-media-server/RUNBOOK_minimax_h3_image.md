# Runbook: MiniMax-H3-Image on a Blackhole Galaxy (tt-media-server)

Serves `MiniMaxAI/MiniMax-H3-Image` (text to image, and edits of 1 to 16 reference images) at
`POST /v1/generations/minimaxai/minimax-h3-image` on one 4x8 Blackhole Galaxy. The request schema and its limits
are in `docs/api/openapi/minimaxai/minimax-h3-image.yaml`.

## 1. What you need

| Item | Value |
|---|---|
| Hardware | One Blackhole Galaxy (32 chips, 4x8 mesh, ring fabric). One replica uses the whole Galaxy. |
| tt-metal | A build that includes tenstorrent/tt-metal#65139 (the H3-Image pipeline) and the tt-metal policy modules this server's generations API imports (`models/tt_dit/pipelines/minimax_h3/policy.py` and `wan/policy.py` with the names `domain/minimax_h3_request.py` and `domain/wan22_i2v_a14b_request.py` import). Without the latter the server fails at import, before any model loads. |
| Weights | The raw release snapshot `MiniMaxAI/MiniMax-H3-Image`, plus the text encoder from `MiniMaxAI/MiniMax-H3` (`text_encoder/`, 23 files). The release does not ship the text encoder. |
| Disk | Room for a converted copy of the snapshot, the tt_dit weight cache and the kernel cache, on paths the server's user owns. |

## 2. Environment

| Variable | Set to |
|---|---|
| `MODEL_RUNNER` | `tt-minimax-h3-image` |
| `DEVICE` | `bh-galaxy` |
| `MODEL` | `MiniMax-H3-Image` |
| `MINIMAX_H3_IMAGE_MODEL_PATH` | The raw release snapshot, or an already converted one |
| `MINIMAX_H3_TEXT_ENCODER_DIR` | `MiniMaxAI/MiniMax-H3`'s `text_encoder/` |
| `MINIMAX_H3_IMAGE_CONVERTED_DIR` | Where the first start writes the converted snapshot (kept across restarts) |
| `TT_DIT_CACHE_DIR` | The tt_dit weight cache (written on the first load) |
| `TT_METAL_CACHE` | The kernel binary cache. **Persist it.** On a node with an empty cache, start-up took 17 min; with a warm one, warm-up takes about 1.5 min |
| `API_KEY` | The bearer key clients send |

We also ran with `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000`.

## 3. First start

The runner converts a raw release snapshot once into `MINIMAX_H3_IMAGE_CONVERTED_DIR` (one converter per host under a
file lock; a `.converted` marker skips it afterwards) and links the text encoder into it. It then loads the model and
runs `pipeline.warmup()`, which compiles every program a request can reach: every served size, 1 to 16 references,
and every prompt length. After warm-up, requests compile nothing.

```bash
cd tt-media-server
python -m uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000
```

## 4. Verify

```bash
curl -s http://<host>:8000/tt-liveness            # "model_ready": true once warm-up is done
curl -s -X POST http://<host>:8000/v1/generations/minimaxai/minimax-h3-image \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox asleep in fresh snow", "resolution": "2k"}'
# -> 202 and a job; poll GET /v1/jobs/{id} until "completed", then GET the output's url (relative to the server)
```

A request outside the schema (an unknown field, 17 references, references over 48 MP of effective area) returns 422
with the reason, before anything is queued.

## 5. Capacity

One request runs at a time. Measured on the served path (Nov 11, 2026):

| Request | Pipeline | Client sees |
|---|---|---|
| Text to image, 4k (any ratio) | 2.75–2.85 s | 3.1–3.2 s |
| Text to image, 2k | 1.26 s | 1.36 s |
| Edit, 1 reference, 4k | 3.5–4.6 s | 4.0–5.1 s |
| Edit, 16 x 1024x1024, 4k | 13.6 s | 14.1 s |
| Edit, 8 phone photos, 4k | 22.2 s | 23.1 s |
| PNG instead of JPEG, 4k | +0 s | +6.5 s (encode) |

Under a mixed load (55% 4k text to image, 15% 2k, 30% edits), at 60% of capacity: 547 images per hour, p50 4.8 s,
p95 15.5 s; at 40%: 370 images per hour, p50 3.3 s, p95 13.9 s. The p95 is queueing behind long edits, so it moves
little with load.
