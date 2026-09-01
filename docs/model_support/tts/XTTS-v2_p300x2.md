# xtts-v2 Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other tts models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`xtts-v2` is also supported on hardware:

- [P150](XTTS-v2_p150.md)
- [N150](XTTS-v2_n150.md)

## Quickstart - Deploy xtts-v2 Inference Server on BH QuietBox 2

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
# --dev-mode is required: the model spec currently lives in the dev catalog (EXPERIMENTAL)
python3 run.py --dev-mode --model XTTS-v2 --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| Model Status | 🟡 Experimental |
| Max Batch Size | 1 (per chip; 4 concurrent requests across the box) |
| Implementation Code | [xtts_v2](https://github.com/tenstorrent/tt-metal/tree/main/models/experimental/xtts_v2) |
| tt-metal Commit | TBD (requires a tt-metal build containing `models/experimental/xtts_v2`) |
| Docker Image | TBD |

## Parallelism

The model is single-chip, so a BH QuietBox 2 (2x P300 cards = 4 Blackhole chips) is used
**data-parallel**: the server opens one worker per chip (`device_ids = "(0),(1),(2),(3)"`),
each holding its own full copy of the model on a `(1,1)` mesh, and the scheduler hands each
incoming request to a free worker. Four requests are therefore synthesized concurrently at
single-chip latency; a single request is no faster than on N150.

Each worker sees exactly one chip via `TT_VISIBLE_DEVICES`, which makes tt-metal classify the
board as a CUSTOM cluster; `TT_MESH_GRAPH_DESC_PATH` is set to the single-Blackhole-chip
descriptor for that reason (handled automatically in `utils/runner_utils.py`, same as
speecht5).

## Model-specific configuration

| Env var | Required | Purpose |
|---------|----------|---------|
| `XTTS_REF_AUDIO` | no | Reference voice clip the server clones at warmup (a `.wav`, or a torch-saved tensor `.pt` @ 22050 Hz). Defaults to the English sample voice shipped in the coqui/XTTS-v2 HF repo (`samples/en_sample.wav`). For voice cloning, use a clean, mono clip of **~6 seconds** — short or noisy clips audibly degrade output. |
| `XTTS_CKPT` | no | Explicit path to `model.pth`. Defaults to the server's downloaded weights, else the model fetches from HF hub. |

Notes:

- **License:** the XTTS-v2 checkpoint is distributed under the [Coqui Public Model License](https://coqui.ai/cpml) (non-commercial). Downloading implies acceptance (`COQUI_TOS_AGREED`). Review before production use.
- **Languages:** the request's optional `language` field (default `"en"`) selects one of the
  17 languages the pipeline supports: `ar cs de en es fr hi hu it ja ko nl pl pt ru tr zh`.
  Region variants normalize to their base code (`pt-br` → `pt`, `zh-cn` → `zh`); an
  unsupported code is rejected at the API (422). The ja/ko/zh romanizers
  (`cutlet`/`hangul-romanize`/`pypinyin`) ship in the image and load on a language's
  first request. The reference voice is cloned across languages — see `XTTS_REF_AUDIO`.
- Long request texts are split at sentence boundaries (including CJK `。！？`) and
  synthesized per chunk with a short stitched pause. The per-chunk budget follows coqui's
  per-language character limits (en 240, ja 71, zh 82, ko 95, …), so non-Latin scripts
  produce more, shorter chunks — one request may take several seconds per ~10 s of audio.
- Fixing the request's `seed` makes identical text reproduce identical audio; omitting it draws randomly per request.
- Voice cloning is fixed to the `XTTS_REF_AUDIO` voice at warmup; per-request `speaker_id`/`speaker_embedding` are not yet supported.
- All four workers warm up in parallel on first start (each JIT-compiles into its own
  `built/<device_id>` cache), so first-start warmup is CPU-bound and slower than a single-chip
  start; subsequent starts hit the caches.
