# xtts-v2 Tenstorrent Support on N150

#### Useful links

- [N150 details](https://tenstorrent.com/hardware/wormhole)
- [Search other tts models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`xtts-v2` is also supported on hardware:

- [P150](XTTS-v2_p150.md)
- [BH QuietBox 2](XTTS-v2_p300x2.md)

## Quickstart - Deploy xtts-v2 Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
# --dev-mode is required: the model spec currently lives in the dev catalog (EXPERIMENTAL)
python3 run.py --dev-mode --model XTTS-v2 --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| Model Status | 🟡 Experimental |
| Max Batch Size | 1 |
| Implementation Code | [xtts_v2](https://github.com/tenstorrent/tt-metal/tree/main/models/experimental/xtts_v2) |
| tt-metal Commit | TBD (requires a tt-metal build containing `models/experimental/xtts_v2`) |
| Docker Image | TBD |

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
