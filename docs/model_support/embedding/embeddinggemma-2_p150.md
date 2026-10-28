# embeddinggemma-2 Tenstorrent Support on P150

#### Useful links

- [P150 details](https://tenstorrent.com/hardware/blackhole)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`embeddinggemma-2` is a dev-catalog model: it has no release pins (tt-metal commit, docker image) yet, so it runs with `--dev-mode` and an image you build or pick yourself. The release process generates the pinned page and adds the model to the tables in [README.md](./README.md) and [models_by_hardware.md](../models_by_hardware.md).

## Quickstart - Deploy embeddinggemma-2 Inference Server on P150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by the [tt-media-server](../../../tt-media-server/README.md) inference engine (cpp_server backend, `MODEL_RUNNER_TYPE=tt_embedding_gemma2`).

The weights are gated on Hugging Face: accept the license for [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and provide `HF_TOKEN`.

**via run.py command**

```bash
python3 run.py --model google/embeddinggemma-2 --device p150 --workflow server --docker-server \
    --dev-mode --override-docker-image <tt-media-inference-server image with a tt-metal that has models/demos/embedding_gemma2>
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

**Example request**

```bash
curl -X POST 'http://127.0.0.1:8000/v1/embeddings' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/embeddinggemma-2",
    "input": ["task: search result | query: What causes the northern lights?",
              "title: none | text: The northern lights are caused by charged particles from the sun."]
  }'
```

Each embedding has 768 values and unit L2 norm. It equals `SentenceTransformer("google/embeddinggemma-2").encode(text)` for the same string. The server does not add a task prompt. Prepend one yourself, as in the example. The prompts are listed in the checkpoint's `config_sentence_transformers.json`, for example `task: search result | query: ` for queries and `title: none | text: ` for documents. Matryoshka truncation (512, 256 or 128 dimensions) is not exposed by the server. To use it, keep the first values and normalize them again on the client.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), revision `914f7f89142e33e77833254d9c9b90c3cef7303b` (pinned by the model; `EMBEDDING_GEMMA2_REVISION` overrides it) |
| Model Status | 🛠️ Experimental |
| Modalities served | text (the TT model also embeds image, audio and video; the server API carries text only) |
| Embedding Dimension | 768 |
| Max Sequence Length | 8192 tokens (the model supports buckets up to 65536) |
| Max Batch Size | 32 requests per forward, split so that batch x sequence bucket <= 8192 tokens |
| Implementation Code | [embedding-gemma2](https://github.com/tenstorrent/tt-metal/tree/gtobarTT/embedding_gemma2_bringup/models/demos/embedding_gemma2) (tt-metal branch `gtobarTT/embedding_gemma2_bringup` until it merges) |
| tt-metal Commit | not pinned (dev catalog) |
| Docker Image | not pinned (dev catalog) |

## Serving behavior

- **Batching.** A batch runs at batch size 1, 8 or 32 and at the smallest sequence bucket (32, 64, ..., 8192) that holds its longest input. A shape is used when the batch is 1 or batch x bucket is at most 8192 tokens. A larger batch is split into chunks.
- **Warmup.** The model captures one trace per served shape (19 shapes) during warmup. On the first start, warmup also builds the ttnn weight cache under `EMBEDDING_GEMMA2_CACHE`, which the spec puts on the `cache_root` volume. The warmup timeout is 30 minutes (`EMBEDDING_WARMUP_TIMEOUT_MS`).
- **Precision.** The model's selected precision config sets the precision (`doc/datatype_sweep/selected_precision_config.json` in the model folder). `EMBEDDING_GEMMA2_PRECISION` overrides it.
- **Overrides** (model environment variables):

  | Variable | Effect |
  |---|---|
  | `EMBEDDING_GEMMA2_SERVE_TRACE=0` | Runs eager forwards and captures no traces. |
  | `EMBEDDING_GEMMA2_SERVE_BATCHES` | Sets the batch ladder (default `1,8,32`). |
  | `EMBEDDING_GEMMA2_SERVE_MAX_TOKENS` | Sets the batch x bucket limit (default `8192`). |

## Device performance (model only)

Measured on one P150a during the tt-metal bring-up, with traced forwards, inputs resident on the device, and the selected precision config. Server latency adds tokenization, batching, and the HTTP/IPC path.

| Shape (batch x tokens) | Forward p50 |
|---|---|
| 1 x 512 | 6.87 ms |
| 8 x 512 | 24.3 ms |
| 32 x 128 | 22.2 ms |
| 1 x 8192 | 93.0 ms |
