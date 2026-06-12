# dots.ocr local endpoint — quickstart

A live, OpenAI-compatible OCR endpoint backed by `dots.ocr` running on Tenstorrent
hardware (e.g. T3K). The vLLM server listens on **port 8000**; a small resize proxy
on **port 8001** letterboxes any image to the model's validated geometry and forwards
to the server. Customer integrations call the proxy on 8001.

## Architecture at a glance

```
your client ──► resize proxy (:8001) ──► vLLM server (:8000) ──► dots.ocr on TT hardware
                letterboxes images to
                the validated geometry
```

You talk to the **proxy on 8001**; it normalizes image resolution and forwards to the
**vLLM server on 8000**. Hitting 8000 directly works too, but sending an
arbitrary-resolution image straight to the server can crash the vision tower.

## Prerequisites

- A built **tt-metal** checkout (contains `python_env/` and `build/lib/`).
- Tenstorrent hardware available (this guide assumes a **T3K**).
- A folder of images to OCR (PNG/JPG).

## Step 0 — Set paths (run once per shell)

All commands below are run from the **tt-inference-server repo root**. Adjust the
paths to your machine:

```bash
cd /path/to/tt-inference-server
export TT_METAL_HOME=/path/to/tt-metal          # built tt-metal (has python_env/ and build/lib/)
export VLLM_DIR=$TT_METAL_HOME/vllm             # vLLM source tree (or your own checkout)
export SAMPLE_DOCS=/path/to/sample_docs         # folder of images to OCR
PY=$TT_METAL_HOME/python_env/bin/python         # interpreter with requests + Pillow
```

## Step 1 — Start the vLLM server (terminal 1)

From the repo root, launch the local server on **port 8000**:

```bash
python run.py \
    --dev-mode \
    --model dots.ocr \
    --tt-device t3k \
    --local-server \
    --tt-metal-home $TT_METAL_HOME \
    --tt-metal-python-venv-dir $TT_METAL_HOME/python_env \
    --vllm-dir $VLLM_DIR \
    --service-port 8000 \
    --disable-trace-capture \
    --skip-system-sw-validation
```

The first start loads weights and can take **a few minutes**. Wait until the health
check returns `200`:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
# 200  -> ready
```

> **Auth note:** this command starts the server with auth **disabled**, so the
> `Authorization` header is optional. If you enable auth, pass your key as a
> `Bearer` token — the examples below include one, and it is simply ignored when
> auth is disabled.

## Step 2 — Start the resize proxy (terminal 2)

In a **second terminal** (re-run Step 0 there first), start the proxy. It listens on
**8001** and forwards to the server on **8000**:

```bash
$PY evals/dots_ocr_image_resize_proxy.py
# listening on :8001 -> upstream http://127.0.0.1:8000
```

Override the defaults with the `PROXY_PORT` / `UPSTREAM_URL` environment variables if
your ports differ.

## Step 3 — Send one image (smoke test)

Confirm the endpoint works end-to-end with a single request. Use either the curl or
Python snippet in the [sections below](#curl-single-image), pointed at
`http://127.0.0.1:8001/v1/chat/completions`. A successful call prints the extracted
text. (For the full API reference, see [`ENDPOINT_USAGE.md`](ENDPOINT_USAGE.md).)

## Step 4 — Run the batch demo over a folder

Point the bundled client at your images folder. It sends each document through the
endpoint one at a time:

```bash
$PY demo/dots_ocr_endpoint_demo.py \
    --image-dir $SAMPLE_DOCS \
    --limit 20 \
    --max-tokens 2048
```

## Step 5 — Read the results

The batch demo writes everything to `demo/demo_outputs/`:

| Output | What it is |
|---|---|
| `txt/` | per-image transcriptions, one `.txt` per document |
| `results.json` | structured metrics (latency, throughput, token counts) |
| `report.html` | self-contained visual gallery — open in a browser to see each document beside its extracted text |

---

## Reference: call the endpoint directly

These are the request snippets referenced in **Step 3**. For the full API reference
(headers, image-input formats, `max_tokens`/`temperature`/`finish_reason` notes), see
[`ENDPOINT_USAGE.md`](ENDPOINT_USAGE.md).

### curl (single image)

```bash
IMG=$SAMPLE_DOCS/sample_doc_01.png
DATA_URI="data:image/png;base64,$(base64 -w0 "$IMG")"
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d "{\"model\":\"rednote-hilab/dots.ocr\",\"temperature\":0,\"max_tokens\":2048,
       \"messages\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"text\",\"text\":\"Extract all the text content from this image.\"},
         {\"type\":\"image_url\",\"image_url\":{\"url\":\"$DATA_URI\"}}]}]}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### Python (OpenAI client)

```python
import base64, os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")
img_path = os.path.join(os.environ["SAMPLE_DOCS"], "sample_doc_01.png")
b64 = base64.b64encode(open(img_path, "rb").read()).decode()
resp = client.chat.completions.create(
    model="rednote-hilab/dots.ocr",
    temperature=0,
    max_tokens=2048,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Extract all the text content from this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

## Batch demo flags (Step 4)

The `demo/dots_ocr_endpoint_demo.py` client (used in Step 4) accepts:

| Flag | Purpose | Default |
|---|---|---|
| `--image-dir DIR` | folder of images to OCR | `$SAMPLE_DOCS` |
| `--limit N` | only process the first N images | `20` |
| `--max-tokens N` | cap transcription length (higher = slower) | `2048` |
| `--prompt "..."` | tailor the instruction (e.g. tables / key-value extraction) | extract-all-text |
| `--base-url URL` | hit the server directly, bypassing the resize proxy | `http://127.0.0.1:8001/v1` |
| `--out DIR` | output directory | `demo/demo_outputs` |

```bash
# Example: extract directly from the server (port 8000) with a custom prompt
$PY demo/dots_ocr_endpoint_demo.py \
    --image-dir $SAMPLE_DOCS \
    --base-url http://127.0.0.1:8000/v1 \
    --prompt "Extract all tables as Markdown." \
    --limit 5
```

## Troubleshooting

- **`curl: (7) Failed to connect` on 8001** — the resize proxy (Step 2) isn't running.
- **Connection refused / non-200 on 8000** — the server (Step 1) is still loading
  weights, or failed to start. Re-check the health endpoint and the server logs.
- **Vision-tower crash / garbled output on odd-sized images** — you're hitting the
  server (8000) directly; send through the proxy (8001) instead.
- **Output truncated** — the response `finish_reason` was `"length"`; raise
  `--max-tokens`.
