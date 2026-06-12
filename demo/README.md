# dots.ocr local endpoint — quickstart

A live, OpenAI-compatible OCR endpoint backed by `dots.ocr` running on Tenstorrent
hardware (e.g. T3K). The vLLM server listens on **port 8000**; a small resize proxy
on **port 8001** letterboxes any image to the model's validated geometry and forwards
to the server. Customer integrations call the proxy on 8001.

All commands below are run from the **tt-inference-server repo root** with a few paths
set once as environment variables (adjust to your machine):

```bash
cd /path/to/tt-inference-server
export TT_METAL_HOME=/path/to/tt-metal          # built tt-metal (has python_env/ and build/lib/)
export VLLM_DIR=$TT_METAL_HOME/vllm             # vLLM source tree (or your own checkout)
export SAMPLE_DOCS=/path/to/sample_docs         # folder of images to OCR
PY=$TT_METAL_HOME/python_env/bin/python         # interpreter with requests + Pillow
```

## One-paragraph step-by-step

**(1)** Start the local vLLM server from the repo root — `python run.py --dev-mode --model dots.ocr --tt-device t3k --local-server --tt-metal-home $TT_METAL_HOME --tt-metal-python-venv-dir $TT_METAL_HOME/python_env --vllm-dir $VLLM_DIR --service-port 8000 --disable-trace-capture --skip-system-sw-validation` and wait until `curl -s http://127.0.0.1:8000/health` returns `200` (first start loads weights and can take a few minutes); **(2)** in a second terminal start the resize proxy so arbitrary-resolution images can't crash the vision tower — `$PY evals/dots_ocr_image_resize_proxy.py` (listens on 8001, forwards to 8000); **(3)** call the endpoint by POSTing an OpenAI Chat Completions request to `http://127.0.0.1:8001/v1/chat/completions` with your image inlined as a base64 `data:` URI in an `image_url` content part (see snippets below); **(4)** to run the full sample set, point the bundled client at your images folder — `$PY demo/dots_ocr_endpoint_demo.py --image-dir $SAMPLE_DOCS --limit 20 --max-tokens 2048` — which sends the first 20 docs through the endpoint one at a time; **(5)** read the results in `demo/demo_outputs/`: per-image transcriptions in `txt/`, structured metrics in `results.json`, and a self-contained visual gallery in `report.html` (open it in a browser to show each document beside its extracted text with latency/throughput).

> Auth note: the demo server is started with auth disabled, so the `Authorization`
> header is optional. If you run the server with auth enabled, pass your key as a
> `Bearer` token (the examples below include it; it is simply ignored when disabled).

## curl (single image)

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

## Python (OpenAI client)

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

## Batch demo over a folder

```bash
$PY demo/dots_ocr_endpoint_demo.py \
    --image-dir $SAMPLE_DOCS \
    --limit 20 \
    --max-tokens 2048
# Outputs -> demo/demo_outputs/{txt/, results.json, report.html}
```

Useful flags: `--prompt "..."` (tailor the instruction, e.g. for tables/key-value
extraction), `--base-url http://127.0.0.1:8000/v1` (hit the server directly, bypassing
the resize proxy), `--limit N`, `--max-tokens N`, `--out DIR`.
