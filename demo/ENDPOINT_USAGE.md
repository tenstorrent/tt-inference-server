# dots.ocr — Local Endpoint Usage

`dots.ocr` is served on Tenstorrent hardware behind a standard **OpenAI-compatible
Chat Completions API**. Any OpenAI client (or a plain HTTP request) works — point
it at the local endpoint, send an image, get the transcription back.

| | |
|---|---|
| **Base URL** | `http://127.0.0.1:8001/v1` |
| **Endpoint** | `POST /v1/chat/completions` |
| **Model** | `rednote-hilab/dots.ocr` |
| **Auth** | `Authorization: Bearer <api-key>` |
| **Image input** | `image_url` content part — a public URL **or** a `data:image/...;base64,...` data URI |

> The endpoint accepts images at **any resolution** — they are automatically
> resized to the model's validated geometry before inference.

---

## 1. curl

```bash
# Base64-encode a local image into a data URI, then send it.
IMG_B64=$(base64 -w0 sample_docs/sample_doc_01.png)

curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "rednote-hilab/dots.ocr",
    "temperature": 0,
    "max_tokens": 2048,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract all the text content from this image, preserving the reading order."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG_B64"'"}}
      ]
    }]
  }' | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

## 2. Python — OpenAI SDK

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="your-secret-key")

with open("sample_docs/sample_doc_01.png", "rb") as f:
    data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="rednote-hilab/dots.ocr",
    temperature=0,
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all the text content from this image, preserving the reading order."},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }],
)
print(resp.choices[0].message.content)
```

---

## 3. Python — requests (no SDK dependency)

```python
import base64, requests

with open("sample_docs/sample_doc_01.png", "rb") as f:
    data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()

r = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    headers={"Authorization": "Bearer your-secret-key"},
    json={
        "model": "rednote-hilab/dots.ocr",
        "temperature": 0,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all the text content from this image, preserving the reading order."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
    },
    timeout=600,
)
print(r.json()["choices"][0]["message"]["content"])
```

---

## Batch demo

To run the model over a folder of images and produce a TXT/JSON/HTML report:

```bash
"$TT_METAL_HOME"/python_env/bin/python demo/dots_ocr_endpoint_demo.py \
    --image-dir /path/to/sample_docs --limit 20 --max-tokens 2048
```

Outputs land in `demo/demo_outputs/` (`txt/`, `results.json`, `report.html`).

## Notes

- **`max_tokens`** caps the transcription length. Large multi-column / multi-table
  pages may need a higher value; raising it increases latency.
- **`temperature: 0`** gives deterministic, repeatable output (recommended for OCR).
- A `finish_reason` of `"length"` means the cap was hit (output truncated); `"stop"`
  means the model finished the page on its own.
