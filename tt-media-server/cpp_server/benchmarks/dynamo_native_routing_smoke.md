# Dynamo Native Routing Smoke Test

These commands exercise the local Dynamo frontend plus two `cpp_server`
workers using the CI-style `mock_pipeline` backend. The C++ binary still needs
`./build.sh --blaze` because `mock_pipeline` is registered behind
`ENABLE_BLAZE` on current main.

## 1. Prepare Dynamo Venv

Use this section when testing the local patched Dynamo checkout at
`/localdev/ztorlak/tt-inference-server/dynamo`. It installs both
`ai-dynamo-runtime` and `ai-dynamo` from the local clone into the mock backend
venv.

```bash
export DYN_REPO=/localdev/ztorlak/tt-inference-server/dynamo
export DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv

python3 -m venv "$DYN_VENV"
source "$DYN_VENV/bin/activate"

python -m pip install -U pip maturin patchelf hatchling editables

cd "$DYN_REPO/lib/bindings/python"
python -m maturin develop

cd "$DYN_REPO"
python -m pip install -e . --no-build-isolation --no-deps

python - <<'PY'
from importlib.metadata import version
from dynamo.llm import KvRouterConfig

print("ai-dynamo-runtime", version("ai-dynamo-runtime"))
print("ai-dynamo", version("ai-dynamo"))
c = KvRouterConfig(router_max_local_prefill_length=128)
print("threshold", c.router_max_local_prefill_length)
PY
```

Expected: `ai-dynamo-runtime 1.3.0`, `ai-dynamo 1.3.0`, and `threshold 128`.

## 2. Start Native Dynamo Stack

```bash
cd /localdev/ztorlak/tt-inference-server/tt-media-server/cpp_server

benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH=128 \
  DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv \
  benchmarks/run_stack.sh up

export MODEL=deepseek-ai/DeepSeek-R1-0528
export TARGET=http://127.0.0.1:8080
```

## 3. Verify Health And Registration

```bash
curl -sS http://127.0.0.1:8001/health | jq
curl -sS http://127.0.0.1:8002/health | jq
curl -sS "$TARGET/v1/models" | jq

docker exec etcd etcdctl get --prefix --keys-only v1/instances/
docker exec etcd etcdctl get --prefix v1/mdc/dynamo/ | grep -aE '"worker_type"|"model_type"'
```

Expected registration:

- Decode: `"worker_type":"Decode"`, `"model_type":"Chat"`
- Prefill: `"worker_type":"Prefill"`, `"model_type":"Prefill"`
- No `Unsupported model configuration` error in `/tmp/tt_frontend.log`
- Frontend should import local patched Dynamo from `DYN_VENV`

## 4. Conditional Local Prefill Sanity Check

The patched local Dynamo frontend supports
`DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH`. When set, tokenized prompts at or below
that length should stay on the decode worker and bypass remote prefill.

Start with a high threshold to force local prefill for small prompts:

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH=100000 \
  DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv \
  benchmarks/run_stack.sh up

export MODEL=deepseek-ai/DeepSeek-R1-0528
export TARGET=http://127.0.0.1:8080
```

```bash
curl --max-time 30 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}" | jq
```

Expected:

- Normal completion.
- No new `Native prefill request` in `/tmp/tt_prefill.log` for this request.
- Decode log may show `accepting Dynamo decode route`.

Now force remote prefill by setting the threshold to zero:

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH=0 \
  DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv \
  benchmarks/run_stack.sh up

export MODEL=deepseek-ai/DeepSeek-R1-0528
export TARGET=http://127.0.0.1:8080

curl --max-time 30 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}" | jq
```

Expected:

- Normal completion.
- `Native prefill request` / `tt_prefill_result` activity in prefill logs.
- No `Segmentation fault`.

Useful log check:

```bash
grep -aE "Unsupported model configuration|Segmentation fault|no workers are alive|reserved decode slot_id|Native prefill request|tt_prefill_result|disaggregated_params|accepting Dynamo decode route" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 5. Small ISL Request With Realistic Threshold

In `mock_pipeline`, native prefill results do not need a real device-side
reserved decode slot. If Dynamo routes through prefill, decode allocates a
local mock slot before continuing.

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH=128 \
  DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv \
  benchmarks/run_stack.sh up

export MODEL=deepseek-ai/DeepSeek-R1-0528
export TARGET=http://127.0.0.1:8080

curl --max-time 30 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}" | jq
```

Expected in this mock setup:

- No `Segmentation fault`
- A normal completion if Dynamo routes through prefill or decode-local
- `accepting Dynamo decode route` if Dynamo chose decode-local prefill

## 6. Big ISL Request

```bash
python3 - <<'PY' >/tmp/big_isl.json
import json
prompt = " ".join(["Please summarize this context."] * 1500)
print(json.dumps({
  "model": "deepseek-ai/DeepSeek-R1-0528",
  "messages": [{"role": "user", "content": prompt}],
  "max_tokens": 16
}))
PY

curl --max-time 45 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/big_isl.json | jq

grep -aE "Segmentation fault|reserved decode slot_id|Native prefill request|tt_prefill_result|disaggregated_params|accepting Dynamo decode route|Prefix cache" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

Expected: with `DYN_ROUTER_MAX_LOCAL_PREFILL_LENGTH=128`, this large prompt
should route through the prefill worker.

## 7. Prefix Cache Reuse

```bash
python3 - <<'PY' >/tmp/prefix_a.json
import json
prefix = " ".join(["Shared prefix token block."] * 300)
print(json.dumps({
  "model": "deepseek-ai/DeepSeek-R1-0528",
  "messages": [{"role": "user", "content": prefix + " First question?"}],
  "max_tokens": 8
}))
PY

python3 - <<'PY' >/tmp/prefix_b.json
import json
prefix = " ".join(["Shared prefix token block."] * 300)
print(json.dumps({
  "model": "deepseek-ai/DeepSeek-R1-0528",
  "messages": [{"role": "user", "content": prefix + " Second question?"}],
  "max_tokens": 8
}))
PY

curl --max-time 45 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/prefix_a.json | jq '.usage // .error // .'

curl --max-time 45 -sS "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/prefix_b.json | jq '.usage // .error // .'

grep -aE "Prefix cache HIT|Prefix cache MISS|reserved decode slot_id|Native prefill request|accepting Dynamo decode route|Segmentation fault" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 8. Streaming Request

```bash
curl --max-time 30 -N "$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Stream a short answer.\"}],\"max_tokens\":16}"
```

## 9. Concurrent Requests

```bash
for i in $(seq 1 5); do
  curl --max-time 45 -sS "$TARGET/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello concurrent $i\"}],\"max_tokens\":8}" \
    > "/tmp/native_req_$i.json" &
done
wait

for f in /tmp/native_req_*.json; do
  echo "$f"
  jq '.choices[0].finish_reason? // .error? // .' "$f"
done

grep -aE "Segmentation fault|no workers are alive|reserved decode slot_id|Native prefill request|accepting Dynamo decode route" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 10. Stop Stack

```bash
benchmarks/run_stack.sh down
```
