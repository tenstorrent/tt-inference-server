# Dynamo Native Routing Smoke Test

These commands exercise the local Dynamo frontend plus two `cpp_server`
workers using the CI-style `mock_pipeline` backend. The C++ binary still needs
`./build.sh --blaze` because `mock_pipeline` is registered behind
`ENABLE_BLAZE` on current main.

## 1. Prepare Dynamo Venv

Use this section when testing Dynamo's upstream conditional-disaggregation PR.
The local `/localdev/ztorlak/tt-inference-server/dynamo` checkout is not needed;
these commands clone the upstream repo to `/tmp`, check out the pinned PR commit,
and install both `ai-dynamo-runtime` and `ai-dynamo` into the mock backend venv.

```bash
export DYN_COMMIT=f022c6b228901b1785dabde1b579abf32949b258
export DYN_SRC=/tmp/dynamo-conditional-disagg-$DYN_COMMIT
export DYN_VENV=/localdev/ztorlak/tt-inference-server/dynamo-mock-backend/.venv

if [ ! -d "$DYN_SRC/.git" ]; then
  git clone https://github.com/ai-dynamo/dynamo.git "$DYN_SRC"
fi

cd "$DYN_SRC"
git fetch origin pull/11357/head:conditional-disagg-pr
git checkout "$DYN_COMMIT"

python3 -m venv "$DYN_VENV"
source "$DYN_VENV/bin/activate"

python -m pip install -U pip maturin patchelf hatchling editables

cd "$DYN_SRC/lib/bindings/python"
python -m maturin develop

cd "$DYN_SRC"
python -m pip install -e . --no-build-isolation --no-deps

python - <<'PY'
from importlib.metadata import version
from dynamo.llm import KvRouterConfig

print("ai-dynamo-runtime", version("ai-dynamo-runtime"))
print("ai-dynamo", version("ai-dynamo"))
c = KvRouterConfig(
    conditional_disagg_enabled=True,
    conditional_disagg_policy="isl_bounding",
    conditional_disagg_eff_isl_threshold=128,
    conditional_disagg_eff_isl_ratio_threshold=0.7,
)
print("conditional_disagg", c.conditional_disagg_enabled)
print("policy", c.conditional_disagg_policy)
print("eff_isl_threshold", c.conditional_disagg_eff_isl_threshold)
PY
```

Expected: `ai-dynamo-runtime` / `ai-dynamo` import successfully, and the printed
config shows `conditional_disagg True`, policy `isl_bounding`, and threshold
`128`.

## 2. Start Native Dynamo Stack

```bash
cd /localdev/ztorlak/tt-inference-server/tt-media-server/cpp_server

benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_bounding \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=128 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
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
- Frontend should import the pinned upstream conditional-disagg Dynamo from
  `DYN_VENV`

## 4. Conditional Disaggregation Sanity Check

Dynamo PR `ai-dynamo/dynamo#11357` adds router-owned conditional
disaggregation. The `isl_bounding` policy does **not** bypass every short
first-turn prompt. It bypasses when the effective ISL after decode-side KV cache
overlap is below the absolute threshold and the effective-ISL/prompt ratio is
below the ratio threshold.

Start with conditional disaggregation disabled to confirm the baseline remote
prefill path still works:

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
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
- Remote prefill activity in `/tmp/tt_prefill.log`.
- No `Segmentation fault`.

Now enable conditional disaggregation. A small first-turn prompt may still go
through remote prefill because it has no decode-side cache overlap yet. That is
expected for the upstream `isl_bounding` policy.

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_bounding \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=128 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
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
- Remote prefill may still be used for this first-turn prompt.
- No `Segmentation fault`.

Useful log check:

```bash
grep -aE "Unsupported model configuration|Segmentation fault|no workers are alive|reserved decode slot_id|Native prefill request|tt_prefill_result|disaggregated_params|accepting Dynamo decode route|x-bypass-remote-prefill|conditional_disagg" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 5. First-Turn Small Request Under Conditional Disagg

In `mock_pipeline`, native prefill results do not need a real device-side
reserved decode slot. If Dynamo routes through prefill, decode allocates a
local mock slot before continuing.

```bash
benchmarks/run_stack.sh down

DYNAMO_NATIVE_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_bounding \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=256 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
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
- A normal completion if Dynamo routes through prefill or decode-local.
- This first-turn request may still route through prefill because it has no
  decode-cache overlap.

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

Expected: with `isl_bounding`, this large prompt should route through the
prefill worker unless it has enough decode-side cache overlap to make effective
ISL small.

## 7. Effective-ISL Bypass With Decode Cache Reuse

This is the main conditional-disagg check for the `isl_bounding` policy. The
first request establishes decode-side KV cache state. The second request reuses
the long prefix, so its effective ISL should be much smaller than its full prompt
length and may bypass remote prefill.

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

grep -aE "Prefix cache HIT|Prefix cache MISS|reserved decode slot_id|Native prefill request|accepting Dynamo decode route|x-bypass-remote-prefill|conditional_disagg|Segmentation fault" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

Expected:

- First request may use remote prefill.
- Second request should prefer decode-local prefill if Dynamo sees enough
  decode-side overlap and logs a conditional-disagg bypass.
- No `Segmentation fault`.

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
