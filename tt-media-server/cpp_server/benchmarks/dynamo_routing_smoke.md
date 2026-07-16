# Dynamo Routing Smoke Test

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
kwargs = {
    "conditional_disagg_enabled": True,
    "conditional_disagg_policy": "isl_or_load",
    "conditional_disagg_eff_isl_threshold": 128,
    "conditional_disagg_eff_isl_ratio_threshold": 0.7,
    "conditional_disagg_prefill_busy_threshold": 0.5,
}
KvRouterConfig(**kwargs)
print("conditional_disagg", kwargs["conditional_disagg_enabled"])
print("policy", kwargs["conditional_disagg_policy"])
print("eff_isl_threshold", kwargs["conditional_disagg_eff_isl_threshold"])
print("prefill_busy_threshold", kwargs["conditional_disagg_prefill_busy_threshold"])
PY
```

Expected: `ai-dynamo-runtime` / `ai-dynamo` import successfully, and the printed
config kwargs show `conditional_disagg True`, policy `isl_or_load`, effective
ISL threshold `128`, and prefill busy threshold `0.5`.

## 2. Start Dynamo Routing Stack

```bash
cd /localdev/ztorlak/tt-inference-server/tt-media-server/cpp_server

benchmarks/run_stack.sh down

DYNAMO_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_REQUEST_PLANE_CODEC=json \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_or_load \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=128 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
  DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD=0.5 \
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

- Decode: `"worker_type":"decode"`, `"model_type":"Chat"`
- Prefill: `"worker_type":"prefill"`, `"model_type":"Prefill"`
- No `Unsupported model configuration` error in `/tmp/tt_frontend.log`
- Frontend should import the pinned upstream conditional-disagg Dynamo from
`DYN_VENV`
- Frontend should use `DYN_REQUEST_PLANE_CODEC=json`; the C++ Dynamo endpoint
expects JSON request-plane payloads.

## 4. Conditional Disaggregation Sanity Check

Dynamo PR `ai-dynamo/dynamo#11357` adds router-owned conditional
disaggregation. This smoke test uses the `isl_or_load` policy so Dynamo may
bypass remote prefill when either the effective ISL is small/cache-hot enough or
the chosen prefill worker is busy. The effective-ISL side still does **not**
bypass every short first-turn prompt; it depends on decode-side KV cache overlap.

Start with conditional disaggregation disabled to confirm the baseline remote
prefill path still works:

```bash
benchmarks/run_stack.sh down

DYNAMO_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_REQUEST_PLANE_CODEC=json \
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
expected for the effective-ISL side of the upstream `isl_or_load` policy.

```bash
benchmarks/run_stack.sh down

DYNAMO_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_REQUEST_PLANE_CODEC=json \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_or_load \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=128 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
  DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD=0.5 \
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
- Remote prefill may still be used for this first-turn prompt unless the
prefill-load side of `isl_or_load` bypasses it.
- No `Segmentation fault`.

Useful log check:

```bash
grep -aE "Unsupported model configuration|Segmentation fault|no workers are alive|reserved decode slot_id|Prefill request|tt_prefill_result|disaggregated_params|accepting Dynamo decode route|x-bypass-remote-prefill|conditional_disagg" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 5. First-Turn Small Request Does Not Prove Bypass

In `mock_pipeline`, Dynamo-routed prefill results do not need a real device-side
reserved decode slot. If Dynamo routes through prefill, decode allocates a
local mock slot before continuing.

This check intentionally does **not** assert direct-to-decode routing. Dynamo's
`isl_bounding` side of `isl_or_load` requires both:

- `effective_isl < DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD`
- `effective_isl / prompt_tokens < DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD`

For a first-turn prompt, decode-side overlap is zero, so `effective_isl` equals
the full prompt and the ratio is `1.0`. With the normal `0.7` ratio threshold,
even a tiny first-turn prompt is expected to go through remote prefill unless
the load side of `isl_or_load` says the selected prefill worker is busy.

```bash
benchmarks/run_stack.sh down

DYNAMO_ROUTING=1 MODEL=deepseek-ai/DeepSeek-R1-0528 \
  DYN_REQUEST_PLANE_CODEC=json \
  DYN_ROUTER_CONDITIONAL_DISAGG=1 \
  DYN_ROUTER_CONDITIONAL_DISAGG_POLICY=isl_or_load \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=256 \
  DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7 \
  DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD=0.5 \
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
- A normal completion.
- Remote prefill is acceptable and expected for this first-turn prompt.
- Direct-to-decode validation is covered by the effective-ISL/cache-overlap
test below, not by this first-turn request.

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

grep -aE "Segmentation fault|reserved decode slot_id|Prefill request|tt_prefill_result|disaggregated_params|accepting Dynamo decode route|Prefix cache" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

Expected: with `isl_or_load`, this large prompt should route through the
prefill worker unless it has enough decode-side cache overlap to make effective
ISL small or the selected prefill worker is considered busy.

## 7. Effective-ISL Bypass With Decode Cache Reuse

This is the main effective-ISL check for the `isl_or_load` policy. The
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

grep -aE "Prefix cache HIT|Prefix cache MISS|reserved decode slot_id|Prefill request|accepting Dynamo decode route|x-bypass-remote-prefill|conditional_disagg|Segmentation fault" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

Expected:

- First request may use remote prefill.
- Second request should prefer decode-local prefill if Dynamo sees enough
decode-side overlap and logs a conditional-disagg bypass.
- The positive direct-to-decode signal is `x-bypass-remote-prefill` in
`/tmp/tt_frontend.log` and no new prefill worker request for the second
request in `/tmp/tt_prefill.log`.
- If `/tmp/tt_frontend.log` reports `KV event routing degraded`, this mock
stack is not currently validating cache-overlap-based bypass; treat that as a
test-environment limitation rather than a functional pass for direct decode.
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
    > "/tmp/dynamo_routing_req_$i.json" &
done
wait

for f in /tmp/dynamo_routing_req_*.json; do
  echo "$f"
  jq '.choices[0].finish_reason? // .error? // .' "$f"
done

grep -aE "Segmentation fault|no workers are alive|reserved decode slot_id|Prefill request|accepting Dynamo decode route" \
  /tmp/tt_frontend.log /tmp/tt_decode.log /tmp/tt_prefill.log
```

## 10. Stop Stack

```bash
benchmarks/run_stack.sh down
```
