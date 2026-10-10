# Gemma 4 31B serving: device test plan

Track 2 validation on the four P150 cards. Everything up to here is tested
without hardware (`README.md`, "Tests"). This plan needs exclusive use of all
four cards: no other tt-lab, tt-metal or benchmark job may run on any of them.

Paths used below:

```sh
SERVE=/home/ttuser/workspace/tt-inference-server-gemma-review   # branch work/gemma31-qb2-serving
LAB=/home/ttuser/workspace/tt-lab-gemma31-serve                 # branch work/gemma31-qb2-serve
PY=/home/ttuser/workspace/tt-inference-server/.venv-ttlab/bin/python
export PYTHONPATH=/home/ttuser/workspace/gemma-reference/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=/home/ttuser/workspace/bringup-logs-gemma31/serving-$(date +%Y%m%d)
mkdir -p $OUT
```

Rules for the whole run:

- Never `pkill -f` (it matches and kills the calling shell). Stop the server by PID.
- If anything hangs (no token for minutes, a worker that never answers), stop that
  one process, then check the cards with `tt-smi -s` before the next run. Do not
  start a new job on a card that a hung kernel may have left in a bad state.
- Run the Ethernet state check before measuring, and after any `tt-smi -r`. If any
  link differs from the cold-boot reference, stop and ask for a power cycle.

## 0. Preconditions

```sh
tt-smi -s > $OUT/tt-smi-before.json                       # four P150, no errors
cd $LAB && python3 tools/gemma31_eth_state_check.py _out/tt-lab | tee $OUT/eth-state.txt
```

## 1. Build and gate the native worker

The pinned `tt-lab-serve-03f582cf` predates the merge with `work/gemma31-qb2`
(`2eb15b4`) and message 3 (`ea8a211`, ignore_eos). Build the serve branch head:

```sh
cd $LAB && git log --oneline -3                           # ea8a211 at the top, or later
uv run --python 3.12 --no-project python make.py --env CXX=g++-12 :build
H=$(git rev-parse --short=8 HEAD)
cp _out/tt-lab /home/ttuser/workspace/bringup-logs-gemma31/tt-lab-serve-$H
sha256sum /home/ttuser/workspace/bringup-logs-gemma31/tt-lab-serve-$H | tee $OUT/binary.sha256
```

Native identity gate: `--serve-batch` against `--serve`, with a cancel and an
ignore_eos request. It must print `"passed": true`, both without and with the
prefix cache:

```sh
cd $LAB
TOK=/home/ttuser/workspace/tt-lab-gemma31/_out/qb2/needle-3968.tokens
python3 tools/gemma31_serve_batch_check.py --binary _out/tt-lab --prompt-file $TOK \
  --lengths 3968,2000,1025,512,129,25,3000 --generate 48 --context 4096 --slots 8 \
  --output $OUT/native-batch-check.json
python3 tools/gemma31_serve_batch_check.py --binary _out/tt-lab --prompt-file $TOK \
  --lengths 3968,2000,1025,512,129,25,3000 --generate 48 --context 4096 --slots 8 --prefix-cache \
  --output $OUT/native-batch-check-apc.json
```

Then point the batch deployment at the new binary (local edit, committed once
this plan passes):

```sh
cd $SERVE
sed -i "s|^TT_LAB_BINARY=.*|TT_LAB_BINARY=/home/ttuser/workspace/bringup-logs-gemma31/tt-lab-serve-$H|; s|^TT_LAB_IGNORE_EOS=.*|TT_LAB_IGNORE_EOS=1|" \
  deploy/tt-lab-gemma31/server-batch.env
```

## 2. CLI reference (server stopped)

One `tt-lab gemma --native-device --tokens FILE -n N` run per prompt, on the exact
token IDs the server builds (chat template, `enable_thinking=false`, no extra
special tokens), with the env file's `TT_GEMMA*` settings. Eight prompts: short
answers, code, a 256-token story, and three ~4K-token prompts that share a
prefix. Two long prompts cover 16K and 28K.

```sh
cd $SERVE
$PY deploy/tt-lab-gemma31/verify_cli_identity.py reference --env deploy/tt-lab-gemma31/server-batch.env \
  --long-tokens 16000 30000 --output $OUT/cli-reference.json 2>&1 | tee $OUT/cli-reference.log
```

Add `--dry-run` first to see the commands without opening a device.

## 3. Start the server on the four cards

```sh
cd $SERVE
nohup bash deploy/tt-lab-gemma31/start.sh deploy/tt-lab-gemma31/server-batch.env \
  > $OUT/server.log 2>&1 &
echo $! > $OUT/server.pid              # start.sh execs Python: this is the server PID
until curl -sf http://127.0.0.1:8002/tt-liveness | grep -q '"model_ready": *true'; do sleep 15; done
curl -s http://127.0.0.1:8002/tt-liveness | tee $OUT/liveness-start.json
grep "batch worker ready" $OUT/server.log         # "context=32768, 8 slots"
```

To stop it: `kill $(cat $OUT/server.pid)`. The native child has a parent-death
signal, so it exits with the server. Confirm with `ps -ef | grep tt-lab-serve`
(read only) before step 2 is ever rerun.

## 4. Greedy identity through the server

`check` sends every reference prompt sequentially (plain, then streamed), then
`--repeat` copies of each concurrently (mixed streamed and plain), and requires
the text and `usage.completion_tokens` to equal the CLI's. It also counts
prefix-cache hits from the worker's `reused=N` lines in the server log.

```sh
cd $SERVE
$PY deploy/tt-lab-gemma31/verify_cli_identity.py check --reference $OUT/cli-reference.json \
  --concurrency 8 --repeat 2 --server-log $OUT/server.log --output $OUT/identity-c8.json
$PY deploy/tt-lab-gemma31/verify_cli_identity.py check --reference $OUT/cli-reference.json \
  --concurrency 16 --repeat 3 --server-log $OUT/server.log --output $OUT/identity-c16.json
```

Pass: `"passed": true` for both (all rows match; at least one prefix-cache hit).
At concurrency 16 with 8 slots, half the requests wait for a slot; that tests the
runner's slot accounting under queueing. Repeat once with `server.env` (one
sequence, D64 prefill) for the single-sequence mode.

Cancellation, then recovery:

```sh
for i in 1 2 3 4; do
  curl -s -N --max-time 3 http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 2000."}],"max_tokens":4000,"temperature":0,"stream":true}' \
    > /dev/null &
done; wait
$PY deploy/tt-lab-gemma31/verify_cli_identity.py check --reference $OUT/cli-reference.json \
  --concurrency 8 --repeat 1 --output $OUT/identity-after-cancel.json
curl -s http://127.0.0.1:8002/tt-liveness | tee $OUT/liveness-after-cancel.json   # no worker restarts
```

ignore_eos (needs step 1's binary): a request whose answer ends early must still
return exactly `max_tokens`:

```sh
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":64,"temperature":0,"ignore_eos":true}' \
  | tee $OUT/ignore-eos.json | python3 -c 'import json,sys; u=json.load(sys.stdin)["usage"]; assert u["completion_tokens"]==64, u; print("ok")'
```

## 5. Standard benchmark, concurrency 1 / 8 / 16

The repository's benchmark workflow against the running server.
`benchmark-requirements.json` defines the off-catalog model (`google/gemma-4-31B-it`
on P150X4) and the sweep: ISL 1024 / OSL 128 and ISL 128 / OSL 512, each at
concurrency 1, 8 and 16 (16, 64 and 128 prompts). It sets no targets, so it only
measures. With neither `--local-server` nor `--docker-server`, run.py starts no
server and does no host setup. An explicit `--tt-device` avoids its `tt-smi`
device detection.

```sh
cd $SERVE
python3 run.py --workflow benchmarks --tools vllm \
  --requirements-json deploy/tt-lab-gemma31/benchmark-requirements.json \
  --model google/gemma-4-31B-it --tt-device p150x4 \
  --server-url http://127.0.0.1:8002 --skip-system-sw-validation --no-auth 2>&1 | tee $OUT/bench-vllm.log
python3 run.py --workflow benchmarks --tools aiperf \
  --requirements-json deploy/tt-lab-gemma31/benchmark-requirements.json \
  --model google/gemma-4-31B-it --tt-device p150x4 \
  --server-url http://127.0.0.1:8002 --skip-system-sw-validation --no-auth 2>&1 | tee $OUT/bench-aiperf.log
cp -r workflow_logs $OUT/
```

- `vllm bench serve` sends `ignore_eos: true` on the random dataset. It needs
  `TT_LAB_IGNORE_EOS=1` and step 1's binary; otherwise every request gets HTTP 400.
- aiperf does not send ignore_eos, so its outputs can end before OSL. Report the
  measured output lengths.
- Both fetch their own venvs and the HF tokenizer on first use, so they need network.
- Concurrency 16 with 8 slots measures queueing. For a true 16-way batch, copy
  `server-batch.env` with `TT_GEMMA31_SLOTS=16`, `MAX_NUM_SEQS=16` and
  `MAX_BATCH_SIZE=16`. If 16 slots at 32K do not fit DRAM, lower both
  `MAX_MODEL_LENGTH` and `TT_LAB_CONTEXT` (for example to 8192, which is enough
  for ISL 1024 plus OSL 512). Restart, then rerun only the concurrency-16 points.

Record per point: TTFT (mean/p99), TPOT, output tok/s, request throughput. Put
them next to the CLI single-sequence numbers in tt-lab `docs/gemma31-qb2-*.md`
(about 43 tok/s decode short; b8 batch step about 47 ms at 4K).

## 6. Prefix caching

Step 4 already requires worker-logged hits on the shared-prefix prompts
(`doc-q1..q3`, about 4K tokens each) and identical text, so the cache is exact.
Also compare TTFT: `doc-q2`/`doc-q3` run sequentially after `doc-q1` should show
a much smaller TTFT in the `identity-c8.json` rows (stream mode) than `doc-q1`.

The repository's prefix-cache benchmark (TTFT per scenario against its baseline):

```sh
python3 run.py --workflow benchmarks --prefix-cache --prefix-cache-preset ci \
  --requirements-json deploy/tt-lab-gemma31/benchmark-requirements.json \
  --model google/gemma-4-31B-it --tt-device p150x4 \
  --server-url http://127.0.0.1:8002 --skip-system-sw-validation --no-auth 2>&1 | tee $OUT/bench-prefix.log
grep -c "reused=[1-9]" $OUT/server.log
```

Its hit-rate column will be empty: it reads Prometheus counters
(`tt_prefix_cache_hits/queries` or `vllm:prefix_cache_*`), and tt-media-server
does not export those. The worker's `reused=` lines are the hit evidence.

## 7. Wrap up

```sh
kill $(cat $OUT/server.pid)
tt-smi -s > $OUT/tt-smi-after.json
```

Commit the `server-batch.env` binary update and a short results section in
`VALIDATION.md` (binary SHA, identity pass/fail counts, benchmark table,
prefix-cache hits) on `work/gemma31-qb2-serving`.
