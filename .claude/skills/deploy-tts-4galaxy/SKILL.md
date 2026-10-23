---
name: deploy-tts-4galaxy
description: Deploy the Inworld TTS-2 stack across a 4-galaxy Blackhole quad (codec + blaze ring + audio decoders at T=48 + the C++ inference server), and verify it actually serves audio. Use when asked to deploy, redeploy, bring up, or tear down TTS-2 on a quad such as bh-glx-110-d0Xu[02,08,14,20]; when a deploy hangs, aborts, or comes up serving 44-byte responses; when changing --n-slots / admission caps; or when adding decoders for more concurrency. Self-contained: every command is inline. Validated 2026-09-24 with image tts_inworld_ai_demo (digest f9cc8137).
---

# Deploy TTS-2 on a 4-Galaxy Quad

Three tiers must come up in order, and each one only works if the one below it is
already healthy:

```
1. codec servers  (audio decoders)        -> on the decoder host
2. blaze ring     (speechlm, 32 layers)   -> across all four hosts
3. inference server (C++, drogon)         -> on the decoder host, in docker
```

The whole procedure is below as copy-paste blocks. **Do not skip the pre-flight or
the audio proof** — those two exist because of specific, repeated failures.

---

## Inputs

```bash
# The quad. Every command derives its host list from this one variable.
H4=bh-glx-110-d05u02,bh-glx-110-d05u08,bh-glx-110-d06u02,bh-glx-110-d06u08
DRIVER=bh-glx-110-d05u08      # where the launcher runs; NOT necessarily stage-0
NSLOTS=800                    # concurrent sessions; caps below MUST match
MQSIZE=4000                   # admission queue

# --- Environment. Defaults to the shared, world-readable ljovanovic tree — the
# --- validated T=48 stack with the 64512 FIFO fix. Just run it. (Override only if
# --- deploying a different tree.) Every later command references these variables.
ENV_SCRIPT=/data/ljovanovic/env_tts2.sh   # sources TT_METAL_HOME, PYTHONPATH, checkpoints, python_env
CACHE=/data/ljovanovic/weight_cache       # weight cache — must be writable by whoever runs this
LOGDIR=/data/ljovanovic/tts_logs          # log dir — must be writable (blaze writes blaze.log here)
MGD=/data/ljovanovic/tt-blaze/tests/pipeline_builder/mesh_graph_descriptors/llama_8b_4galaxy_mesh_graph_descriptor.textproto
IMG=ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:tts_inworld_ai_demo  # needs ghcr login

L=/data/ljovanovic/bench/run_$(date +%m%d_%H%M)   # log dir for this deploy
RING=$L/ring.log
mkdir -p "$L"
```

`DRIVER` is only where `tts_runner` is invoked. **The decoder/stage-0 host is
discovered, never asserted** — see stage 3.

---

## Prerequisite: the decoder H2D FIFO must be batch-aligned (64512)

**This one edit is the difference between a stack that serves and one that wedges
under any load.** Check it before every deploy:

```bash
grep -n "socket_fifo_size_bytes: int" \
  "$(source "$ENV_SCRIPT" >/dev/null 2>&1; echo $TT_METAL_HOME)/models/demos/inworld_tts/tt/decoder_tts2.py"
# must read:  socket_fifo_size_bytes: int = 64512,
# if it says 2048 (or 8192 / 65536), fix it:
f="$(source "$ENV_SCRIPT" >/dev/null 2>&1; echo $TT_METAL_HOME)/models/demos/inworld_tts/tt/decoder_tts2.py"
sed -i 's/        socket_fifo_size_bytes: int = [0-9]*,/        socket_fifo_size_bytes: int = 64512,/' "$f"
```

Why: the engine writes a full 8-page (1536 B) decoder batch in **one** contiguous
H2D socket write, but `H2DSocket::write` does **not** split at the FIFO wrap. At
`--tokens 48` each page is 192 B, so when a batch straddles the end of the FIFO the
tail overruns neighbouring L1 (silent corruption) and the decoder goes idle with work
still queued — neither FIFO ever looks full. **64512 = 42 × 1536 is an exact multiple
of one batch, so a batch write can never straddle the wrap.** Any non-multiple (2048,
8192, 65536) only moves *where* the first crossing lands: 2048 wedges on chunk ~2
(the classic 44-byte / chunk-4 stall), 65536 survives ~42 chunks then wedges under
sustained load. Do **not** raise it past ~128 KB — the FIFO is L1-backed and 256 KB
crashes the decoder. `T=32` (the old default) hides the bug because 128 B pages divide
every power-of-two FIFO — but this engine requires `--tokens 48`.

No image rebuild is needed: the fix is decoder-side, and the media-server image reads
the FIFO size from the descriptor the decoder publishes.

---

## 0. Guard: do you hold these nodes?

```bash
squeue -u "$USER" -o "%i %N" 2>/dev/null
```

A Slurm allocation is the normal proof. Nodes can also be held by `drain` (an
admin-requested hold), which `squeue` does not show — if the nodes are drain-held
by you, proceed; `salloc` on them will fail with `Required node not available`.

`sinfo` may die with `Can't find plugin for select/graph` on a compute host — that
plugin is a custom one absent from some nodes. `squeue` still works.

---

## 1. Teardown (processes only — shm comes later)

```bash
cat > /tmp/_td.sh <<'EOF'
#!/usr/bin/env bash
# The host list lives INSIDE this file on purpose. See the TRAP below.
HOSTS="bh-glx-110-d05u02 bh-glx-110-d05u08 bh-glx-110-d06u02 bh-glx-110-d06u08"
A='[b]laze.models.cli'; B='[p]rterun'; C='[t]ts_runner'; D='[g]enerate_rank_bindings'
for h in $HOSTS; do
  timeout 60 ssh -o StrictHostKeyChecking=no "$h" \
    "pkill -TERM -f '$A'; pkill -TERM -f '$B'; pkill -TERM -f '$C'; pkill -TERM -f '$D'" 2>/dev/null
done
sleep 5
for h in $HOSTS; do
  timeout 30 ssh -o StrictHostKeyChecking=no "$h" \
    "kill -9 \$(pgrep -f '$A|$B|$C|$D') 2>/dev/null" 2>/dev/null
  timeout 30 ssh -o StrictHostKeyChecking=no "$h" 'docker rm -f tt-cpp-worker >/dev/null 2>&1' 2>/dev/null
done
n=0
for h in $HOSTS; do
  n=$((n + $(timeout 20 ssh -o StrictHostKeyChecking=no "$h" "pgrep -cf '$A|$C|$D'" 2>/dev/null || echo 0)))
done
echo "torn down (ranks=$n)"
EOF
bash /tmp/_td.sh
```

> **TRAP — `pkill -f` kills the caller.** If the host you are working from is *in*
> the quad, that remote `pkill -f '[t]ts_runner'` also matches your own shell's
> `argv`, which contains the pattern text. The shell dies with **exit 144** mid-script.
> Putting the host list and patterns in a *file* keeps them out of the caller's argv.
> The same applies to any `pgrep`/`pkill` you type by hand — anchor on a cmdline
> prefix (`case "$cmd" in "python3 foo.py"*)`) or use `[f]oo` bracket forms.

> **TRAP — a hardcoded host list silently tears down nothing.** If the list points at
> a different (or dead) quad, this prints `torn down (ranks=0)` while leaving live
> ranks holding `CHIP_IN_USE`, and stage 3 then fails validation with
> `leftover processes are still holding chips`. Always derive it from `$H4`.

Verify, with patterns that cannot match this command:

```bash
for h in ${H4//,/ }; do
  echo "  $h: $(ssh -n -o BatchMode=yes "$h" 'pgrep -cf "bl""aze.models.cli|tt""s_runner|prt""erun"' 2>/dev/null) procs"
done
```

---

## 2. Recover hosts (a warm reset, not `tt-smi -glx_reset`)

```bash
ssh -o BatchMode=yes "$DRIVER" "bash -lic 'export HOSTS=$H4; recover-hosts'" 2>&1 \
  | grep -iE "Recovery succeeded|Reset completed|stress test|FAIL|error" | tail -8
```

Expect `Reset completed successfully` per host, then `Recovery succeeded on attempt 1 of 1`.

---

## 2b. Clear shm — **after** recovery, never before

```bash
for h in ${H4//,/ }; do
  ssh -o BatchMode=yes "$h" 'sudo -n /usr/local/bin/tt_rm_shm.sh >/dev/null 2>&1' 2>/dev/null
  n=$(ssh -o BatchMode=yes "$h" 'ls /dev/shm/ 2>/dev/null | grep -ciE "tt_|UMD"' 2>/dev/null | tail -1)
  echo "  $h shm_remaining=$n"
  [ "${n:-0}" -ne 0 ] && { echo "  !! shm not clean on $h -- abort"; exit 6; }
done
```

**Order matters.** `recover-hosts` itself takes `CHIP_IN_USE` locks while resetting;
any it leaves behind outlive it. Clearing shm first means its own leftovers survive,
and blaze then waits forever on locks owned by PIDs that no longer exist.

The segments are **root-owned** (the container creates them), so a plain `rm` fails
with permission denied — `tt_rm_shm.sh` is one of the five NOPASSWD sudo helpers and
is required here. Leftovers are large: `tt_tts_task_queue` alone is ~2.1 GB.

---

## 2c. Sysmem pre-flight — **the single most valuable gate**

```bash
cat > /tmp/_probe.sh <<'PROBE'
source /data/ljovanovic/env_tts2.sh >/dev/null 2>&1
cd "$TT_METAL_HOME" || exit 9
TT_VISIBLE_DEVICES=0 timeout 200 python -c "
import ttnn
d = ttnn.open_mesh_device(physical_device_ids=[])
print('HEALTHY')
ttnn.close_mesh_device(d)
" 2>&1 | grep -oiE "HEALTHY|unexpected NOC address|No such device" | head -1
PROBE

for h in ${H4//,/ }; do
  r=""
  for try in 1 2; do
    r=$(timeout 240 ssh -o BatchMode=yes "$h" 'bash -s' < /tmp/_probe.sh 2>&1 | tail -1)
    [ -n "$r" ] && break
    echo "  $h -> no result (try $try), retrying"
  done
  echo "  $h -> ${r:-NO OUTPUT}"
  [ "$r" = "HEALTHY" ] || { echo "  !! $h cannot open a device -- sysmem stranded"; exit 7; }
done
```

`Sysmem mapped at unexpected NOC address` was the single root cause of ~6 failed
deploys on one quad. **A warm reset does not clear it** — the host needs a power
cycle or admin intervention. Sixty seconds here saves twenty minutes of a deploy
that cannot possibly succeed.

> **TRAP — do not shorten the probe timeout.** Device-open takes ~30 s on some hosts
> and ~90 s on others. A short timeout turns a slow-but-healthy host into a fake
> failure; that aborted a perfectly good quad once. Retry on *empty* output; fail
> only on a real error string.

---

## 3. Ring: codec + blaze + 2 decoders

```bash
rm -f "$RING"
ssh -o BatchMode=yes "$DRIVER" '
  for v in $(env | grep -oE "^SLURM_[A-Z_]+"); do unset $v; done
  source '"$ENV_SCRIPT"' && cd "$TT_METAL_HOME"
  nohup python -u -m models.demos.inworld_tts.tts_runner --model inworld_tts \
    --model-path "$TTS2_SPEECHLM_PATH" \
    --hosts '"$H4"' \
    --mgd '"$MGD"' \
    --dec-chip 24:48 --dec-chip 25:48 \
    --cache-path '"$CACHE"' \
    --env-script '"$ENV_SCRIPT"' \
    --logdir '"$LOGDIR"' \
    --n-slots '"$NSLOTS"' --launch-only > '"$RING"' 2>&1 &
  echo "  ring launched pid $!"'
```

`--dec-chip 24:48 --dec-chip 25:48` gives **two** audio decoders **at wire size
T=48** — the `:48` is mandatory. This engine (`tt_llm_engine` b4e10f8, in the
`tts_inworld_ai_demo` image) hard-codes `WIRE_CODES_PER_CHUNK = 48` and writes
192-byte pages; a decoder launched at the tt-metal default `T=32` expects 128-byte
pages and **hangs the shm socket rather than raising**. Both decoders must be `:48`.
`--launch-only` keeps codec + blaze alive after the runner returns, which is what lets
you re-run only the later stages if something downstream fails.

Confirm each decoder came up at 48 after the ring is ready:

```bash
ssh -o BatchMode=yes "$ISHOST" 'pgrep -af "[t]ts_runner --model inworld_tts_decoder" | grep -oE "\-\-tokens [0-9]+"'
# expect: --tokens 48   (one line per decoder)
```

**Adding a 3rd+ decoder for more concurrency** (blaze stays up; measured to raise
max-C from ~576 to ~640 on this quad): stop the server, then launch one extra
decoder per free chip with a unique socket suffix, and add its pair + chip to the
server env (stage 4):

```bash
# free decoder chips are listed as "free chips there: [24, 25, 26, ...]" in $RING
ssh -o BatchMode=yes "$ISHOST" 'source  >/dev/null 2>&1; cd "$TT_METAL_HOME";
  TT_VISIBLE_DEVICES=26 nohup python -u -m models.demos.inworld_tts.tts_runner \
    --model inworld_tts_decoder --model-path "$TTS2_CODEC_DECODER_CKPT" \
    --socket-suffix _3 --tokens 48 > /tmp/dec_3.log 2>&1 & echo pid $!'
# then in stage 4 add  tts2_decoder_h2d_3:tts2_decoder_d2h_3  to DECODER_SOCKET_PAIRS
# and 26 to DEVICE_IDS.  Suffix _2 -> chip 25, _3 -> chip 26, _4 -> chip 27, ...
```

Note the load balancer (`pick_child`, fewest-outstanding) skews toward decoder0, so
extra decoders are underused until decoder0 saturates — diminishing returns past ~3.

Then wait, watching **the work** and not the poller:

```bash
STALE=0; OLD=""; BLOG="$LOGDIR/blaze.log"
for i in $(seq 1 240); do
  grep -q "TTS_RUNNER_READY v1 component=inworld_tts " "$RING" 2>/dev/null && { echo "  ring READY"; break; }
  if grep -qiE "VALIDATION FAILED|Traceback|TT_FATAL|TT_THROW|out of memory|Address already in use" "$RING" 2>/dev/null; then
    grep -iE "VALIDATION FAILED|Traceback|TT_FATAL|TT_THROW|Error" -A6 "$RING" | tail -20; exit 1; fi
  # blaze.log does not exist during Phase 1 pre-warm -- fall back to ring.log until it does
  if [ -f "$BLOG" ]; then WATCH="$BLOG"; else WATCH="$RING"; fi
  NEW=$(stat -c%s "$WATCH" 2>/dev/null || echo 0)
  [ "$NEW" = "$OLD" ] && STALE=$((STALE+1)) || STALE=0
  OLD=$NEW
  if grep -q "Waiting for lock 'CHIP_IN_USE" "$BLOG" 2>/dev/null; then
    for pid in $(grep -oE "PID: [0-9]+" "$BLOG" | awk '{print $2}' | sort -u); do
      alive=0
      for h in ${H4//,/ }; do ssh -o BatchMode=yes "$h" "kill -0 $pid" 2>/dev/null && alive=1; done
      [ $alive -eq 0 ] && { echo "  !! CHIP_IN_USE held by DEAD pid $pid -- will wait forever"; exit 5; }
    done
  fi
  [ $STALE -ge 90 ] && { echo "  !! $WATCH silent 15 min -- not advancing"; tail -8 "$RING"; exit 1; }
  sleep 10
done
```

> **TRAP — `ring.log` is the poll loop, not the work.** It prints
> `blaze still coming up (Ns)` every ~2 min whether or not the ranks are advancing;
> a 22-minute `CHIP_IN_USE` deadlock sailed straight past a 6-minute staleness check
> on that file. `blaze.log` is written by the ranks themselves.

> **TRAP — but do not key staleness on `blaze.log` from t=0.** It does not exist
> during Phase 1 pre-warm, so the counter ticks against a file that cannot grow yet
> and aborts before blaze has been asked to start. Fall back to `ring.log`.

> **TRAP — 6 minutes of silence is too strict.** Startup varies enormously between
> quads: codec 36 s–354 s, blaze 490 s–660 s. A 6-minute window false-aborted a
> healthy quad whose blaze needed 660 s. Use **15 min** (`STALE >= 90`); a genuine
> `CHIP_IN_USE` deadlock still announces itself within ~90 s via the dead-PID check.

Discover the decoder/stage-0 host — **never assume it positionally**:

```bash
ISHOST=$(grep -oE "descriptors present on bh-glx-110-[a-z0-9]+" "$RING" | tail -1 | awk '{print $NF}')
[ -z "$ISHOST" ] && ISHOST=$(grep -oE "picked bh-glx-110-[a-z0-9]+" "$RING" | tail -1 | awk '{print $2}')
[ -z "$ISHOST" ] && { echo "  !! cannot determine decoder host"; exit 3; }
echo "$ISHOST" > "$L/ishost"; echo "  decoder/stage-0 host = $ISHOST"
ssh -o BatchMode=yes "$ISHOST" 'ls /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* 2>/dev/null | xargs -n1 basename'
```

`tts_runner` picks the host with free chips by reading the placement it actually
generated (`rank_bindings.yaml`). The same positional host that worked on one quad
runs a full passthrough galaxy on another, where `--decoder-host` fails validation.

---

## 4. Inference server (docker, on `$ISHOST`)

```bash
ssh -o BatchMode=yes "$ISHOST" "
docker rm -f tt-cpp-worker >/dev/null 2>&1
docker run -d --name tt-cpp-worker \
  --privileged --ipc=host \
  --ulimit nofile=65536:65536 --ulimit nproc=65536:65536 \
  -p 8010:8000 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /etc/udev/rules.d:/etc/udev/rules.d \
  -v /lib/modules:/lib/modules \
  -v /var/run/tenstorrent:/var/run/tenstorrent \
  -v /mnt/models/InWorld-Propritery/tts-models/tts-2/checkpoints/speechlm:/tokenizers/tts:ro \
  -e MODEL_SERVICE=tts \
  -e MODEL_RUNNER_TYPE=tt_tts \
  -e MODEL='meta-llama/Llama-3.1-8B-Instruct' \
  -e TTS_TOKENIZER_PATH=/tokenizers/tts/tokenizer.json \
  -e TTS_ENCODER_ENABLED=0 \
  -e TTS_BOS_TOKEN='<|begin_of_text|>' \
  -e TTS_DECODER_TRANSPORT=socket \
  -e TTS_DECODER_SOCKET_PAIRS='tts2_decoder_h2d:tts2_decoder_d2h,tts2_decoder_h2d_2:tts2_decoder_d2h_2' \
  -e TTS_PAGE_WIDTH=8 \
  -e TTS_PAGE_LINGER_US=500 \
  -e TTS_PAGE_MIN_ROWS=8 \
  -e TTS_PREFILL_CHUNK_SIZE=256 \
  -e TTS_CHUNK_TOKENS_1=17 -e TTS_CHUNK_TOKENS_2=32 -e TTS_CHUNK_TOKENS_3=48 \
  -e TTS_FIRST_CHUNK_TOKENS=17 -e TTS_SECOND_CHUNK_TOKENS=32 -e TTS_CHUNK_TOKENS=48 \
  -e TTS_MAX_NEW_TOKENS=1020 \
  -e TTS_EDF_SAFETY_FACTOR=0.75 \
  -e TTS_MAX_BATCH_SIZE=$NSLOTS \
  -e TTS_MAX_USERS=$NSLOTS \
  -e MAX_IN_FLIGHT_COUNT=$NSLOTS \
  -e MAX_SESSIONS_COUNT=$NSLOTS \
  -e PM_MAX_USERS=$NSLOTS \
  -e MAX_QUEUE_SIZE=$MQSIZE \
  -e DEVICE_IDS='(2,3,6,7,10,11,14,15,24,25)' \
  -e TT_LOG_LEVEL=debug \
  -e TTS_TIMING=1 \
  --entrypoint /bin/bash \
  $IMG \
  -c 'cd cpp_server && ./build/tt_media_server_cpp'
"
```

> **Image: `tts_inworld_ai_demo`** (validated digest `f9cc8137…`, pulled 2026-09-24) —
> **not** `tts_inworld_ai_demo_4_glx`. Pull it first (`docker pull …:tts_inworld_ai_demo`)
> — a cached copy can be weeks stale, and an old build rejected its own config with
> `[Config] TTS_CHUNK_TOKENS must be in [1, 30]`. The `TTS_CHUNK_TOKENS_1/2/3=17/32/48`
> ramp (added above) is required: without it the steady chunk defaults out of range and
> the stack stalls silently.

> **Stop the container with `docker stop -t 30`, never `docker rm -f`.** SIGKILL skips
> the socket destructors and leaves the decoder H2D mid-transfer
> (`Bytes sent: N, acknowledged: M`); the next container then aborts in
> `DecoderSocketPipeline::Impl::Impl` and only a full ring relaunch clears it.

Wait for it and confirm the declared capacity:

```bash
for i in $(seq 1 45); do
  ok=$(ssh -o BatchMode=yes "$ISHOST" "curl -s -o /dev/null -m 240 -w '%{http_code}' -X POST \
    http://localhost:8010/v1/audio/speech -H 'Authorization: Bearer your-secret-key' \
    -H 'Content-Type: application/json' -d '{\"text\": \"Hi there.\"}'" 2>/dev/null)
  [ "$ok" = "200" ] && { echo "  server ready"; break; }
  sleep 10
done
ssh -o BatchMode=yes "$ISHOST" 'curl -s -m 10 http://localhost:8010/max-session-count; echo
  curl -s -m 10 http://localhost:8010/metrics | grep -E "^tt_max_queue_size"
  docker logs tt-cpp-worker 2>&1 | grep -E "capacity=|makeTtsScheduler: decoder" | tail -4'
```

**Check three things:**

1. `{"max_session_count":<NSLOTS>}` — if it says 700 when you asked for 800, the caps
   did not propagate and the extra slots are invisible.
2. **One `makeTtsScheduler: decoder N over shared memory` line per decoder** —
   `decoder 0`, `decoder 1`, (`decoder 2` if you added a third). The count must match
   your `DECODER_SOCKET_PAIRS`. `workers=1` in the `capacity=` line is the TTS service
   worker, *not* the decoder count; two decoders with `workers=1` is correct. Also expect
   `drained 0 + 0 residual D2H pages … clean handoff` — nonzero means poisoned sockets
   (relaunch the ring).
3. `capacity=<NSLOTS>`.

> **TRAP — the six caps must all track `--n-slots`.** `TTS_MAX_BATCH_SIZE`,
> `TTS_MAX_USERS`, `MAX_IN_FLIGHT_COUNT`, `MAX_SESSIONS_COUNT`, `PM_MAX_USERS` and
> blaze's `--n-slots` are separate knobs. Raising `--n-slots` alone gives blaze more
> slots while the server still refuses past the old cap, and load tests shed 429s
> exactly as before. `MAX_QUEUE_SIZE` is separate and does not need to match.

> **TRAP — `DEVICE_IDS` takes LOGICAL chip ids in bracket syntax.** Each bracket pair
> is one worker, and its contents become that worker's `TT_VISIBLE_DEVICES` — which is
> why exporting `TT_VISIBLE_DEVICES` yourself does nothing. The list must include the
> speechlm mesh chips *and* every decoder chip. The decoder logs
> `Opening local chip ids/PCIe ids: {0}/[16]` for `--dec-chip 24` (logical 24 ->
> PCIe 16); passing the PCIe ids here fails with
> `H2D socket connector cannot find physical PCIe device /dev/tenstorrent/16`.

---

## 5. Audio proof — `health=200` is **not** proof of life

```bash
AB=$(ssh -o BatchMode=yes "$ISHOST" 'curl -s -o /dev/null -w "%{size_download}" -m 60 -X POST \
  http://localhost:8010/v1/audio/speech -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"The quick brown fox jumps over the lazy dog.\"}"' 2>/dev/null)
echo "  single request returned ${AB} bytes"
[ "${AB:-0}" -lt 10000 ] && { echo "  !! NO AUDIO (header only) -- stack is wedged"; exit 4; }
echo "########## DEPLOY DONE $(date -Is) ##########"
```

A wedged stack answers `health=200` and returns **44 bytes** — a bare WAV header with
no audio. Expect ~288 KB for that sentence. This check has caught a "successful"
deploy that served nothing.

**Required warmup before ANY measurement — 100 concurrent requests.** `tokenizer_resolve`
is a ~14 s cold cost paid **per drogon IO thread** (64 of them); a request landing on an
un-warmed thread returns a 14 s TTFB that poisons FC p99 and inverts every conclusion. A
short or sequential warmup does not cover the thread pool — fire ~100 concurrent:

```bash
ssh -o BatchMode=yes "$ISHOST" 'for i in $(seq 1 100); do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8010/v1/audio/speech \
    -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
    -d "{\"text\": \"The quick brown fox jumps over the lazy dog.\"}" & done; wait; echo warmed'
```

Only then sweep. On this quad, all graded targets (FC p50 100 / p99 125, SC p99 180,
TC p99 400, 4C+ p99 720) pass to **C≈160 with 1 decoder, ~576 with 2, ~640 with 3**
(ISL=64). Decoder duty at the ceiling is ~100% on decoder0, so the decoders are the
bottleneck — add decoders (above) to push higher.

---

## Sanity-check the deploy before trusting numbers

Closed loop pins occupancy, so **measured C must equal the configured concurrency**.
If `C/N` deviates, the measurement is wrong, not the system:

```bash
python3 - <<'PY'
import json,sys,statistics as st
sys.path.insert(0,"/data/ljovanovic/tt-inference-server/test_module/benchmark_tests")
import tts_load_report as R
d=json.load(open("/data/ljovanovic/bench/run_XXXX/warm.json")); lv={l["conc"]:l for l in d["levels"]}
for r in R.aggregate(d)[0]:
    N=r["conc"]; m=lv[N]
    ok=[x for x in R._arrival_cohort([y for y in d["records"] if y["conc"]==N],
        m["window_start"],m["window_end"]) if x.get("ok")]
    W=st.mean([x["t_end"]-x["t_send"] for x in ok if x.get("t_end")])
    print(f"C={r['C']:.1f} (N={N})  C/N={r['C']/N:.4f}  rps*W/N={r['rps']*W/N:.4f}")
PY
```

`C/N` should read 1.0000 and Little's law (`rps*W/N`) should agree within ~1%. A
3%+ disagreement means the arrival process is not stationary — usually un-staggered
closed-loop workers phase-locking into a convoy.

---

## Teardown when finished

Run stage 1, then clear shm (stage 2b) — leftover segments hold ~2.6 GB and survive
a process-only teardown because they are root-owned.

---

## Failure quick reference

| Symptom | Cause | Action |
|---|---|---|
| `torn down (ranks=0)` but stage 3 says leftover chips | teardown host list ≠ deploy quad | derive `HOSTS` from `$H4`, re-run stage 1 |
| shell dies, **exit 144** | remote `pkill -f` matched your own argv | put patterns in a file; anchor cmdline prefixes |
| `Sysmem mapped at unexpected NOC address` | stranded mapping | **not** fixable by `recover-hosts`; power-cycle / admin |
| `Waiting for lock 'CHIP_IN_USE_0_PCIe'` | leftover ranks, possibly dead PIDs | stage 1 properly scoped, then 2b |
| ring silent, ranks at ~5% CPU | Phase 1 deadlock (looks alive in `ps`) | dead-PID check in stage 3 |
| aborted at ~6 min but blaze was fine | staleness window too short | 15 min; codec 36–354 s, blaze 490–660 s |
| `health=200`, 44-byte responses; or serves a few chunks then stalls under load | decoder H2D FIFO not batch-aligned (FIFO-wrap L1 overrun) | set `socket_fifo_size_bytes = 64512` in `decoder_tts2.py` (Prerequisite section); relaunch ring |
| decoder idle with work queued, `No TTS scheduler output`, neither FIFO full | same FIFO-wrap bug, or decoder launched at `T=32` | 64512 FIFO **and** `--dec-chip N:48` on every decoder |
| `Timeout waiting for device to send acknowledgement over H2D socket. Bytes sent: N, acknowledged: M` | prior container killed with `docker rm -f` (skipped socket destructors) | `docker stop -t 30` next time; full ring relaunch to clear |
| `[Config] TTS_CHUNK_TOKENS must be in [1, 30]` | stale cached image | `docker pull …:tts_inworld_ai_demo` (not `_4_glx`) |
| `max_session_count` < `NSLOTS` | caps did not propagate | all six knobs must track `--n-slots` |
| only one `makeTtsScheduler: decoder` | `--dec-chip` given once | pass it twice; check `DECODER_SOCKET_PAIRS` |
| `cannot find physical PCIe device /dev/tenstorrent/16` | PCIe ids in `DEVICE_IDS` | use LOGICAL ids (24/25, not 16/17) |
| `sinfo: Can't find plugin for select/graph` | custom plugin absent on this host | use `squeue`; ask an admin for node state |
| `salloc: Required node not available (down, drained…)` | nodes drain-held | expected if the hold is yours; skip the Slurm guard |

## Notes that are easy to get wrong

- **Kernel cache needs no clearing after a blaze source change.** `blaze/kernel_codegen.py`
  names each generated kernel `.cpp` by the SHA-256 of its own source, and tt-metal's
  `Kernel::compute_hash()` hashes the source *path* — so an edited `.hpp` yields a new
  filename, a new cache key, and an automatic recompile. Confirm by looking for a new
  `gqa_fused_<hash>` under `~/.cache/tt-metal-cache/*/kernels/`.
- **`${VAR:+X=$VAR}` is not an assignment prefix.** From an expansion, bash parses it
  as a *command name*. Use `env X="$VAR" cmd ...`.
- **`ssh -n` eats stdin.** Drop `-n` for any `ssh host 'bash -s' < file`.
- **tt-metal here is a fork checkout, not a plain submodule.** It sits on branch
  `inworld-tts-on-multi_token_new-pin` (remote `ssinghalTT/tt-metal`) with the
  inworld_tts port, 4x GLX and multi-decoder commits on top of blaze's recorded pin,
  plus uncommitted local edits to `models/demos/inworld_tts/tts_runner.py`.
  **`git submodule update` rewinds all of that** and will refuse rather than clobber
  the runner edits. Do not run it expecting a "clean state" — that state cannot run
  this workload.