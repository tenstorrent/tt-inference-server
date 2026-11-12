---
name: deploy-tts-4galaxy
description: Bare-metal deploy of the Inworld TTS-2 stack on a 4-galaxy Blackhole quad (blaze speechlm ring + 2 audio decoders + the C++ inference server in docker) from two ISOLATED trees - tt-blaze on its own pinned tt-metal, and a separate tt-metal fork checkout for the decoders - then prove it serves audio. Use when asked to build, deploy, redeploy, bring up, or tear down TTS-2 on a quad such as 110-A-Quad2 (bh-glx-110-a0[2,3]u0[2,8]); when a deploy hangs, aborts, or serves 44-byte responses; when changing slot caps or the server image. Self-contained: every command is inline. Validated 2026-11-12 on 110-A-Quad2 with tt-blaze 1721755859 (tt-metal v0.83.0-dev20261105), tt-metal fork 00e3f2137f, image tts_sampling_topk64_penalties.
---

# Deploy TTS-2 on a 4-Galaxy Quad (bare metal, isolated trees)

Three tiers come up in this exact order. Each depends on the one before it:

```
1. audio decoders  (2 x tts_runner --model inworld_tts_decoder, T=48, batch 2)  -> decoder host
2. blaze ring      (35 MPI ranks, tt-run + blaze.models.cli, --n-slots N)       -> all 4 hosts
3. inference server (docker, tt_media_server_cpp)                               -> decoder host
```

Decoders go **before** blaze: they enumerate every chip on their host at start-up,
and if blaze already holds its chips there the decoder deadlocks on `CHIP_IN_USE`.

Validated stack (2026-11-12, 110-A-Quad2):

| piece | commit | where |
|---|---|---|
| tt-blaze | `1721755859` ("Repetition and frequency penalties") | `tenstorrent/tt-blaze` |
| blaze's tt-metal | `860b4917` = tag `v0.83.0-dev20261105` (blaze's own submodule pin) | `tenstorrent/tt-metal` |
| decoder tt-metal | `00e3f2137f` ("on-device per-value mask table") | `ssinghalTT/tt-metal` fork |
| server image | `tt-media-inference-server-blaze:tts_sampling_topk64_penalties` | ghcr `tt-shield` |

Result: audio proof 266,924 B, 100/100 warmup requests OK, warm TTFB ~1 ms (~1.9 s total
for one sentence), `max_session_count` 1000.

---

## What "isolated" means here

Two trees, two environments, and no process ever imports both:

- **blaze side** (`env_blaze.sh`): the launcher and all 35 blaze ranks use
  `tt-blaze/tt-metal/python_env`, `TT_METAL_HOME=tt-blaze/tt-metal`, `PYTHONPATH=tt-blaze`.
  tt-run forwards this interpreter + env to every rank.
- **decoder side** (`env_tts.sh`): the decoders use the fork checkout's own `python_env`
  and `build_Release`.

Never put the fork tree on the blaze side's `PYTHONPATH`: blaze is built against v0.83
and the fork is v0.79-based, so mixing them is an ABI mismatch. The two versions *do*
coexist as separate processes on the decoder host - but only after a fresh
`recover-hosts` (see stage 2).

---

## Inputs

```bash
STACK=/data/ljovanovic/stacks/tts2_b1721755_m00e3f21     # everything for this stack lives here
H4=bh-glx-110-a02u02,bh-glx-110-a02u08,bh-glx-110-a03u02,bh-glx-110-a03u08   # 110-A-Quad2
LAUNCH_HOST=bh-glx-110-a02u02   # where tt-run is started (any quad host)
NSLOTS=1000                     # blaze --n-slots; the server's five caps MUST equal it
DECCHIPS="24 25"                # decoder chips on the decoder host (free per rank_bindings)
IMG=ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:tts_sampling_topk64_penalties
```

**Which hosts make up a quad:** the "Cluster" column of the allocation sheet (spreadsheet
`1HGi57KTOnNwoIeFeDyMEpElrfHdY7AGXK_pwZbRZs98`, weekly tab). E.g.
`110-A-Quad2: 110-A0[2,3]u0[2,8]` = `bh-glx-110-a02u02,a02u08,a03u02,a03u08`. Slurm pod
names do not match the sheet's quad numbers.

**Access:** the quad hosts have no ssh keys of their own. Run everything from the
Mac via `herdr-ssh -A '... ssh -A <host> ...'` so the user's key is lent for the call.
Quad2 is drain-held ("InWorld demo"), so there is no Slurm job; `pam_slurm_adopt`
intermittently answers `Access denied ... you have no active jobs on this node`
anyway - **retry the ssh**, it gets through on the next try.

---

## 0. Build the isolated stack (once per commit pair)

```bash
# --- clone (needs the GitHub key: herdr-ssh -A) ---
mkdir -p $STACK && cd $STACK
git clone git@github.com:tenstorrent/tt-blaze.git tt-blaze
git -C tt-blaze checkout 1721755859256affa97f4e1dedde914a1174b2f6
git -C tt-blaze submodule update --init --recursive tt-metal     # -> 860b4917 (v0.83.0-dev20261105)
git -C tt-blaze submodule status tt-metal                        # verify the pin

git clone --no-checkout git@github.com:ssinghalTT/tt-metal.git tt-metal
cd tt-metal
git remote add upstream git@github.com:tenstorrent/tt-metal.git
git fetch upstream --tags                         # the fork has no release tags
git checkout 00e3f2137f3ca5aad1fbcc88b9c4adb5e6354dba
git submodule update --init --recursive
git tag -d $(git tag -l 'tt-tts-*')               # REQUIRED - see trap below
git describe --abbrev=10 --first-parent           # must print v0.79.0-dev20260903-...
```

> **TRAP - the fork's own tag breaks the CMake version.** tt-metal's `cmake/version.cmake`
> runs `git describe --first-parent`, which picks the fork's annotated tag
> `tt-tts-inworld-0.1.0-beta.1`. That does not parse as a version and configure dies with
> `No VERSION specified for WRITE_BASIC_CONFIG_VERSION_FILE()`. Deleting the tag in the
> local clone (only) makes describe fall back to the upstream `v0.79.0-dev...` tag.

Build both on a quad host (64 cores; ~18 min for tt-metal, ~30 min for blaze). Start each
with `nohup ... &` over ssh and poll the log:

```bash
# blaze + its pinned tt-metal + venv (python bundled so it runs from any host)
cd $STACK/tt-blaze && UV_LINK_MODE=copy ./install.sh --no-pre-commit
tt-metal/python_env/bin/python -c "import ttnn, blaze"

# decoder tt-metal, standalone
cd $STACK/tt-metal && export TT_METAL_HOME=$PWD UV_LINK_MODE=copy
./build_metal.sh --build-type Release
./create_venv.sh --bundle-python --force
python_env/bin/python -m pip install absl-py      # tt_modeling imports absl; not in the venv
python_env/bin/python -c "import ttnn"
```

> **TRAP - never run two builds of the same tree at once.** They corrupt the CPM cache
> (`Target "common" links to: BLAKE3::blake3 but the target was not found`). Fix: move
> `build_Release` and `.cpmcache/blake3` aside and rebuild once. Deleting a large build
> tree on NFS takes many minutes - `mv` it to a trash dir and `rm` that in the background.

---

## 0b. The two environment files

`$STACK/env_blaze.sh` (launcher + blaze ranks):

```bash
STACK=/data/ljovanovic/stacks/tts2_b1721755_m00e3f21
BLAZE_DIR=$STACK/tt-blaze
TTS2_DIR=/mnt/models/InWorld-Propritery/tts-models/tts-2
source "$BLAZE_DIR/tt-metal/python_env/bin/activate"
export TT_METAL_HOME="$BLAZE_DIR/tt-metal"
export TT_METAL_RUNTIME_ROOT="$BLAZE_DIR/tt-metal"
export TTS2_ROOT="$TTS2_DIR"
export TTS2_CODEC_DECODER_CKPT="/data/ssinghal/inworld/tts-models/checkpoints/decoder/model.pt"
export TTS2_SPEECHLM_PATH="/data/ssinghal/inworld/tts-models/checkpoints/speechlm"
export PYTHONPATH="${BLAZE_DIR}:${TTS2_DIR}"     # never the fork tree
export TT_BLAZE_M_TOKENS=8                        # must equal the server's TTS_PAGE_WIDTH
export TT_METAL_ALLOCATOR_MODE_HYBRID=1
export TTS_DECODER_BATCH=2
```

`$STACK/env_tts.sh` (decoders):

```bash
STACK=/data/ljovanovic/stacks/tts2_b1721755_m00e3f21
BLAZE_DIR=$STACK/tt-blaze
METAL_DIR=$STACK/tt-metal
TTS2_DIR=/mnt/models/InWorld-Propritery/tts-models/tts-2
source "$METAL_DIR/python_env/bin/activate"
export TT_METAL_HOME="$METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$METAL_DIR"
export TTS2_ROOT="$TTS2_DIR"
export TTS2_CODEC_DECODER_CKPT="/data/ssinghal/inworld/tts-models/checkpoints/decoder/model.pt"
export TTS2_SPEECHLM_PATH="/data/ssinghal/inworld/tts-models/checkpoints/speechlm"
export PYTHONPATH="${BLAZE_DIR}:${METAL_DIR}:${TTS2_DIR}"
export TT_BLAZE_M_TOKENS=8
export TT_METAL_ALLOCATOR_MODE_HYBRID=1
export TTS_DECODER_BATCH=2                        # must equal the server's TTS_DECODER_BATCH
```

One export per line - `export A=1 B=$A` leaves `B` empty, and an empty
`TT_METAL_RUNTIME_ROOT` trips `TT_FATAL: Root Directory is not set`. Do **not** export
`TT_METAL_SLOW_DISPATCH_MODE` in either file: blaze needs it, the decoders cannot run
under it (`TT_THROW: Not supported for slow dispatch`). It is set only on the blaze launch.

---

## 1. One-time host prep: ssh host keys

tt-run/MPI fans out over ssh with strict host-key checking. Each quad host has its own
`$HOME`, so every host must know the other three, or Phase 1 dies with
`Host key verification failed` / `PRTE has lost communication with a remote daemon`:

```bash
H="${H4//,/ }"
for h in $H; do ssh $h "mkdir -p ~/.ssh; for t in $H; do ssh-keygen -F \$t >/dev/null || ssh-keyscan -T 5 -t ed25519,rsa,ecdsa \$t 2>/dev/null >> ~/.ssh/known_hosts; done"; done
```

---

## 2. Teardown -> recover -> shm -> probe (before EVERY launch)

```bash
cat > $STACK/teardown.sh <<'EOF'
#!/usr/bin/env bash
# Host list + patterns live in THIS file so a remote pkill -f never matches the caller's argv.
HOSTS="${HOSTS:-bh-glx-110-a02u02 bh-glx-110-a02u08 bh-glx-110-a03u02 bh-glx-110-a03u08}"
A='[b]laze.models.cli'; B='[p]rterun'; C='[t]ts_runner'; D='[g]enerate_rank_bindings'; E='[t]t-run'
for h in $HOSTS; do
  timeout 60 ssh -o BatchMode=yes "$h" "docker stop -t 30 tt-cpp-worker >/dev/null 2>&1; docker rm tt-cpp-worker >/dev/null 2>&1; pkill -TERM -f '$A'; pkill -TERM -f '$B'; pkill -TERM -f '$C'; pkill -TERM -f '$D'; pkill -TERM -f '$E'" 2>/dev/null
done
sleep 5
for h in $HOSTS; do timeout 30 ssh -o BatchMode=yes "$h" "kill -9 \$(pgrep -f '$A|$B|$C|$D|$E') 2>/dev/null" 2>/dev/null; done
n=0; for h in $HOSTS; do n=$((n + $(timeout 20 ssh -o BatchMode=yes "$h" "pgrep -cf '$A|$B|$C|$D'" 2>/dev/null | head -1))); done
echo "torn down (procs left=$n)"
EOF
bash $STACK/teardown.sh

# warm reset of all four galaxies (~3 min); expect "Recovery succeeded on attempt 1 of 1"
ssh -A $LAUNCH_HOST "bash -lic 'export HOSTS=$H4; recover-hosts'" 2>&1 | grep -iE "Recovery succeeded|Reset completed|FAIL"

# shm AFTER recovery (recover-hosts leaves its own CHIP_IN_USE locks); root-owned -> sudo helper
for h in ${H4//,/ }; do ssh $h 'sudo -n /usr/local/bin/tt_rm_shm.sh >/dev/null 2>&1; echo "'$h' shm=$(ls /dev/shm | grep -ciE "tt_|UMD")"'; done
```

> **TRAP - relaunching without `recover-hosts` hangs blaze.** Observed: after a crashed
> decoder and an earlier ring on the same chips, the next blaze launch got 2 of 35 ranks
> through setup; rank 2 (stage 0) sat for 20+ min in `ttnn.synchronize` ->
> `wait_for_cores_idle` polling a core on its own mesh, the rest spinning in
> "Setting up". The identical launch after a fresh `recover-hosts` reached KEEP_ALIVE in
> 4.5 min. Reset before every relaunch, not just after failures.

Device-open probe, per host - `HEALTHY` or abort. Do not shorten the timeout (device open
takes 30-90 s on some hosts); retry only on empty output:

```bash
cat > $STACK/probe.sh <<'EOF'
source /data/ljovanovic/stacks/tts2_b1721755_m00e3f21/env_blaze.sh >/dev/null 2>&1
cd "$TT_METAL_HOME" || exit 9
TT_VISIBLE_DEVICES=0 timeout 200 python -c "
import ttnn
d = ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print('HEALTHY'); ttnn.close_mesh_device(d)
" 2>&1 | grep -oiE "HEALTHY|unexpected NOC address|No such device|Error.*" | head -1
EOF
for h in ${H4//,/ }; do r=""; for t in 1 2; do r=$(timeout 240 ssh $h 'bash -s' < $STACK/probe.sh | tail -1); [ -n "$r" ] && break; done; echo "$h -> ${r:-NO OUTPUT}"; done
for h in ${H4//,/ }; do ssh $h 'sudo -n /usr/local/bin/tt_rm_shm.sh'; done   # probe leaves locks
```

`Sysmem mapped at unexpected NOC address` is not fixable by `recover-hosts` - the host
needs a power cycle / admin.

---

## 3. Rank placement: who is rank 0, where the decoders go, DEVICE_IDS

**Rank 0 is NOT the first `--hosts` entry.** tt-run's Phase 1 (`generate_rank_bindings`)
fits the mesh graph descriptor onto the discovered cabling and writes
`tt-blaze/generated/ttrun/<hash>/{rank_bindings.yaml,rankfile}` (relative to the
tt-blaze root, where tt-run is launched from). On the old d05/d06 quad rank 0 was the
*last* `--hosts` entry; on Quad2 it is the first. Run Phase 1 alone once per quad:

```bash
# on $LAUNCH_HOST, with ssh -A
source $STACK/env_blaze.sh && cd $STACK/tt-blaze
tt-run --mesh-graph-descriptor $STACK/tt-blaze/tests/pipeline_builder/mesh_graph_descriptors/llama_8b_4galaxy_mesh_graph_descriptor.textproto \
  --hosts $H4 --tcp-interface ens5f0np0 --force-rediscovery \
  --mpi-args "--map-by ppr:1:node:OVERSUBSCRIBE" -- true      # Phase 2 then fails; expected
cat $(ls -t generated/ttrun/*/rankfile | head -1)
```

Quad2's layout (35 ranks):

| host | ranks | chips used |
|---|---|---|
| a02u02 | 0 (mesh 0) | all 32 |
| **a02u08** | 1 (mesh 1: `16,17,20,21`), 2 (mesh 2 = **stage 0**: `0,1,4,5,8,9,12,13`) | 12 of 32 |
| a03u08 | 3-18 (mesh 3) | all 32 |
| a03u02 | 19-34 (mesh 4) | all 32 |

The **decoder host** is the one host with free chips - the one carrying the two small
meshes (here a02u08). Stage 0 (rank 2) and its `tt_{h2d,d2h}_tts2_speechlm.bin`
descriptors land there too, so decoders, blaze I/O and the server all share memory on it.

`DEVICE_IDS` for the server = **every chip blaze uses on the decoder host + the decoder
chips**, read from the bindings, never guessed:

```bash
cat > $STACK/device_ids.py <<'EOF'
import sys, yaml
f, host, dec = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split()]
chips = set()
for e in yaml.safe_load(open(f))["rank_bindings"]:
    ov = e.get("env_overrides", {})
    if ov.get("TT_METAL_CLUSTER_ID") == host:
        chips |= {int(x) for x in ov["TT_VISIBLE_DEVICES"].split(",")}
print("(" + ",".join(str(c) for c in sorted(chips) + dec) + ")")
EOF
$STACK/tt-blaze/tt-metal/python_env/bin/python $STACK/device_ids.py \
  $(ls -t $STACK/tt-blaze/generated/ttrun/*/rank_bindings.yaml | head -1) bh-glx-110-a02u08 "24 25"
# Quad2 -> (0,1,4,5,8,9,12,13,16,17,20,21,24,25)
```

Both ranks on that host matter: rank 2 owns the speechlm H2D socket, rank 1 the
lm-head/sampling stage. A missing chip fails the server worker with
`H2D socket connector cannot find physical PCIe device /dev/tenstorrent/<n>`. The old
quad's hard-coded `(2,3,6,7,10,11,14,15,24,25)` does not carry over.

---

## 4. Audio decoders (T=48, batch 2) on the decoder host

```bash
cat > $STACK/launch_decoders.sh <<'EOF'
#!/usr/bin/env bash
S=/data/ljovanovic/stacks/tts2_b1721755_m00e3f21
L=${L:?need L}; mkdir -p "$L"
DECCHIPS=${DECCHIPS:-"24 25"}; WIRE_T=${WIRE_T:-48}
source $S/env_tts.sh >/dev/null
cd "$TT_METAL_HOME"
i=0
for c in $DECCHIPS; do
  i=$((i+1)); sfx=""; [ $i -gt 1 ] && sfx="_$i"
  TT_VISIBLE_DEVICES=$c nohup python -u -m models.demos.inworld_tts.tts_runner --model inworld_tts_decoder \
    --model-path "$TTS2_CODEC_DECODER_CKPT" ${sfx:+--socket-suffix $sfx} \
    --tokens $WIRE_T --batch $TTS_DECODER_BATCH > "$L/decoder$i.log" 2>&1 < /dev/null &
  echo "decoder$i chip=$c suffix=${sfx:-none} T=$WIRE_T batch=$TTS_DECODER_BATCH pid=$!"
done
EOF
L=$STACK/tts_logs/run_$(date +%m%d_%H%M)
ssh bh-glx-110-a02u08 "L=$L bash $STACK/launch_decoders.sh"
# wait for BOTH:  TTS_RUNNER_READY v1 component=inworld_tts_decoder ... codes_per_chunk=48 ... users_per_chunk=2
grep -h TTS_RUNNER_READY $L/decoder1.log $L/decoder2.log      # ~30 s (trace capture)
```

Use the **plain** `inworld_tts_decoder` target, launched directly - not
`--model inworld_tts_codec` / `--dec-chip`:

> **TRAP - `00e3f21`'s codec wrapper only starts `inworld_tts_decoder_valid`.** That
> decoder asserts `T % 32 == 0` (`runtime-mask path assumes tile-aligned T (got 48)`), and
> at T=32 it builds an `8*T*4 = 1024 B` FIFO. The server engine writes 48-code pages
> (8 rows x 48 x 4 = 1536 B), so it dies at connect with
> `TT_FATAL: Page size must be less than or equal to the FIFO size`. The plain target
> takes T=48 and uses `TtDecoder`'s 64512 B FIFO. (`83feb0fd` has the same assertion.)

**The 64512 B FIFO is load-bearing.** `H2DSocket::write` does not split at the FIFO wrap,
so a write straddling the end overruns L1 and the decoder goes idle with work queued
(44-byte responses, chunk-4 stall). 64512 = 42 x 1536 = 168 x 384 is a multiple of both
the 8-page and the batch-2 write, so no write can straddle. Check it before deploying:

```bash
grep -n "socket_fifo_size_bytes: int" $STACK/tt-metal/models/demos/inworld_tts/tt/decoder_tts2.py
# must read:  socket_fifo_size_bytes: int = 64512,
```

Do not raise it past ~128 KB (L1-backed; 256 KB crashes the decoder).

---

## 5. Blaze ring (direct tt-run, `--n-slots`)

```bash
cat > $STACK/launch_blaze.sh <<'EOF'
#!/usr/bin/env bash
S=/data/ljovanovic/stacks/tts2_b1721755_m00e3f21
NSLOTS=${NSLOTS:-1000}
H4=${H4:-bh-glx-110-a02u02,bh-glx-110-a02u08,bh-glx-110-a03u02,bh-glx-110-a03u08}
L=${L:?need L}; mkdir -p "$L"
for v in $(env | grep -oE "^SLURM_[A-Z_]+"); do unset $v; done
source $S/env_blaze.sh >/dev/null
cd "$(dirname "$(realpath "$TT_METAL_HOME")")"     # tt-blaze root: Phase 1 cache in ./generated/ttrun
export TT_METAL_SLOW_DISPATCH_MODE=1               # blaze only
nohup tt-run --mesh-graph-descriptor $S/tt-blaze/tests/pipeline_builder/mesh_graph_descriptors/llama_8b_4galaxy_mesh_graph_descriptor.textproto \
  --hosts "$H4" --tcp-interface ${TCPIF:-ens5f0np0} -- \
  python -m blaze.models.cli --model inworld-ai/tts-2-speechlm --weights real \
    --model-path "$TTS2_SPEECHLM_PATH" --cache-path $S/weight_cache \
    --n-slots "$NSLOTS" --launch-only --io-socket-descriptor-prefix tts2_speechlm \
  > "$L/blaze.log" 2>&1 < /dev/null &
echo "launched pid $! log $L/blaze.log"
# keep this ssh (and the lent agent) alive until mpirun has reached every host
for i in $(seq 1 36); do
  grep -aqiE "Traceback|Host key|lost communication" "$L/blaze.log" && break
  [ "$(grep -ac rank "$L/blaze.log")" -gt 50 ] && break; sleep 5
done
EOF
# ATTACHED, with -A at both hops - see trap
herdr-ssh -A "ssh -A $LAUNCH_HOST 'NSLOTS=$NSLOTS L=$L bash $STACK/launch_blaze.sh'"
```

Wait for `Rank 2 (stage 0) host loop: KEEP_ALIVE` in `$L/blaze.log` (first launch with an
empty weight cache ~4 min; later ~2.5 min) and confirm all 35 ranks finished setup:

```bash
sed 's/\x1b\[[0-9;]*m//g' $L/blaze.log | grep -a 'Pipeline setup complete' | grep -oE '^\[1,[0-9]+\]' | sort -u | wc -l   # 35
ls /dev/shm/tt_*_tts2_speechlm.bin     # on the decoder host
```

Treat it as stalled if the log is silent for 15 min **and** fewer than 35 ranks reached
setup. A rank is stuck (not slow) when a `gdb -p <pid> -batch -ex 'bt'` shows
`SDMeshCommandQueue::wait_for_cores_idle` for minutes -> teardown + `recover-hosts`.

> **TRAP - the launch must stay attached until MPI has fanned out.** prterun reaches the
> other three hosts over ssh with the *lent* agent. Backgrounding the launch call closes
> the ssh session (and the agent) first, and every remote daemon fails with
> `Permission denied (publickey)` -> `PRTE has lost communication with a remote daemon`.
> Once the daemons are up the session can end; the ring keeps running.

> **TRAP - do not launch blaze through `tts_runner` with this fork.** `00e3f21`'s runner
> never forwards a slot count, so blaze defaults to `n_slots=1` and every rank dies with
> `batch_slots needs n_slots (1) > max(slot_ids) (7)` (its error path then crashes with
> `NameError: name 'h2d' is not defined`). The direct command above is exactly what the
> runner runs, plus `--n-slots`.

The log is full of `UserWarning: CT arg schema mismatch ...` and
`Multi-phase CB remap ... spec-based fallback` from blaze. They are harmless on this
commit - the ring serves correctly with them.

---

## 6. Inference server (docker, on the decoder host)

The image must be in the host's local docker (`docker images | grep tts`); there are no
ghcr credentials on the quad hosts. Load it there (`docker load` from a tarball on
`/data`, or `docker login ghcr.io` + `docker pull` by the user).

```bash
cat > $STACK/launch_is.sh <<'EOF'
#!/usr/bin/env bash
NSLOTS=${NSLOTS:-1000}
IMG=${IMG:-ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:tts_sampling_topk64_penalties}
SPEECHLM=${SPEECHLM:-/data/ssinghal/inworld/tts-models/checkpoints/speechlm}   # SAME checkpoint blaze loaded
DECPAIRS=${DECPAIRS:-tts2_decoder_h2d:tts2_decoder_d2h,tts2_decoder_h2d_2:tts2_decoder_d2h_2}
DEVIDS=${DEVIDS:?from device_ids.py, e.g. (0,1,4,5,8,9,12,13,16,17,20,21,24,25)}
docker stop -t 30 tt-cpp-worker >/dev/null 2>&1; docker rm tt-cpp-worker >/dev/null 2>&1
docker run -d --name tt-cpp-worker --privileged --ipc=host \
  --ulimit nofile=65536:65536 --ulimit nproc=65536:65536 \
  -p 8010:8000 \
  -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /etc/udev/rules.d:/etc/udev/rules.d -v /lib/modules:/lib/modules \
  -v /var/run/tenstorrent:/var/run/tenstorrent \
  -v "$SPEECHLM":/tokenizers/tts:ro \
  -e SERVER_MODE=cpp \
  -e MODEL_SERVICE=tts -e MODEL_RUNNER_TYPE=tt_tts \
  -e MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -e TTS_TOKENIZER_PATH=/tokenizers/tts/tokenizer.json \
  -e TTS_ENCODER_ENABLED=0 \
  -e TTS_DECODER_TRANSPORT=socket \
  -e TTS_DECODER_SOCKET_PAIRS="$DECPAIRS" \
  -e TTS_PAGE_WIDTH=8 -e TT_BLAZE_M_TOKENS=8 -e TTS_DECODER_BATCH=2 \
  -e TTS_MAX_SEQ_LEN=12288 \
  -e TTS_PAGE_LINGER_US=500 -e TTS_PAGE_MIN_ROWS=8 \
  -e TTS_PREFILL_CHUNK_SIZE=256 \
  -e TTS_EDF_SAFETY_FACTOR=0.75 \
  -e TTS_FC_DEADLINE_US=150 \
  -e TTS_MAX_NEW_TOKENS=6000 -e TTS_MIN_NEW_TOKENS=25 \
  -e TTS_CHUNK_TOKENS_1=17 -e TTS_CHUNK_TOKENS_2=32 -e TTS_CHUNK_TOKENS_3=48 \
  -e TTS_FIRST_CHUNK_TOKENS=17 -e TTS_SECOND_CHUNK_TOKENS=32 -e TTS_CHUNK_TOKENS=48 \
  -e TTS_MAX_BATCH_SIZE=$NSLOTS -e TTS_MAX_USERS=$NSLOTS -e MAX_IN_FLIGHT_COUNT=$NSLOTS \
  -e MAX_SESSIONS_COUNT=$NSLOTS -e PM_MAX_USERS=$NSLOTS \
  -e MAX_QUEUE_SIZE=4000 \
  -e TTS_TEMPERATURE=1.0 -e TTS_TOP_P=0.9 -e TTS_TOP_K=64 \
  -e TTS_REPETITION_PENALTY=1.1 -e TTS_FREQUENCY_PENALTY=0.1 \
  -e TT_LOG_LEVEL=debug -e TTS_TIMING=1 \
  -e OPENAI_API_KEY=your-secret-key \
  -e DEVICE_IDS="$DEVIDS" \
  --entrypoint /bin/bash "$IMG" -c 'cd cpp_server && ./build/tt_media_server_cpp'
EOF
ssh bh-glx-110-a02u08 "DEVIDS='(0,1,4,5,8,9,12,13,16,17,20,21,24,25)' bash $STACK/launch_is.sh"
sleep 45; ssh bh-glx-110-a02u08 'docker logs tt-cpp-worker 2>&1 | grep -aiE "capacity=|makeTtsScheduler: decoder|residual|TT_THROW|TT_FATAL|terminate"'
```

Expect: `capacity=1000`, one `makeTtsScheduler: decoder N over shared memory` line per
decoder (0 and 1), `drained 0 + 0 residual D2H pages ... clean handoff`, and **no**
`TT_THROW`. `workers=1` is the TTS service worker, not the decoder count.

Knobs that must agree:

- **Five caps = blaze `--n-slots`**: `TTS_MAX_BATCH_SIZE`, `TTS_MAX_USERS`,
  `MAX_IN_FLIGHT_COUNT`, `MAX_SESSIONS_COUNT`, `PM_MAX_USERS`. Raising one side only
  makes the extra slots invisible (429s). `MAX_QUEUE_SIZE` is independent.
- `TTS_PAGE_WIDTH` = `TT_BLAZE_M_TOKENS` = 8 (a mismatch clamps prefill -> empty audio).
- `TTS_DECODER_BATCH` = every decoder's `--batch` (2).
- `TTS_CHUNK_TOKENS_1/2/3` (17/32/48) are required; without them the stack stalls silently.
- Tokenizer mount = the speechlm checkpoint blaze loaded. The old
  `/mnt/models/.../tts-2/checkpoints/speechlm` has different token ids.
- No `TTS_BOS_TOKEN` with this image / prompt format.

> **Stop the container with `docker stop -t 30`, never `docker rm -f`.** SIGKILL skips the
> socket destructors and leaves the decoder H2D mid-transfer; the next container then
> aborts in `DecoderSocketPipeline` and only a full relaunch (stage 2 onward) clears it.

---

## 7. Audio proof + warmup - `health=200` is not proof of life

```bash
ssh bh-glx-110-a02u08 '
for i in $(seq 1 45); do
  ok=$(curl -s -o /dev/null -m 240 -w "%{http_code}" -X POST http://localhost:8010/v1/audio/speech \
    -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" -d "{\"text\": \"Hi there.\"}")
  [ "$ok" = "200" ] && break; sleep 10; done
curl -s -o /tmp/proof.wav -w "http=%{http_code} bytes=%{size_download} ttfb=%{time_starttransfer}s\n" -m 120 -X POST \
  http://localhost:8010/v1/audio/speech -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
  -d "{\"text\": \"The quick brown fox jumps over the lazy dog.\"}"
curl -s http://localhost:8010/max-session-count; echo'
```

Expect ~270-320 KB for that sentence (validated: 266,924 B, RMS ~4000) and
`{"max_session_count":1000}`. A wedged stack answers 200 with **44 bytes** (a bare WAV
header). The streamed WAV header's frame count is a placeholder - judge by bytes, not by
`wave`'s reported duration.

**Warm up before any measurement - 100 concurrent requests.** `tokenizer_resolve` costs
~11-14 s once per drogon IO thread (64 of them); a request on a cold thread shows an
11 s TTFB. Sequential warmup does not cover the pool:

```bash
ssh bh-glx-110-a02u08 'for i in $(seq 1 100); do curl -s -o /dev/null -w "%{http_code} %{size_download}\n" -m 180 -X POST \
  http://localhost:8010/v1/audio/speech -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
  -d "{\"text\": \"The quick brown fox jumps over the lazy dog.\"}" & done > /tmp/warm.txt; wait
  echo "ok=$(grep -c "^200 " /tmp/warm.txt)/100 small=$(awk "\$2<10000" /tmp/warm.txt | wc -l)"'
```

Validated: 100/100 OK, 0 short; then warm singles TTFB ~1 ms, ~1.9 s total.

> `socket_driver.py` from the fork can NOT be used as a server-less audio check with this
> blaze: it imports `METADATA_SIZE_BYTES` from `blaze.ops.utils.metadata`, which blaze's
> metadata rewrite removed. The server image is the matching client.

---

## Sanity-check measurements before trusting numbers

Closed loop pins occupancy, so **measured C must equal configured concurrency**:

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

`C/N` should read 1.0000 and Little's law (`rps*W/N`) agree within ~1%.

---

## Teardown when finished

`bash $STACK/teardown.sh` (stops the container with `docker stop -t 30` first), then the
shm clear from stage 2 - leftover segments are root-owned and hold ~2.6 GB.

---

## Failure quick reference

| Symptom | Cause | Action |
|---|---|---|
| CMake: `No VERSION specified for WRITE_BASIC_CONFIG_VERSION_FILE()` | fork tag `tt-tts-inworld-*` picked by `git describe` | `git tag -d $(git tag -l 'tt-tts-*')` in the clone |
| CMake: `BLAKE3::blake3 ... target was not found` | two builds raced on one tree | move `build_Release` + `.cpmcache/blake3` aside, build once |
| `ModuleNotFoundError: No module named 'absl'` | fork venv lacks absl-py | `python_env/bin/python -m pip install absl-py` |
| Phase 1: `Host key verification failed` | quad hosts don't know each other | stage 1 known_hosts loop |
| `Permission denied (publickey)` + `PRTE has lost communication` | blaze launch backgrounded; lent agent gone | launch attached with `-A` at both hops |
| `Access denied by pam_slurm_adopt` | drain-held quad, flaky Slurm | retry the ssh |
| `batch_slots needs n_slots (1) > max(slot_ids) (7)` | blaze started via the fork's `tts_runner` (no slot flag) | launch blaze directly with `--n-slots` |
| 2/35 ranks set up, rank 2 stuck in `wait_for_cores_idle` | relaunched without a reset | teardown + `recover-hosts` + shm, relaunch |
| decoder: `runtime-mask path assumes tile-aligned T (got 48)` | `_valid` decoder via codec wrapper | plain `inworld_tts_decoder --tokens 48` |
| server: `Page size must be less than or equal to the FIFO size` | decoders at T=32 (1024 B FIFO) | T=48 plain decoders |
| server: `cannot find physical PCIe device /dev/tenstorrent/<n>` | `DEVICE_IDS` missing a blaze chip | derive from rank_bindings (stage 3) |
| `Waiting for lock 'CHIP_IN_USE_...'` in a decoder | blaze started before decoders | full relaunch, decoders first |
| `health=200`, 44-byte responses | FIFO not batch-aligned / socket wedged | 64512 FIFO; relaunch |
| `H2D ... Bytes sent: N, acknowledged: M` | container killed with `docker rm -f` | `docker stop -t 30`; full relaunch |
| `max_session_count` < `NSLOTS` | caps didn't propagate | all five caps = `--n-slots` |
| `Sysmem mapped at unexpected NOC address` | stranded mapping | power-cycle / admin |
| shell dies, exit 144 | remote `pkill -f` matched caller argv | patterns in a file, `[x]yz` bracket form |

## Notes that are easy to get wrong

- **`pgrep -c` over ssh counts itself** when the pattern appears in the remote command
  line - use `[x]yz` bracket forms *and* keep the pattern out of the echoed text.
- **`ssh -n` eats stdin.** Drop `-n` for any `ssh host 'bash -s' < file`.
- **`${VAR:+X=$VAR}` is not an assignment prefix.** Use `env X="$VAR" cmd ...`.
- **Kernel cache needs no clearing after a blaze source change** - generated kernels
  are named by the SHA-256 of their source, so an edit gets a new cache key.
- **The weight cache is per stack** (`$STACK/weight_cache`). The first blaze launch fills
  it (~4 min total); reusing another tree's cache across blaze versions is unsafe.
