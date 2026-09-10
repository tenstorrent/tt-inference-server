# Running the C12 recipe on OM Quad3

`../c12_quad/h3_deploy.sh` is the MiniMax-H3 deployment that ran clean on the C12 4x32 quad on 2026-09-10. This directory
makes the **same script, same pins, same knobs** run on OM Quad3 -- racks F1/F2, `172.16.104.120-123`, 4 hosts x 32 Blackhole
chips -- purely through the script's `H3_*` overrides. Nothing in the recipe is forked. Written 2026-09-10 from read-only probes
of the F hosts (via the `ubuntu`/maas.pem hop from gbh-e4-02); **nothing has run on Quad3 yet** -- see the last section.

## TL;DR

Pins, unchanged and not up for discussion: device tree tt-metal **`162a86b008a`** (fetchable as `tenstorrent/tt-metal` branch
`zni/h3-c12-clean-device-tree`; the public commit with the same message, `193c08b2944`, is NOT equivalent), python tree
**`34260b25483`**, server **`78d516584` + `70a756282`** plus the inspect-gated `dit_runners.py` patch the script applies;
`MINIMAX_H3_TRACE_DENOISE=0`, `MINIMAX_H3_BUCKET_DENOISE=1`, `MINIMAX_H3_CONSTRUCTION_WARMUP=0`, `TT_METAL_CACHE` unset.
`h3_deploy.sh setup` fetches the right refs itself when a commit is missing locally. (For the record, not to act on: metal
`8fb0c4c0483`, post-rebase with the LLK change included, ran the OM tier suites clean on Quad1+Quad2 on 2026-09-06.)

| file here | role |
|---|---|
| `quad3_deploy.sh` | **the one-command driver**, run on `.120` as `zni`: `quad3_deploy.sh all fl2va` = preflight -> prep -> setup -> replicate -> reset (once) -> start -> probe 3; also every `h3_deploy.sh` action with the overrides applied, and `k8s-status` / `k8s-cordon` / `k8s-uncordon`. Refuses chip work until `H3_K8S_OK=1` |
| `env_om_quad3.sh` | the `H3_*` overrides; `source` it before every `h3_deploy.sh` call (the Quad3 cluster fact list, every line says why) |
| `base_env_om.sh` | the OM stand-in for C12's `metal_env_H3.sh` (`H3_BASE_ENV`): weights, DiT cache overlay, MPI, fabric timeouts, SHM names |
| `om_quad3_prep.sh` | run on the launch node `.120` as `zni` (idempotent; again after any reboot): per-host dirs, ffmpeg, overlay mount, health checks; clones, env files, rankfile, media-venv check |
| `copy_media_env_from_quad1.sh` | run on gbh-e4-02 as `zni`: the media-server venv (and optionally the converted DiT cache upper) over the 100 GbE rack net |

What the C12 script assumes that is not true on an OM quad, and what covers it (same order as `env_om_quad3.sh`):

1. **No `metal_env_H3.sh`.** The script's env file `source`s `$H3_BASE_ENV`; here that is `base_env_om.sh` = the OM base env
   (weights `/data_bh/h3`, `TT_DIT_CACHE_DIR` on the overlay, `MPICC`/ulfm PATH, `TT_METAL_OPERATION_TIMEOUT_SECONDS=300`,
   `TT_METAL_SHM_TRACKING_DISABLED=1`, `TT_METAL_LOGS_PATH=/tmp/tt-logs`, `tt_video_in`/`tt_video_out`, ...). It sets none of
   `TT_METAL_HOME` / `PYTHONPATH` / `LD_LIBRARY_PATH` -- the C12 env file derives those from the two pinned trees.
2. **No cabling/deployment descriptors for a 4x32 quad**, so `recover.sh` cannot run (`/data_bh/scaleout_configs/` holds only a
   16-host factory descriptor; a 4-host subset through `--factory-descriptor-path` is untested). `H3_RESET_MODE=glx_reset_auto`
   makes `reset` run `timeout 400 tt-smi -glx_reset_auto` over ssh on every host in parallel, then settle 60 s. Never `tt-smi -r`.
3. **No rankfile.** `om_quad3_prep.sh` writes `$H3_VM/quad3_rankfile` (`rank <i>=<ip> slot=0:*`, rank i = host i of `H3_HOSTS`);
   `verify_setup` generates the same when it is missing and `H3_HOSTS` is set.
4. **No shared filesystem.** `/home` is local ext4 per host; the NFS `/data_bh` is writable by `ubuntu` only and 95 % full,
   `/data_models` is not writable. So `H3_VM=/home/zni/c12` is a LOCAL path, identical on all four hosts, and `H3_SHARED_FS=0`
   turns on `h3_deploy.sh replicate`: rsync of `$H3_VM/` from rank 0 to the three peers excluding `deploy_logs/`, every `built/`
   (tt-metal's per-rank compiled-kernel cache -- syncing rank 0's over a peer's deletes that rank's kernels, ~2,700 recompiled per
   request, measured on Quad1), `cache-upper/`, `cache-work/`, `cache-merged/`, `videos/`, `*.log`; it then checks `libtt_metal.so`
   md5 and the python tree's HEAD on every host. The git worktrees reference `$H3_METAL_REPO/.git/worktrees/...`, which is fine
   because the whole `$H3_VM` lands at the same absolute path.
5. **DiT weight cache.** The prebuilt `/data_bh/h3/dit_cache/tt_dit_cache/{minimax-h3,minimax-h3-adaln}` (635 GB) is readable by
   all, writable only by `ubuntu`. Each host mounts an overlay -- `lowerdir=` that cache, `upperdir=$H3_VM/cache-upper`,
   `workdir=$H3_VM/cache-work`, mounted at `$H3_VM/cache-merged` = `TT_DIT_CACHE_DIR`. Without it the first start converts ~80 GB
   of weights per host into the upper, or fails on a read-only dir. gbh-e4-02's 202 GB `cache-upper` (converted for metal
   `94cff3e426a`) can seed it (`copy_media_env_from_quad1.sh --with-cache`); whether the `34260b25483` pipeline reuses those
   entries is unverified -- a miss just re-converts.
6. **Media-server venv.** None on the F hosts. gbh-e4-02 has a relocatable one (3.6 GB, uv-made, interpreter inside, `bin/tt-run`,
   `bin/uvicorn`, `bin/ffmpeg`; its `.pth` files point at `/home/zni/tt-metal-h3`, harmless because the C12 env file puts the pinned
   tree's `ttnn` first on `PYTHONPATH`). It is copied to `$H3_MEDIA_ENV=$H3_VM/media_python_env` and chowned, because zni's uid differs
   between hosts. zni keys are per quad, so the copy goes `ubuntu` + maas.pem, run on gbh-e4-02.
7. **Symlinks into the shared clone** (`.cpmcache`, `runtime`, `internal-prodia`, `python_env`) have no target in a fresh clone;
   `setup` now creates them only when the target exists and says what it skipped. CMake fetches CPM deps from the internet on the
   first build (GitHub/PyPI are reachable from the F hosts).
8. **ffmpeg is not installed on the F hosts** (Quad1/2 have `/usr/bin/ffmpeg`). `om_quad3_prep.sh` apt-installs it on all four;
   `probe` also accepts `$H3_MEDIA_ENV/bin/ffmpeg`.
9. **Canary.** `H3_CANARY` stays at its default `true` (C12 behaviour). The OM bare-metal stack runs `CANARY_ENABLED=false` because a
   cancelled job frees the API slot while the SP runner keeps generating; the canary probes a busy mesh, misses 3x in 12 s and
   declares the model DEAD (that killed the Quad1/Quad2 k8s deployments). If that bites, `export H3_CANARY=false` -- `wait_ready`
   then keys on `"model_ready": true`.
10. **Hosts by IP**, not name: on OM a hostname resolves to `127.0.1.1` on itself and MPI misidentifies the local rank.
    `H3_HOSTS=172.16.104.120,...,123`, `H3_RANK0=172.16.104.120`, API `http://172.16.104.120:8000`. ssh trust for `zni` inside the quad
    comes from quad-agent `scripts/provision-zni.sh --quad3 --trust`.
11. `fuser` (used by `check`/`stop`) is present on the F hosts, as are `lsof`, `script`, tmux, rsync, OpenMPI ulfm 5.0.7 at
    `/opt/openmpi-v5.0.7-ulfm`, tt-kmd 2.8.0, tt-smi 5.0.0, firmware bundle 19.8.1.0 (identical to Quad1 and to C12's 19.8.1).

## Preconditions -- STOP gates

Reachability: the F hosts answer over the existing OME1 VPN (`172.16.0.0/12` already covers them, no VPN change). BMCs are
`172.16.4.120-123` (BMC, not the host OS). Hardware = Quad1: EPYC 9354P, 64 threads, 566 GB RAM, `/` 880 GB local ext4
(642/507/750/509 GB free on .120/.121/.122/.123), `/dev/shm` 284 GB, Ubuntu 22.04.5, kernel 6.8.0-138-lowlatency.

1. **STOP -- k8s is live on Quad3.** rke2 runs on all four F nodes (`.120` is a server/control-plane node, `.121-.123` agents) with
   the tt-operator DRA driver, fabric-manager-agent and telemetry daemonsets. No H3 pods now, but the DRA driver can hand chips to
   a k8s job at any time. Cordon the four F nodes or get the k8s owner's OK before `start`/`reset`. `sudo
   /var/lib/rancher/rke2/bin/kubectl --kubeconfig /etc/rancher/rke2/rke2.yaml get nodes` should work on `.120` itself (untested;
   the join server `.118` = gbh-e4-01 is down). Quad2 had the same residue and needed one `tt-smi -glx_reset_auto` per host before
   the first mesh open (`Sysmem mapped at unexpected NOC address`) -- hence the single `reset` in the runbook.
2. **STOP -- `zni` does not exist on the F hosts yet.** Accounts there: `ubuntu` (NOPASSWD sudo) and `user` (uid/gid 1001). From
   the quad-agent box: `scripts/provision-zni.sh --quad3 --trust` (creates `zni` through `ubuntu` + maas.pem via a reachable Quad1
   host, lets the system pick the uid since 1001 is taken, adds the `gbh-f1-01/f1-02/f2-01/f2-02` aliases, and `--trust` puts the
   per-quad `zni` key on `.120` and authorizes it on the peers). Never hand-create the account. Verify with `ssh gbh-f1-01 'id; hostname -I'`.
3. **STOP -- gbh-e4-01 is Quad1's launch node and is DOWN (2026-09-10; UBB0 tray failures 09-06 and 09-09).** It is not part of
   Quad3. The copy source is **gbh-e4-02 (`172.16.104.119`)**, which has the media venv, the 202 GB `cache-upper` and maas.pem.
   Never copy trees through the VPN from a laptop (5-9 MB/s); the rack network is 100 GbE.

Rules that carry over: one deployment per quad at a time (Quad3's chips are a separate pool from Quad1/Quad2, so the rule applies
per quad); work as `zni`, `ubuntu` only to create or repair `zni`; never `tt-smi -r`; never put credentials or the cluster-internal
`zni` key in a repo.

## Runbook (ordered)

One command, once `zni` exists on the four hosts and the media venv has been copied (steps 0-1 below):
```bash
D=/home/zni/c12/tt-inference-server/tt-media-server/tests/deploy
H3_K8S_OK=1 $D/om_quad3/quad3_deploy.sh all fl2va       # H3_K8S_OK=1 = you ran `quad3_deploy.sh k8s-cordon` or have the k8s owner's OK
```
It logs to `/home/zni/c12/zni_worktrees/deploy_logs/quad3_deploy.<ts>.log`, stops at the first failing step, and refuses to
reset while any host still holds `/dev/tenstorrent` handles (`H3_ALLOW_HOLDERS=1` overrides; `H3_SKIP_RESET=1` skips the
one-time reset). Step by step, the same thing:

```bash
# 0. on gbh-e4-02 as zni (Quad1): the media venv -> Quad3's launch node; --with-cache adds the 202 GB cache-upper seed (optional).
#    Do NOT switch branches in /home/zni/h3-deploy/tt-inference-server -- that checkout IS the Quad1 deployment. Take the two
#    files out of the fetched branch (they travel together: the copy script sources env_om_quad3.sh from its own directory),
#    or scp them over -- they are small. FETCH_HEAD, not origin/$B: a single-branch clone never updates that ref.
R=/home/zni/h3-deploy/tt-inference-server; B=zni/h3-test-coverage-on-78d51658; mkdir -p /home/zni/om_quad3
git -C $R fetch origin $B && for f in env_om_quad3.sh copy_media_env_from_quad1.sh; do
  git -C $R show FETCH_HEAD:tt-media-server/tests/deploy/om_quad3/$f > /home/zni/om_quad3/$f; done
bash /home/zni/om_quad3/copy_media_env_from_quad1.sh [--with-cache]

# 1. on 172.16.104.120 (OM-F1-GBH01) as zni: the server clone IS H3_TIS_REPO; check out the branch that carries these scripts
git clone https://github.com/tenstorrent/tt-inference-server.git /home/zni/c12/tt-inference-server
git -C /home/zni/c12/tt-inference-server checkout zni/h3-test-coverage-on-78d51658
D=/home/zni/c12/tt-inference-server/tt-media-server/tests/deploy
bash $D/om_quad3/om_quad3_prep.sh          # idempotent; fix every FAIL/MISSING line and re-run until PREP OK

# 2. every h3_deploy.sh call needs the overrides in the shell
source /home/zni/c12/env_om_quad3.sh

# 3. worktrees at the pins, device-tree build of 162a86b008a on the first run, symlinks, SP cherry-pick, dit_runners patch,
#    env file, import smoke test. Build: ~5 min on 64 cores -- expected (Quad1/Quad2 builds), plus the first-time CPM download.
$D/c12_quad/h3_deploy.sh setup

# 4. no shared fs: push /home/zni/c12 to .121/.122/.123 (verifies libtt_metal.so md5 + python HEAD on every host).
#    Repeat after EVERY setup or edit under /home/zni/c12 -- `start` reminds you.
$D/c12_quad/h3_deploy.sh replicate

# 5. once: the k8s residue on the chips (tt-smi -glx_reset_auto on all four in parallel, ~2 min per host -- expected
#    (Quad1/Quad2) -- then a 60 s settle). rc != 0 on any host usually means an FRB2/FRB3 tray hang -- confirm on that host
#    (lspci | grep -ci 1e52 < 32, or tt-smi 'No chips detected') before asking for a BMC power cycle; do not re-reset.
$D/c12_quad/h3_deploy.sh reset

# 6. stop whatever runs, launch 4x32 ranks + the SP frontend on .120, wait until ready
$D/c12_quad/h3_deploy.sh start fl2va

# 7. three identical requests, judged (audio/bppf + frame-0 vs keyframe PCC); needs ffmpeg (prep installed it)
$D/c12_quad/h3_deploy.sh probe fl2va 3
$D/c12_quad/h3_deploy.sh status | check | logs | stop
```

Logs: `/home/zni/c12/zni_worktrees/deploy_logs/` (`workers.log` = merged rank log, `frontend.<task>.<ts>.log`,
`build_device_tree.log`, `probe/`). Script knobs as on C12: `H3_SKIP_BUILD=1`, `H3_FORCE_CHECKOUT=1`, `H3_FORCE_ENV=1`, `H3_API_KEY`.

## Readiness, stop, reset on OM

* **Ready** = `SHM bridge ready` in the rank log and `/tt-liveness` reporting `"canary_state": "healthy"` (`H3_CANARY=true`,
  default) or `"model_ready": true` (`H3_CANARY=false`). Start -> ready on C12: ~40 s-2 min with a warm kernel cache, ~12 min cold
  (C12 figures). The kernel cache is per host (`TT_METAL_CACHE` unset -> `~/.cache/tt-metal-cache`, plus each tree's `built/`), so
  on Quad3 every host's first start is cold. Ignore `TT_FATAL: DRAM Auto slice could not find valid slice configuration` lines.
* **Stop** (`h3_deploy.sh stop`): TERM then KILL the frontend on `.120`, `video_runner` then `prted` on every host, then wait until
  no `/dev/tenstorrent/*` handle is held (`fuser`), then 5 s for the driver -- relaunching earlier fails with
  `Query mappings failed ... No such device`. With `H3_SHARED_FS=0`, `start` prints a one-line reminder that `replicate` must have
  run after the last setup/edit.
* **Reset** (`h3_deploy.sh reset`, `H3_RESET_MODE=glx_reset_auto`): `stop`, then `timeout 400 tt-smi -glx_reset_auto` over ssh on
  every host in parallel, rc reported per host, then a 60 s settle (~2 min per host, then ~45-60 s for the inter-host links to
  retrain -- expected (Quad1/Quad2)). One rc != 0 -> the action returns 1 with the OM hint: **FRB2/FRB3 tray hang -> BMC power
  cycle, do not re-reset**. The overlay is a plain `mount` (nothing in fstab), so after a host reboot or power cycle it is gone:
  re-run `om_quad3_prep.sh` before `start`.
* **Ownership vs the overlay.** Never `chown -R` (or otherwise touch metadata) across a mounted `cache-merged`: overlayfs copies a
  lower file up on any metadata change, and the lower is the 635 GB NFS cache -- it would land in `cache-upper` on the local `/`
  (507-750 GB free). Fix ownership on the sibling only (`sudo chown -R zni:zni /home/zni/c12/media_python_env`, `sudo chown zni:zni
  /home/zni/c12`); `copy_media_env_from_quad1.sh` chowns only what it copied, and with `--with-cache` refuses while the overlay is
  mounted -- `sudo umount /home/zni/c12/cache-merged` on `.120` first, `om_quad3_prep.sh` re-mounts it.
* **Requests time out or hang** with `device timeout ... unrecoverable` / `Timed out while waiting for active ethernet core`
  -> `reset`, then `start` again.

## Requests

```bash
URL=http://172.16.104.120:8000; KEY=${H3_API_KEY:-your-secret-key}
B64=$(base64 -w0 keyframe_1344x768.jpg)
curl -s -X POST $URL/v1/videos/generations/i2v -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d "{\"prompt\":\"A calm seaside village at golden hour, gentle waves\",\"aspect_ratio\":\"16:9\",\"duration_seconds\":5,\"seed\":7,
       \"image_prompts\":[{\"image\":\"$B64\",\"frame_pos\":0}]}"
# endpoints on this server version: t2va -> /v1/videos/generations, fl2va -> /v1/videos/generations/i2v, ref2va -> /v1/videos/generations/ref2va
# frame_pos 0 = first keyframe, -1 = last keyframe (both allowed); 4-15 s; aspect 16:9 21:9 4:3 1:1 3:4 9:16
curl -s -H "Authorization: Bearer $KEY" $URL/v1/videos/generations/<id>                 # status
curl -s -H "Authorization: Bearer $KEY" $URL/v1/videos/generations/<id>/download -o out.mp4
```
For t2va / ref2va: `start t2va` / `start ref2va` (`MODEL_RUNNER` and the frontend `MODEL` follow the task). Clean clips sit at
0.15-0.45 bits/pixel/frame with audio around -35..-45 dB mean; noise/scramble shows up as > 0.6 bits/pixel/frame and audio at 0 dB;
frame 0 vs the keyframe should correlate at > 0.95 (C12 figures, what `probe` checks). C12 request times (20 steps): 16:9 5 s
~25 s warm, 10 s ~35 s, 15 s ~50-55 s, first request per padded length +5-30 s for the compile -- expected to be similar here, not measured.

## What is not verified yet on Quad3

* **Nothing has executed on Quad3.** Everything above comes from read-only probes on 2026-09-10; the device-tree build of
  `162a86b008a` on the F hosts, the overlay, `replicate`, `reset`, the whole recipe -- all first runs.
* The `H3_BASE_ENV` / `H3_RESET_MODE` / `H3_SHARED_FS` / rankfile / `H3_CANARY` / symlink-hygiene hooks in `h3_deploy.sh` are new
  and have not run on any quad yet (C12 defaults are unchanged in effect).
* `kubectl` on `.120` (the join server `.118` is down); whether cordoning the F nodes is enough to keep the DRA driver off the chips.
* Whether the `34260b25483` pipeline reuses gbh-e4-02's `cache-upper` entries (keyed by module names like
  `transformer_resident_adaln`, converted for metal `94cff3e426a`). Seeding is optional; a miss re-converts (~80 GB per host).
* `apt-get install ffmpeg` on the F hosts (`apt-cache policy` shows candidate `7:4.4.2-0ubuntu0.22.04.1`; not installed yet).
* The `tt-inference-server` clone over public https from `.120` (tt-metal is public; GitHub/PyPI are reachable from the F hosts).
* `recover.sh --factory-descriptor-path` with the 16-host factory descriptor for a 4-host subset -- not used by this recipe, untested.
* Timings: build ~5 min, reset ~2 min/host + link retrain 45-60 s, rsync over 100 GbE -- all expected (Quad1/Quad2), none measured here.
