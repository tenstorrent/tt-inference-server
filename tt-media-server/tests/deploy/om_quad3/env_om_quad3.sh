# env_om_quad3.sh -- the H3_* overrides that make tests/deploy/c12_quad/h3_deploy.sh run on OM Quad3 (racks F1/F2).
# `source` it before EVERY h3_deploy.sh call, on the launch node (172.16.104.120 = OM-F1-GBH01) as zni:
#   source /home/zni/c12/env_om_quad3.sh && <checkout>/tt-media-server/tests/deploy/c12_quad/h3_deploy.sh setup
# Nothing here touches the C12 pins or knobs; it only moves paths and hosts and picks the OM reset / replication mode.
# Every line carries the reason it differs from C12 (README.md next to this file has the same list in prose).
#
# This file IS the Quad3 cluster fact list, so paths are spelled /home/zni/... on purpose, not $HOME: MPI needs the SAME
# absolute tree on all four hosts, and this file is what makes that path the same everywhere. Never put a uid in here --
# zni's uid differs between hosts (1001 is taken by `user` on the F hosts, so zni gets a system-assigned one there).

# /home is local ext4 per host and the writable NFS (/data_bh) belongs to ubuntu, so the deployment root is a LOCAL path
# that exists identically on all four hosts; `h3_deploy.sh replicate` copies it from rank 0 to the three peers.
export H3_VM=/home/zni/c12
export H3_WT=$H3_VM/zni_worktrees              # the three pinned worktrees + env file live here so replicate carries them; deploy_logs/ is here too but stays per host (excluded)
export H3_METAL_REPO=$H3_VM/tt-metal           # local tt-metal clone, remote 'origin' = github; setup fetches zni/h3-c12-clean-device-tree itself
export H3_TIS_REPO=$H3_VM/tt-inference-server  # local tt-inference-server clone; setup fetches sadesoye/add_h3_fl2va_ref2va itself
# No media-server venv exists on the F hosts: this one is copied from gbh-e4-02 by copy_media_env_from_quad1.sh (relocatable
# uv venv with its own interpreter inside; bin/tt-run, bin/uvicorn, bin/ffmpeg). Lives under $H3_VM so replicate carries it.
export H3_MEDIA_ENV=$H3_VM/media_python_env
# IPs, not names: on OM a hostname resolves to 127.0.1.1 on itself and MPI misidentifies the local rank. IP-ascending = rank
# order; rank 0 = launch node = frontend host = .120 (OM-F1-GBH01), so the API is http://172.16.104.120:8000.
export H3_HOSTS=172.16.104.120,172.16.104.121,172.16.104.122,172.16.104.123
export H3_RANK0=172.16.104.120
# OM has no rankfile: om_quad3_prep.sh writes this one (rank i = host i of H3_HOSTS, "slot=0:*"); verify_setup generates it too
# when it is missing and H3_HOSTS is set. Under $H3_VM because tt-run resolves it relative to mpirun's cwd (`cd $VM` in start).
export H3_RANKFILE=$H3_VM/quad3_rankfile
# No metal_env_H3.sh anywhere on OM: base_env_om.sh is its replacement (weights on /data_bh/h3, the DiT cache overlay, MPI,
# fabric timeouts, SHM names). The generated env file `source`s it first and sets TT_METAL_HOME/PYTHONPATH/LD_LIBRARY_PATH itself.
export H3_BASE_ENV=$H3_VM/base_env_om.sh
# No 4x32 cabling/deployment descriptors on OM (/data_bh/scaleout_configs has only a 16-host factory descriptor), so recover.sh
# cannot run here; the OM reset is `tt-smi -glx_reset_auto` on every host, ~2 min each, then ~45-60 s for the links to retrain.
# NEVER `tt-smi -r` on a Galaxy host. H3_DESC_DIR / H3_DESC_PREFIX are therefore deliberately NOT set.
export H3_RESET_MODE=glx_reset_auto
# No shared filesystem between the hosts: `start` reminds you that `replicate` must have run after the last setup/edit.
export H3_SHARED_FS=0
# H3_CANARY stays at its default (true = C12 behaviour: the frontend runs CANARY_ENABLED=true and wait_ready needs
# "canary_state": "healthy"). The OM bare-metal stack runs the canary OFF on purpose: a cancelled job frees the API slot while
# the SP runner keeps generating, the canary probes a busy mesh, misses 3x in 12 s and declares the model DEAD -- that is what
# killed the Quad1/Quad2 k8s deployments. If you see that here, export H3_CANARY=false before sourcing; wait_ready then keys
# on "model_ready": true instead.
# H3_API_KEY is passed through untouched: export it before sourcing if the frontend key must differ from h3_deploy.sh's default.
if [ -n "${H3_API_KEY:-}" ]; then export H3_API_KEY; fi
