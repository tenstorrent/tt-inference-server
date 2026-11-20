#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Starts the video server under `tt-run` on ONE T3K, as if the mesh spanned several
# hosts: every rank is a process that sees only its own PCIe cards, and tt-metal's
# mesh graph descriptor joins them into one 2x4 mesh (host_topology 1x2). Rank 0 serves
# HTTP and runs the model in lockstep with the follower rank(s). Nothing is faster than on
# one host; the point is to run the whole multihost code path (rank detection, lockstep,
# follower, launcher watch) on the hardware that is available.
#
# Usage (from cpp_server):
#   tests/video/real_device/multihost/run_tt_run_t3k.sh \
#       --tt-metal <tt-metal> --dit-cache <tt_dit weight cache> --api-key <token> \
#       [--ranks 2] [--port 8000] [--output-dir <dir>] [--workdir <dir>]
#
# `--ranks 2` is the layout tt-metal itself tests on a T3K (cards {0,1} and {2,3}).
# `--ranks 4` gives each rank one card (--card-order, default 0 1 2 3); it needs a mesh graph
# descriptor with --host-topology (default "2, 2"; written here, tt-metal does not ship one).
# tt-metal rejects some splits ("Inconsistent host ranks"): the chips of one card must fall
# into one host of the topology.
#
# Rank 0's DEVICE_IDS must be the cards tt-run gives rank 0: the server sets
# TT_VISIBLE_DEVICES from DEVICE_IDS and would otherwise override the binding. The
# follower ranks never read DEVICE_IDS.
set -euo pipefail

TT_METAL=""; DIT_CACHE=""; API_KEY="your-secret-key"; RANKS=2; PORT=8000
OUTDIR=""; WORKDIR=""; HOST_TOPOLOGY="2, 2"; CARD_ORDER="0 1 2 3"
while [ $# -gt 0 ]; do
  case "$1" in
    --tt-metal) TT_METAL="$2"; shift 2;;
    --dit-cache) DIT_CACHE="$2"; shift 2;;
    --api-key) API_KEY="$2"; shift 2;;
    --ranks) RANKS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --output-dir) OUTDIR="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --host-topology) HOST_TOPOLOGY="$2"; shift 2;;   # 4 ranks only, e.g. "2, 2" or "1, 4"
    --card-order) CARD_ORDER="$2"; shift 2;;          # 4 ranks only: card of rank 0..3, e.g. "0 1 2 3"
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$TT_METAL" ] && [ -n "$DIT_CACHE" ] || { echo "--tt-metal and --dit-cache are required" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/../../../.." && pwd)"   # cpp_server
WORKDIR="${WORKDIR:-$(mktemp -d /tmp/tt-multihost-XXXXXX)}"
OUTDIR="${OUTDIR:-$WORKDIR/videos}"
mkdir -p "$WORKDIR" "$OUTDIR"

case "$RANKS" in
  2) CARDS=("0,1" "2,3"); MGD="$TT_METAL/tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.textproto"; RANK0_IDS="(0,1)";;
  4) read -r -a CARDS <<< "$CARD_ORDER"; MGD="$WORKDIR/t3k_quad_host_mesh_graph_descriptor.textproto"; RANK0_IDS="(0)"
     cat > "$MGD" <<MGDEOF
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] }
  host_topology   { dims: [ $HOST_TOPOLOGY ] }
  channels { count: 2 policy: RELAXED }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
MGDEOF
     ;;
  *) echo "--ranks must be 2 or 4" >&2; exit 2;;
esac

BIND="$WORKDIR/rank_bindings.yaml"
{
  echo "rank_bindings:"
  for i in $(seq 0 $((RANKS - 1))); do
    echo "  - rank: $i"
    echo "    mesh_id: 0"
    echo "    env_overrides:"
    echo "      TT_VISIBLE_DEVICES: \"${CARDS[$i]}\""
  done
  echo "mesh_graph_desc_path: \"$MGD\""
} > "$BIND"
echo "rank bindings: $BIND ($RANKS ranks), mesh descriptor: $MGD" >&2

# Every rank sets its own environment, then runs the server binary (a follower rank never
# binds the port). tt-run exports the variables it needs; the rest is set here.
SERVER_ENV="source '$TT_METAL/python_env/bin/activate' && cd '$HERE' && \
  export MODEL_SERVICE=video MODEL=Wan2.2-T2V-A14B-Diffusers MODEL_RUNNER=tt-wan2.2 DEVICE=t3k \
  DEVICE_IDS='$RANK0_IDS' TT_DIT_CACHE_DIR='$DIT_CACHE' TT_METAL_HOME='$TT_METAL' \
  HF_HUB_OFFLINE=1 OPENAI_API_KEY='$API_KEY' TT_VIDEO_OUTPUT_DIR='$OUTDIR' TT_VIDEO_TRACED=0 && \
  exec ./build/tt_media_server_cpp -p $PORT"

source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL"
cd "$TT_METAL"   # tt-run resolves relative paths against the launch directory
exec tt-run --rank-binding "$BIND" --mpi-args "--tag-output" bash -c "$SERVER_ENV"
