#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# run_live_test.sh -- run tests/test_minimax_h3_live.py against a MiniMax-H3 deployment.
#
# Works from any checkout, for any user: every path is derived from where this
# script lives or taken from the environment. The full set owns the whole mesh for
# about 1h20 -- run it inside tmux.
#
#   bash tests/run_live_test.sh                     # everything: fl2va + ref2va, 1 compile + 3 gens per combo
#   bash tests/run_live_test.sh -k Fl2va            # one class only
#   bash tests/run_live_test.sh -k "img6 or vid1"   # chosen combinations
#   H3_LIVE_REPEATS=1 bash tests/run_live_test.sh   # fewer warm generations
#   bash tests/run_live_test.sh --collect-only -q   # list the cases, touches nothing
#
# Environment (all optional):
#   H3_LIVE_URL         server, default http://localhost:8000
#   H3_API_KEY          bearer token, default your-secret-key
#   H3_DEPLOY_DIR       directory holding h3ctl.sh (and optionally ref2va-assets/, videos/).
#                       Default: the first of <checkout>/.., $HOME/h3-deploy that has an
#                       h3ctl.sh. Set it to "" to run WITHOUT deployment control: no worker
#                       restarts, no chip resets; the served task is probed, the other skipped.
#   H3_CTL              deployment control script, default $H3_DEPLOY_DIR/h3ctl.sh
#                       (must offer: start <task>, wait-ready <s>, reset)
#   H3_LIVE_ASSETS_DIR  reference assets; default $H3_DEPLOY_DIR/ref2va-assets when present,
#                       otherwise the tests synthesise them with ffmpeg
#   H3_LIVE_VIDEO_DIR   the server's TT_VIDEO_OUTPUT_DIR, enables the on-disk checks; default
#                       $TT_VIDEO_OUTPUT_DIR, else $H3_DEPLOY_DIR/videos when present. Only
#                       meaningful when the server runs on this host.
#   H3_LIVE_OUT_DIR     logs, JSON report, downloaded mp4s; default $H3_DEPLOY_DIR/live-test,
#                       else ${TMPDIR:-/tmp}/h3-live-test-$USER
#   H3_LIVE_REPEATS     warm generations after the compile request, default 3
#   PYTHON              interpreter; default: the active venv, else <tt-media-server>/python_env,
#                       else python3 on PATH
set -u

SELF=$(readlink -f "${BASH_SOURCE[0]}")                   # follow symlinks to the checkout
TMS=$(cd "$(dirname "$SELF")/.." && pwd)                  # tt-media-server/
TS=$(date +%Y%m%d_%H%M%S)

# --- deployment control -------------------------------------------------------------
if [ -z "${H3_DEPLOY_DIR+x}" ]; then                         # unset (not empty): auto-detect
  for cand in "$TMS/../.." "$HOME/h3-deploy"; do
    if [ -x "$cand/h3ctl.sh" ]; then H3_DEPLOY_DIR=$(cd "$cand" && pwd); break; fi
  done
  H3_DEPLOY_DIR=${H3_DEPLOY_DIR:-}
fi
H3_CTL=${H3_CTL:-${H3_DEPLOY_DIR:+$H3_DEPLOY_DIR/h3ctl.sh}}
if [ -n "$H3_CTL" ] && [ ! -x "$H3_CTL" ]; then
  echo "H3_CTL=$H3_CTL is not executable; set H3_DEPLOY_DIR or H3_CTL, or H3_DEPLOY_DIR= for no control" >&2
  exit 2
fi

# --- derived defaults ------------------------------------------------------------------
: "${H3_LIVE_URL:=http://localhost:8000}"
: "${H3_LIVE_REPEATS:=3}"
if [ -z "${H3_LIVE_ASSETS_DIR:-}" ] && [ -n "$H3_DEPLOY_DIR" ] && [ -d "$H3_DEPLOY_DIR/ref2va-assets" ]; then
  H3_LIVE_ASSETS_DIR=$H3_DEPLOY_DIR/ref2va-assets
fi
if [ -z "${H3_LIVE_VIDEO_DIR:-}" ]; then
  if [ -n "${TT_VIDEO_OUTPUT_DIR:-}" ]; then H3_LIVE_VIDEO_DIR=$TT_VIDEO_OUTPUT_DIR
  elif [ -n "$H3_DEPLOY_DIR" ] && [ -d "$H3_DEPLOY_DIR/videos" ]; then H3_LIVE_VIDEO_DIR=$H3_DEPLOY_DIR/videos
  fi
fi
OUT=${H3_LIVE_OUT_DIR:-${H3_DEPLOY_DIR:+$H3_DEPLOY_DIR/live-test}}
OUT=${OUT:-${TMPDIR:-/tmp}/h3-live-test-${USER:-$(id -un)}}
mkdir -p "$OUT" || exit 1
LOG=$OUT/live-test.$TS.log
REPORT=$OUT/live-test.$TS.json

# --- interpreter ------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then PYTHON=python
  elif [ -x "$TMS/python_env/bin/python" ]; then PYTHON=$TMS/python_env/bin/python
  else PYTHON=python3
  fi
fi
"$PYTHON" -c 'import pytest' 2>/dev/null || { echo "$PYTHON has no pytest; set PYTHON to an interpreter that does" >&2; exit 2; }

# --- refuse to start over somebody else's run (one job at a time on the mesh) -----------
if pgrep -f "pytest.*test_minimax_h3_live" >/dev/null; then
  echo "another live test run is in progress (pgrep -f test_minimax_h3_live); wait for it" >&2
  exit 2
fi
queue=$(curl -s -m 5 "$H3_LIVE_URL/tt-liveness" | "$PYTHON" -c 'import sys,json; print(json.load(sys.stdin).get("queue_size", 0))' 2>/dev/null)
if [ -n "$queue" ] && [ "$queue" != "0" ]; then
  echo "$H3_LIVE_URL reports queue_size=$queue: a generation is in flight; wait for it" >&2
  exit 2
fi

# --- run -----------------------------------------------------------------------------------
export HF_TOKEN= MODEL= JWT_SECRET=          # the API module reads these at import; keep them inert
export H3_LIVE_URL H3_LIVE_REPEATS
export H3_LIVE_API_KEY=${H3_API_KEY:-your-secret-key}
[ -n "${H3_LIVE_ASSETS_DIR:-}" ] && export H3_LIVE_ASSETS_DIR
[ -n "${H3_LIVE_VIDEO_DIR:-}" ] && export H3_LIVE_VIDEO_DIR
if [ -n "$H3_CTL" ]; then
  export H3_LIVE_START_CMD="bash $H3_CTL start {task}"
  export H3_LIVE_WAIT_CMD="bash $H3_CTL wait-ready 1800"
  export H3_LIVE_RESET_CMD="bash $H3_CTL reset"
fi
export H3_LIVE_REPORT=$REPORT

echo "server:   $H3_LIVE_URL"
echo "control:  ${H3_CTL:-none (served task is probed; no restarts, no resets)}"
echo "assets:   ${H3_LIVE_ASSETS_DIR:-synthesised with ffmpeg}"
echo "disk chk: ${H3_LIVE_VIDEO_DIR:-off}"
echo "python:   $PYTHON"
echo "log:      $LOG"
echo "report:   $REPORT"
cd "$TMS" || exit 1
"$PYTHON" -m pytest tests/test_minimax_h3_live.py -s -v -p no:cacheprovider \
  -o asyncio_default_fixture_loop_scope=function \
  --basetemp="$OUT/pytest-tmp.$TS" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "pytest exit=$rc   log: $LOG   report: $REPORT"
exit "$rc"
