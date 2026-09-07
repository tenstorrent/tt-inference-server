#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Wedge recovery: restart ltx-media-server if it has been active past the warmup
# grace but /health is not 200. With CANARY_GATE_READINESS=true, a hung device
# makes /health return non-200 while the process stays alive — Restart=on-failure
# can't see that, so this timer-driven check catches it.
set -uo pipefail

SVC="ltx-media-server.service"
PORT="${PORT:-8000}"
GRACE_SEC="${GRACE_SEC:-900}"   # skip while still warming up (~10 min + buffer)

# Only act if the unit is active (don't fight a manual stop / failed state).
[ "$(systemctl is-active "$SVC" 2>/dev/null)" = "active" ] || exit 0

# Seconds since the service became active.
active_str="$(systemctl show -p ActiveEnterTimestamp --value "$SVC" 2>/dev/null)"
active_ts="$(date -d "$active_str" +%s 2>/dev/null || echo 0)"
now_ts="$(date +%s)"
up=$(( now_ts - active_ts ))
[ "$active_ts" -eq 0 ] && exit 0
[ "$up" -lt "$GRACE_SEC" ] && exit 0   # still within warmup grace — don't restart

code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)"
if [ "$code" != "200" ]; then
  logger -t ltx-healthcheck "health=${code} up=${up}s past grace → restarting ${SVC}"
  systemctl restart "$SVC"
fi
