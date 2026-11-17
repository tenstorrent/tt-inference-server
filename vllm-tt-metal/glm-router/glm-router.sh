#!/usr/bin/env bash
# glm-router: vllm-router (cache_aware) in front of every Galaxy's serve. Settings: ENV_FILE (default
# ~/glm-router/glm-router.env; see glm-router.env.example). Commands:
#   run                 the router in the foreground (systemd or the tmux loop run this); exits with it
#   start | stop        a detached tmux session `glm-router` that reruns `run` 2 s after any exit (stop: SIGTERM)
#   status              the session, the live worker list against WORKERS_FILE, the last log lines
#   check               health check, exit 1 on a failure (and ALERT_CMD): router /health; the live list equals
#                       WORKERS_FILE's undrained workers; each worker's /health 200 and a render under 1 s; the share
#                       of cache_aware decisions without token ids since the last check under MISSING_TOKENS_MAX
#   workers             the live worker list
#   add URL             a new or returning worker: waits for its /health, adds it live and to WORKERS_FILE
#   drain URL [SECS]    no new requests for URL (removed live, marked "#drained" in WORKERS_FILE), then waits up to
#                       SECS (default 7200) for its running requests to end; it can then be stopped for maintenance
#   undrain URL         = add URL (its line loses the mark)
# Cache effect (cache_aware keeps a prefix tree per worker in router memory): add gives the new worker an empty tree,
# so it takes new prefixes by least load and earns continuations from then on; drain/remove drops that worker's tree,
# so its sessions' next turns land elsewhere and prefill again (its store's records stay on it, unused until it
# returns); a router restart drops every tree: until conversations come back, turns route by least load (like
# round-robin: ~9 points of hit share at N=2 for the first conversations' lengths, measured in our routing study).
set -uo pipefail
ENV_FILE=${ENV_FILE:-$HOME/glm-router/glm-router.env}
[[ -r $ENV_FILE ]] || { echo "glm-router: no settings file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LOGDIR"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGDIR/glm-router.log" >&2; }
key() { cat "$KEY_FILE"; }
active() { grep -vE '^\s*(#|$)' "$WORKERS_FILE"; }
base() { echo "http://${BIND/0.0.0.0/127.0.0.1}:$PORT"; }
api() {  # METHOD PATH: the router's API, with the key
    curl -s -m 30 -X "$1" -H "Authorization: Bearer $(key)" "$(base)$2"
}
case ${1:-status} in
run)
    if [[ $BIND != 127.0.0.1 && $BIND_APPROVED != 1 ]]; then
        log "refusing BIND=$BIND without BIND_APPROVED=1 (exposing the router is the user's decision)"
        exit 2
    fi
    if [[ $BIND != 127.0.0.1 && $VALIDATE != 1 ]]; then
        log "refusing BIND=$BIND without inbound auth (VALIDATE=1)"
        exit 2
    fi
    mapfile -t W < <(active)
    ((${#W[@]})) || { log "no undrained workers in $WORKERS_FILE"; exit 2; }
    args=(--host "$BIND" --port "$PORT" --prometheus-port "$METRICS_PORT" --worker-urls "${W[@]}"
        --policy cache_aware --cache-threshold "$CACHE_THRESHOLD" --balance-abs-threshold "$SLOTS_PER_WORKER"
        --balance-rel-threshold "$BALANCE_REL" --retry-max-retries "$RETRIES" --request-timeout-secs 7200
        --health-check-endpoint /health --health-check-interval-secs 10 --health-failure-threshold 3
        --log-level info)
    if [[ $VALIDATE == 1 ]]; then
        # VALIDATION_URLS (space-separated) overrides: the start list's /v1/models otherwise (a worker added later
        # is not one; any one that answers 200 validates a key).
        read -ra v <<< "${VALIDATION_URLS:-}"
        if ((${#v[@]} == 0)); then
            for w in "${W[@]}"; do v+=("$w/v1/models"); done
        fi
        args+=(--api-key-validation-urls "${v[@]}")
    fi
    # The patched router's switches (router-media-placeholders.patch in this directory), when set in ENV_FILE.
    for v in VLLM_ROUTER_MEDIA_PLACEHOLDERS VLLM_ROUTER_RANDOM_TIES; do
        [[ -n ${!v:-} ]] && export "$v"
    done
    log "router: $ROUTER_BIN on $BIND:$PORT over ${W[*]} (cache_aware thr $CACHE_THRESHOLD abs $SLOTS_PER_WORKER rel $BALANCE_REL, validate $VALIDATE, media placeholders ${VLLM_ROUTER_MEDIA_PLACEHOLDERS:-0}, random ties ${VLLM_ROUTER_RANDOM_TIES:-0})"
    exec "$ROUTER_BIN" "${args[@]}" >> "$LOGDIR/router.log" 2>&1
    ;;
start)
    tmux has-session -t glm-router 2>/dev/null && { echo "glm-router already runs"; exit 1; }
    tmux new-session -d -s glm-router "$(printf '%q ' env ENV_FILE="$ENV_FILE" bash -c \
        "trap 'kill -TERM \$c 2>/dev/null; wait \$c; exit 0' TERM; while :; do bash '$HERE/glm-router.sh' run & c=\$!; wait \$c; echo \"\$(date -u +%FT%TZ) router exited \$?, restarting in 2 s\" >> '$LOGDIR/glm-router.log'; sleep 2; done")"
    echo "started tmux session glm-router"
    ;;
stop)
    pid=$(tmux list-panes -t glm-router -F '#{pane_pid}' 2>/dev/null) || { echo "glm-router is not running"; exit 1; }
    kill -TERM "$pid"
    echo "SIGTERM sent"
    ;;
status)
    tmux has-session -t glm-router 2>/dev/null && echo "glm-router: running (tmux)" || echo "glm-router: no tmux session"
    echo "live workers: $(api GET /list_workers)"
    echo "workers file: $(active | tr '\n' ' ') drained: $(grep -c '^#drained ' "$WORKERS_FILE")"
    tail -5 "$LOGDIR/glm-router.log"
    ;;
workers)
    api GET /list_workers
    echo
    ;;
check)
    fails=()
    [[ $(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $(key)" "$(base)/health") == 200 ]] ||
        fails+=("router /health")
    live=$(api GET /list_workers)
    for w in $(active); do
        grep -q "\"$w\"" <<< "$live" || fails+=("$w not in the router's list")
        [[ $(curl -s -o /dev/null -w '%{http_code}' -m 5 "$w/health") == 200 ]] || fails+=("$w /health")
        t=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' -m 5 -H "Authorization: Bearer $(key)" \
            -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"ping"}]}' \
            "$w/v1/chat/completions/render")
        [[ ${t% *} == 200 ]] && awk -v s="${t#* }" 'BEGIN { exit !(s < 1.0) }' || fails+=("$w render ($t)")
    done
    # Decisions since the last check (missing_tokens: no token ids from render, so least-loaded instead).
    now=$(curl -s -m 5 "http://127.0.0.1:$METRICS_PORT/metrics" | grep 'prefix_routing_decisions_total{' |
        sed -E 's/.*outcome="([a-z_]+)"} ([0-9.e+]+)/\1 \2/')
    prev=$(cat "$LOGDIR/decisions.last" 2>/dev/null)
    echo "$now" > "$LOGDIR/decisions.last"
    share=$(python3 -c '
import sys
def parse(t):
    d = {}
    for l in t.splitlines():
        k, _, v = l.partition(" ")
        if v:
            d[k] = d.get(k, 0) + float(v)
    return d
a, b = parse(sys.argv[1]), parse(sys.argv[2])
delta = {k: b[k] - a.get(k, 0) for k in b}
total = sum(v for v in delta.values() if v > 0)
print("%.4f %d" % (delta.get("missing_tokens", 0) / total if total else 0.0, total))' "$prev" "$now")
    awk -v s="${share% *}" -v m="$MISSING_TOKENS_MAX" 'BEGIN { exit !(s <= m) }' ||
        fails+=("cache_aware decisions without token ids: ${share% *} of ${share#* }")
    if ((${#fails[@]})); then
        log "check FAILED: ${fails[*]}"
        [[ -n ${ALERT_CMD:-} ]] && $ALERT_CMD "glm-router check failed: ${fails[*]}"
        exit 1
    fi
    echo "check ok: $(active | wc -l) workers; decisions since the last check: ${share#* }, missing tokens ${share% *}"
    ;;
add | undrain)
    url=${2:?URL}
    for _ in $(seq 1 120); do curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/health" | grep -q 200 && break; sleep 5; done
    out=$(api POST "/add_worker?url=$url")
    log "add $url: $out"
    sed -i "\#^\#drained $url\$#d; \#^$url\$#d" "$WORKERS_FILE"
    echo "$url" >> "$WORKERS_FILE"
    ;;
drain | remove)
    url=${2:?URL}
    out=$(api POST "/remove_worker?url=$url")
    log "drain $url: $out"
    sed -i "s#^$url\$#\#drained $url#" "$WORKERS_FILE"
    t0=$(date +%s)
    while :; do
        n=$(curl -s -m 5 "$url/metrics" | awk '/^vllm:num_requests_running/ { print $2 }')
        [[ ${n:-0} == 0 ]] && break
        (($(date +%s) - t0 > ${3:-7200})) && { log "drain $url: $n requests still running after ${3:-7200} s"; exit 1; }
        sleep 5
    done
    log "drain $url: idle after $(($(date +%s) - t0)) s; it takes no router traffic and may be stopped"
    ;;
*)
    sed -n '2,20p' "$0" >&2
    exit 2
    ;;
esac
