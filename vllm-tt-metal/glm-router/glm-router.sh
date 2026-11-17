#!/usr/bin/env bash
# glm-router: vllm-router (cache_aware) in front of every Galaxy's serve. Settings: ENV_FILE (default
# ~/glm-router/glm-router.env; see glm-router.env.example). Commands:
#   run                 the router in the foreground (systemd or the tmux loop run this); exits with it
#   start | stop        a detached tmux session INSTANCE that reruns `run` 2 s after any exit (stop: SIGTERM)
#   retire [SECS]       stop for good: SIGTERM the live loop (its router stops accepting, finishes its streams, and the
#                       loop does not restart it), then wait up to SECS (default 7200) and report when it has exited
#   handoff [BIN]       a new router (BIN, default ROUTER_BIN) takes over PORT while the old one finishes its streams:
#                       the new one runs in the other loop session (INSTANCE <-> INSTANCE-b) on the other metrics
#                       port (METRICS_PORT <-> METRICS_PORT_B), then the old loop gets SIGTERM (it does not restart).
#                       With REUSEPORT=1 and an old router that also shares the port (started by this script with
#                       REUSEPORT=1), both listen until the old one closes: no refused connection. Otherwise (the
#                       first switch from a router without SO_REUSEPORT) the old listener closes first and the new one
#                       binds as soon as it is free: connections are refused for the new router's start time.
#   metrics             the live router's metrics port (it moves at each handoff; collectors read it here or from
#                       LOGDIR/slot: "SESSION METRICS_PORT REUSEPORT")
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
# Instances: several routers can run on one host, each with its own ENV_FILE, INSTANCE (tmux session name, default
# glm-router), PORT, METRICS_PORT and METRICS_PORT_B, LOGDIR (slot, logs, check state) and WORKERS_FILE (add/drain
# change only that instance's list).
set -uo pipefail
ENV_FILE=${ENV_FILE:-$HOME/glm-router/glm-router.env}
[[ -r $ENV_FILE ]] || { echo "glm-router: no settings file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LOGDIR"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGDIR/glm-router.log" >&2; }
key() { cat "$KEY_FILE"; }
METRICS_PORT_B=${METRICS_PORT_B:-$((METRICS_PORT + 1))}
INSTANCE=${INSTANCE:-glm-router}
SESSIONS=("$INSTANCE" "$INSTANCE-b")
# The live loop session and its metrics port (handoff alternates them); the default before any handoff.
SLOT_FILE=$LOGDIR/slot
slot() {
    local s m r
    read -r s m r < "$SLOT_FILE" 2>/dev/null || { s=$INSTANCE m=$METRICS_PORT r=0; }
    echo "${s:-$INSTANCE} ${m:-$METRICS_PORT} ${r:-0}"
}
read -r SESSION LIVE_METRICS LIVE_REUSEPORT < <(slot)
# The PIDs listening on PORT (on any address).
listeners() { ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u; }
loop_cmd() {  # SESSION METRICS REUSEPORT [HANDOFF_FROM]: the restart loop's tmux command
    printf '%q ' env ENV_FILE="$ENV_FILE" RUN_METRICS_PORT="$2" RUN_REUSEPORT="$3" HANDOFF_FROM="${4:-}" \
        ${ROUTER_BIN_OVERRIDE:+ROUTER_BIN_OVERRIDE="$ROUTER_BIN_OVERRIDE"} bash -c \
        "trap 'kill -TERM \$c 2>/dev/null; wait \$c; exit 0' TERM; while :; do bash '$HERE/glm-router.sh' run & c=\$!; wait \$c; echo \"\$(date -u +%FT%TZ) router exited \$? ($1), restarting in 2 s\" >> '$LOGDIR/glm-router.log'; sleep 2; done"
}
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
    # One router per PORT: refuse when another process listens there, except the one a handoff replaces.
    others=$(listeners | grep -vxF "${HANDOFF_FROM:-none}")
    if [[ -n $others ]]; then
        log "refusing to start: PORT $PORT already has a listener (pid $(echo $others)); use handoff to replace it"
        exit 2
    fi
    mapfile -t W < <(active)
    ((${#W[@]})) || { log "no undrained workers in $WORKERS_FILE"; exit 2; }
    ROUTER_BIN=${ROUTER_BIN_OVERRIDE:-$ROUTER_BIN}
    args=(--host "$BIND" --port "$PORT" --prometheus-port "${RUN_METRICS_PORT:-$METRICS_PORT}" --worker-urls "${W[@]}"
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
    # The patched router's switches, when set in ENV_FILE: media placeholders and random ties
    # (router-media-placeholders.patch), retry-exclude (router-retry-exclude.patch); both in this directory.
    for v in VLLM_ROUTER_MEDIA_PLACEHOLDERS VLLM_ROUTER_RANDOM_TIES VLLM_ROUTER_RETRY_EXCLUDE_FAILED; do
        [[ -n ${!v:-} ]] && export "$v"
    done
    # SO_REUSEPORT on PORT (router-reuseport.patch in this directory), so a later handoff shares the port.
    if [[ ${RUN_REUSEPORT:-${REUSEPORT:-0}} == 1 ]]; then export VLLM_ROUTER_REUSEPORT=1; else unset VLLM_ROUTER_REUSEPORT; fi
    log "router: $ROUTER_BIN on $BIND:$PORT metrics ${RUN_METRICS_PORT:-$METRICS_PORT} over ${W[*]} (cache_aware thr $CACHE_THRESHOLD abs $SLOTS_PER_WORKER rel $BALANCE_REL, validate $VALIDATE, media placeholders ${VLLM_ROUTER_MEDIA_PLACEHOLDERS:-0}, random ties ${VLLM_ROUTER_RANDOM_TIES:-0}, retry exclude ${VLLM_ROUTER_RETRY_EXCLUDE_FAILED:-0}, reuseport ${VLLM_ROUTER_REUSEPORT:-0})"
    exec "$ROUTER_BIN" "${args[@]}" >> "$LOGDIR/router.log" 2>&1
    ;;
start)
    for s in "${SESSIONS[@]}"; do
        tmux has-session -t "=$s" 2>/dev/null && { echo "$s already runs"; exit 1; }
    done
    # LOGDIR belongs to one instance (its slot file names that instance's sessions).
    [[ $SESSION == "$INSTANCE" || $SESSION == "$INSTANCE-b" ]] ||
        { echo "LOGDIR $LOGDIR belongs to another instance ($SESSION): give this one its own LOGDIR"; exit 1; }
    [[ -z $(listeners) ]] || { echo "PORT $PORT already has a listener (pid $(listeners | tr '\n' ' '))"; exit 1; }
    ss -Hltn "sport = :$METRICS_PORT" | grep -q . && { echo "metrics port $METRICS_PORT is taken"; exit 1; }
    r=${REUSEPORT:-0}
    tmux new-session -d -s "$INSTANCE" "$(loop_cmd "$INSTANCE" "$METRICS_PORT" "$r")"
    echo "$INSTANCE $METRICS_PORT $r" > "$SLOT_FILE"
    echo "started tmux session $INSTANCE (PORT $PORT, metrics $METRICS_PORT, reuseport $r)"
    ;;
retire)
    loop=$(tmux list-panes -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null) || { echo "$SESSION is not running"; exit 1; }
    rpid=$(pgrep -P "$loop" -n)
    log "retire: SIGTERM to $SESSION (loop $loop, router ${rpid:-none}); it finishes its streams and does not restart"
    kill -TERM "$loop"
    t0=$(date +%s)
    while tmux has-session -t "=$SESSION" 2>/dev/null || { [[ -n $rpid ]] && kill -0 "$rpid" 2>/dev/null; }; do
        (($(date +%s) - t0 > ${2:-7200})) && { log "retire: $SESSION still running after ${2:-7200} s"; exit 1; }
        sleep 1
    done
    log "retire: $SESSION exited after $(($(date +%s) - t0)) s; PORT $PORT listeners: $(listeners | tr '\n' ' ')"
    ;;
metrics)
    echo "$LIVE_METRICS"
    ;;
stop)
    pid=$(tmux list-panes -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null) || { echo "$SESSION is not running"; exit 1; }
    kill -TERM "$pid"
    echo "SIGTERM sent to $SESSION"
    ;;
handoff)
    [[ -z ${2:-} || -x $2 ]] || { echo "not an executable: $2"; exit 2; }
    oldloop=$(tmux list-panes -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null) || { echo "$SESSION is not running"; exit 1; }
    oldpid=$(pgrep -P "$oldloop" -n) && [[ $(listeners) == "$oldpid" ]] ||
        { echo "the listener on PORT $PORT ($(listeners | tr '\n' ' ')) is not $SESSION's router ($oldpid)"; exit 1; }
    if [[ $SESSION == "$INSTANCE" ]]; then new=$INSTANCE-b m=$METRICS_PORT_B; else new=$INSTANCE m=$METRICS_PORT; fi
    tmux has-session -t "=$new" 2>/dev/null && { echo "$new already runs (a handoff in progress?)"; exit 1; }
    ss -Hltn "sport = :$m" | grep -q . && { echo "metrics port $m is taken"; exit 1; }
    # The new router waits (up to 600 s) for every start-list worker's /health before it listens.
    for w in $(active); do
        [[ $(curl -s -o /dev/null -w '%{http_code}' -m 5 "$w/health") == 200 ]] || { echo "$w /health is not 200: no handoff"; exit 1; }
    done
    r=${REUSEPORT:-0}
    export ROUTER_BIN_OVERRIDE=${2:-}
    if [[ $LIVE_REUSEPORT == 1 && $r == 1 ]]; then
        # Both share PORT: the new one listens, then the old one stops accepting and finishes its streams.
        log "handoff (shared port): $SESSION pid $oldpid -> $new on metrics $m"
        tmux new-session -d -s "$new" "$(loop_cmd "$new" "$m" 1 "$oldpid")"
        for _ in $(seq 1 600); do
            [[ $(listeners | grep -vxF "$oldpid") ]] && break
            sleep 0.1
        done
        newpid=$(listeners | grep -vxF "$oldpid")
        [[ -n $newpid ]] || { log "handoff: $new did not listen in 60 s; stopping it, $SESSION keeps serving"
            kill -TERM "$(tmux list-panes -t "=$new" -F '#{pane_pid}')"; exit 1; }
        echo "$new $m 1" > "$SLOT_FILE"
        kill -TERM "$oldloop"
    else
        # The old router does not share PORT: close its listener first, then the new one binds as soon as it is
        # free. Connections are refused from the old listener's close to the new bind (the new router's start).
        log "handoff (port not shared: refused connections until the new router binds): $SESSION pid $oldpid -> $new on metrics $m"
        echo "$new $m $r" > "$SLOT_FILE"
        kill -TERM "$oldloop"
        for _ in $(seq 1 100); do [[ -z $(listeners) ]] && break; sleep 0.02; done
        tmux new-session -d -s "$new" "$(loop_cmd "$new" "$m" "$r")"
        t0=$(date +%s.%N)
        for _ in $(seq 1 600); do [[ $(listeners) ]] && break; sleep 0.05; done
        newpid=$(listeners)
        [[ -n $newpid ]] || { log "handoff: nothing listens on PORT $PORT 30 s after the switch"; exit 1; }
        log "handoff: $new pid $newpid listens $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }') s after its start"
    fi
    log "handoff: $new pid $newpid serves PORT $PORT; $SESSION pid $oldpid finishes its streams and exits (no restart)"
    ;;
status)
    for s in "${SESSIONS[@]}"; do
        tmux has-session -t "=$s" 2>/dev/null && echo "$s: running (tmux)$([[ $s == "$SESSION" ]] && echo ", live: metrics $LIVE_METRICS reuseport $LIVE_REUSEPORT")"
    done
    echo "listening on PORT $PORT: pid $(listeners | tr '\n' ' ')"
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
    now=$(curl -s -m 5 "http://127.0.0.1:$LIVE_METRICS/metrics" | grep 'prefix_routing_decisions_total{' |
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
