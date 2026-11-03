#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Standalone unscored fixture; no Python dependency, installs, or source extraction.
set -eu
export LC_ALL=C
fixture=${1:?fixture-only expectation directory required}
[ "$(id -u)" = 0 ] || { echo 'fixture requires container root' >&2; exit 1; }
work=$(mktemp -d /tmp/offline-apt-probe.XXXXXX)
chmod 755 "$work"
state_hash() {
    if [ -e /app ]; then
        { find /app -xdev -printf '%P %m %U %G %l\n'; find /app -xdev -type f -exec sha256sum '{}' +; } | sort | sha256sum
    else
        echo absent
    fi
}
state_hash > "$work/app.before"
sha256sum /var/lib/dpkg/status > "$work/status.before"
cleanup() {
    status=$?
    trap - EXIT
    state_hash > "$work/app.after"
    if ! cmp -s "$work/app.before" "$work/app.after" || ! sha256sum -c "$work/status.before" >/dev/null; then
        echo 'fixture changed task state or installed status' >&2
        status=1
    fi
    # Preserve diagnostics to captured stderr before removing only this fixture tempdir.
    if [ "$status" != 0 ]; then
        for log in "$work"/*.stderr "$work/simulation" "$work/versions" "$fixture/selected-versions"; do
            if [ -f "$log" ]; then
                printf 'FIXTURE_FAILURE_EVIDENCE %s\n' "${log##*/}" >&2
                cat "$log" >&2
            fi
        done
        apt-get --version >&2
        apt-config dump >&2
    fi
    rm -rf -- "$work"
    if [ "$status" = 0 ]; then
        printf 'OFFLINE_APT_FIXTURE_PASS task=%s uid=%s gid=%s\n' "$(cat "$fixture/task")" "$uid" "$gid"
    fi
    exit "$status"
}
trap cleanup EXIT
now=$(date +%s)
deadline=$(cat "$fixture/deadline")
[ "$deadline" -gt "$now" ] || { echo 'expired fixture window' >&2; exit 1; }
while read -r published expires; do
    [ "$published" -le "$((now + 600))" ] || { echo 'future repository Date' >&2; exit 1; }
    [ "$expires" = 0 ] || [ "$expires" -gt "$deadline" ] || { echo 'repository expires before window end' >&2; exit 1; }
done < "$fixture/time-windows"
sha256sum -c "$fixture/original.sha256" > "$work/original-checks"
sha256sum -c "$fixture/cache.sha256" > "$work/cache-checks"
find /etc/apt/sources.list.d -mindepth 1 -maxdepth 1 -printf '%f\n' | sort > "$work/source-names"
cmp "$work/source-names" "$fixture/original-source-names"
if [ -f /etc/apt/sources.list ] && sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' /etc/apt/sources.list | grep -q .; then
    echo 'unexpected active standalone source list' >&2; exit 1
fi
# EROFS must be distinguishable from EACCES; root on a true read-only bind gets this errno text.
for mount in /opt/offline-repositories "$fixture/sources"; do
    if touch "$mount/.fixture-write-must-fail" 2> "$work/readonly.stderr"; then
        rm -f "$mount/.fixture-write-must-fail"
        echo 'writable cache/source mount' >&2; exit 1
    fi
    grep -q 'Read-only file system' "$work/readonly.stderr" || { echo 'write failed without EROFS' >&2; exit 1; }
done
uid=$(id -u _apt)
gid=$(id -g _apt)
# Actual acquisition identity reads all bytes, not only permissions from a stat call.
setpriv --reuid="$uid" --regid="$gid" --clear-groups sha256sum -c "$fixture/cache.sha256" > "$work/apt-user-checks"
mkdir -p "$work/lists/partial" "$work/archives/partial" "$work/source"
chown "$uid:$gid" "$work/source"
: > "$work/empty.sources"
apt_cmd() {
    apt-get -o "Dir::State::lists=$work/lists" -o "Dir::Cache::archives=$work/archives" \
        -o "Dir::State::extended_states=$work/extended_states" -o Debug::NoLocking=true \
        -o "Dir::Etc::sourcelist=$work/empty.sources" -o "Dir::Etc::sourceparts=$fixture/sources" \
        -o Acquire::Languages=none -o APT::Update::Error-Mode=any "$@"
}
apt_cmd update > "$work/update.stdout" 2> "$work/update.stderr"
if grep -Eq 'Failed to fetch|Some index files failed|Skipping acquire|unsandboxed as root' "$work/update.stderr"; then
    echo 'partial or unsandboxed apt acquisition' >&2; exit 1
fi
apt_cmd indextargets --format '$(URI)|$(FILENAME)|$(IDENTIFIER)' > "$work/targets"
: > "$work/acquired"
while IFS='|' read -r uri filename kind; do
    case "$kind" in Packages|Sources)
        [ -s "$filename" ] || { echo 'index not physically acquired' >&2; exit 1; }
        printf '%s\n' "$uri" | sed 's@^file:///@file:/@' >> "$work/acquired";;
    esac
done < "$work/targets"
sort -u "$work/acquired" > "$work/acquired.sorted"
while IFS= read -r uri; do
    grep -Fxq "$uri" "$work/acquired.sorted" || { echo "missing acquired index: $uri" >&2; exit 1; }
done < "$fixture/indexes"
case "$(cat "$fixture/task")" in
    build-pmars) set -- build-essential dpkg-dev;;
    configure-git-webserver) set -- git;;
    *) echo 'unsupported fixture task' >&2; exit 1;;
esac
apt_cmd --simulate install "$@" > "$work/simulation" 2> "$work/simulation.stderr"
sed -n 's/^Inst \([^ ]*\).* (\([^ ]*\) .*/\1 \2/p' "$work/simulation" | sort > "$work/versions"
cmp "$work/versions" "$fixture/selected-versions"
apt_cmd --download-only --yes install "$@" > "$work/download.stdout" 2> "$work/download.stderr"
if [ -f "$fixture/source-candidate" ]; then
    candidate=$(cat "$fixture/source-candidate")
    (cd "$work/source"; apt_cmd --download-only source "$candidate" > "$work/source.stdout" 2> "$work/source.stderr"; sha256sum -c "$fixture/source.sha256")
fi
for log in "$work/download.stderr" "$work/source.stderr"; do
    if [ -f "$log" ] && grep -Eq 'Failed to fetch|Some index files failed|Skipping acquire|unsandboxed as root' "$log"; then
        echo 'partial or unsandboxed artifact acquisition' >&2; exit 1
    fi
done
printf 'SCOPE=mandatory_acquisition_only NO_INSTALL=1 FULL_TASK_PARITY=0\n'
cat "$work/acquired.sorted" "$work/versions"
apt-config dump
