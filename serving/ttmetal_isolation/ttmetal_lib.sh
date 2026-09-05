# shellcheck shell=bash
# ttmetal_lib.sh - shared helpers for per-model tt-metal build isolation on Exabox.
#
# Core invariant enforced by everything here:
#   A bare-metal Quetzal/ttis serve sets TT_METAL_HOME == TT_METAL_RUNTIME_ROOT to a
#   tt-metal tree. That ONE tree is BOTH the built-lib location (build/lib/libtt_metal.so,
#   build*/lib/_ttnn.so) AND the JIT kernel-source root that TTNN compiles kernels from at
#   serve time. Therefore the git source HEAD of the tree MUST equal the commit its built
#   lib was compiled at. When they diverge (e.g. a later build git-checkouts a new commit
#   into a tree whose lib is older), JIT kernel compiles fail with errors like
#   `NUM_WORKER_CORES was not declared` and the live serve breaks.
#
# The built-lib commit is recorded in <tree>/.ttq-runtime-identity.json ("base_revision"),
# the same file run_vllm_api_server.py fail-closes on. This library reads/writes that file
# and compares it against `git rev-parse HEAD`.

set -o pipefail

# Canonical per-commit tree root (node-local NVMe $HOME, never /data which is shared NFS).
: "${TTM_CACHE_ROOT:=$HOME/.cache/tt-metal}"

# 40-hex full commit sha regex
_TTM_SHA_RE='^[0-9a-f]{40}$'

ttm_die() { echo "ttmetal: $*" >&2; exit 1; }

# Canonical immutable tree path for a given commit.
ttm_tree_for() {
  local commit="$1"
  [[ "$commit" =~ $_TTM_SHA_RE ]] || ttm_die "commit must be 40-hex, got '$commit'"
  printf '%s/%s\n' "$TTM_CACHE_ROOT" "$commit"
}

# git HEAD of a tree (full sha), empty if not a git tree.
ttm_head() {
  local tree="$1"
  git -C "$tree" rev-parse HEAD 2>/dev/null || true
}

# base_revision recorded in the tree's runtime-identity, empty if missing/invalid.
ttm_identity_rev() {
  local tree="$1" f="$1/.ttq-runtime-identity.json"
  [[ -f "$f" && ! -L "$f" ]] || return 0
  # tiny json field extractor; avoids a python dependency in the hot guard path
  sed -n 's/.*"base_revision"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' "$f" | head -1
}

# Does the tree actually carry a built lib?
ttm_has_build() {
  local tree="$1"
  [[ -f "$tree/build/lib/libtt_metal.so" || -f "$tree/build_Release/lib/libtt_metal.so" \
     || -f "$tree/build/lib/_ttnn.so" || -f "$tree/build_Release/lib/_ttnn.so" ]]
}

# Write/refresh the runtime-identity stamp for a freshly built tree.
# The stamp records the commit the lib was built at; it MUST be the current HEAD.
ttm_stamp() {
  local tree="$1" commit="$2"
  [[ "$commit" =~ $_TTM_SHA_RE ]] || ttm_die "stamp: commit must be 40-hex"
  local head; head="$(ttm_head "$tree")"
  [[ "$head" == "$commit" ]] || ttm_die "stamp refused: HEAD($head) != build commit($commit) in $tree"
  ttm_has_build "$tree" || ttm_die "stamp refused: no built lib in $tree"
  printf '{"base_revision": "%s", "patchset_sha256": null, "manifest_sha256": null}\n' \
    "$commit" > "$tree/.ttq-runtime-identity.json"
  echo "ttmetal: stamped $tree base_revision=$commit"
}

# The core verdict. Prints "<VERDICT> head=<sha> identity=<sha>" and returns a code.
#   COHERENT (0) : identity present, base_revision == HEAD, build present
#   CORRUPT  (3) : identity present but base_revision != HEAD  <-- the 2a8253ad-vs-1c2aff50 class
#   UNSTAMPED(4) : built tree with no identity (cannot attest built-lib commit)
#   NOBUILD  (5) : git tree, no built lib
#   NOTREE   (2) : not a git worktree
ttm_verdict() {
  local tree="$1"
  local head ident
  head="$(ttm_head "$tree")"
  if [[ -z "$head" ]]; then
    echo "NOTREE head= identity= tree=$tree"; return 2
  fi
  ident="$(ttm_identity_rev "$tree")"
  if [[ -n "$ident" ]]; then
    if [[ "$ident" != "$head" ]]; then
      echo "CORRUPT head=$head identity=$ident tree=$tree"; return 3
    fi
    if ! ttm_has_build "$tree"; then
      echo "NOBUILD head=$head identity=$ident tree=$tree"; return 5
    fi
    echo "COHERENT head=$head identity=$ident tree=$tree"; return 0
  fi
  if ttm_has_build "$tree"; then
    echo "UNSTAMPED head=$head identity= tree=$tree"; return 4
  fi
  echo "NOBUILD head=$head identity= tree=$tree"; return 5
}

# Pure comparison used by unit tests: compare a source-HEAD sha to a built-lib/identity sha.
# Same logic ttm_verdict applies, but on explicit values (no tree needed).
ttm_verdict_pair() {
  local head="$1" ident="$2"
  if [[ -z "$ident" ]]; then echo "UNSTAMPED head=$head identity="; return 4; fi
  if [[ "$head" != "$ident" ]]; then echo "CORRUPT head=$head identity=$ident"; return 3; fi
  echo "COHERENT head=$head identity=$ident"; return 0
}

# Enumerate live tt-metal serves owned by $USER on THIS node and the tree each depends on.
# Prints lines: "<pid> <TT_METAL_HOME>"  (realpath-resolved), one per serve process.
ttm_live_serve_trees() {
  local pid cmd tmh
  for pid in $(pgrep -u "$USER" -f 'run_vllm_api_server|server_example_tt|vllm.entrypoints' 2>/dev/null); do
    # skip shells / this guard itself
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    case "$cmd" in
      *python*|*/vllm*) : ;;
      *) continue ;;
    esac
    tmh="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TT_METAL_HOME=//p' | head -1)"
    [[ -z "$tmh" ]] && tmh="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TT_METAL_RUNTIME_ROOT=//p' | head -1)"
    [[ -z "$tmh" ]] && continue
    printf '%s %s\n' "$pid" "$(readlink -f "$tmh" 2>/dev/null || echo "$tmh")"
  done
}
