#!/bin/bash
# ttmetal_build.sh - build entrypoint wrapper that gives every (model, tt-metal commit) its
# OWN immutable, node-local tree, so one model's build/checkout can NEVER mutate the source
# a live serve JIT-compiles from.
#
# Usage:
#   ttmetal_build.sh <commit40hex> [--reference <existing-tt-metal>] [--jobs N] [--dry-run]
#   ttmetal_build.sh --from-tree <tree>   # resolve the required commit from a tree's identity
#   ttmetal_build.sh --from-spec <spec.json|model_id>   # resolve via resolve_commit.py
#
# Canonical tree:  $TTM_CACHE_ROOT/<commit>   (default $HOME/.cache/tt-metal/<commit>)
# The wrapper:
#   * resolves the per-commit tree path (never the shared $HOME/tt-metal);
#   * runs the guard: it refuses if the resolved tree is a live serve's TT_METAL_HOME and the
#     commit differs (belt-and-suspenders; the per-commit path makes this collision impossible);
#   * reuses the tree as-is if it is already COHERENT at <commit> (immutable, no re-checkout);
#   * otherwise clones (optionally from a local --reference for speed), checks out <commit>,
#     builds, creates the venv, and stamps .ttq-runtime-identity.json = <commit>.
#   * prints the exact env a serve should use (TT_METAL_HOME / TT_METAL_RUNTIME_ROOT).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ttmetal_lib.sh
source "$HERE/ttmetal_lib.sh"
GUARD="$HERE/ttmetal_guard.sh"

REF=""; JOBS=""; DRY=0; COMMIT=""; REMOTE="https://github.com/tenstorrent/tt-metal.git"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference) REF="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --remote) REMOTE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --from-tree) COMMIT="$(ttm_identity_rev "$2")"; [[ -z "$COMMIT" ]] && COMMIT="$(ttm_head "$2")"; shift 2;;
    --from-spec) COMMIT="$(python3 "$HERE/resolve_commit.py" "$2")"; shift 2;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 64;;
    *) COMMIT="$1"; shift;;
  esac
done

[[ "$COMMIT" =~ $_TTM_SHA_RE ]] || ttm_die "need a resolved 40-hex tt-metal commit (got '$COMMIT')"
TREE="$(ttm_tree_for "$COMMIT")"
echo "ttmetal_build: commit=$COMMIT"
echo "ttmetal_build: tree=$TREE"

# Guard: never build into a tree a live serve depends on unless it already provides <commit>.
if ! "$GUARD" guard-checkout "$TREE" "$COMMIT"; then
  ttm_die "guard refused the build target; see message above"
fi

# Immutable reuse: already-coherent tree is a no-op.
if v="$("$GUARD" check "$TREE")"; then
  echo "ttmetal_build: reuse (already coherent) -> $v"
  echo "TT_METAL_HOME=$TREE"; echo "TT_METAL_RUNTIME_ROOT=$TREE"
  exit 0
fi

if [[ "$DRY" == 1 ]]; then
  echo "ttmetal_build: DRY-RUN would clone/checkout/build/stamp $COMMIT into $TREE"
  exit 0
fi

# Node-local scratch for build tmp/cache (never /data).
mkdir -p "/tmp/$USER/ttm-tmp" "/tmp/$USER/ttm-cache" "$TTM_CACHE_ROOT"
export TMPDIR="/tmp/$USER/ttm-tmp" XDG_CACHE_HOME="/tmp/$USER/ttm-cache" PYTHONNOUSERSITE=1

if [[ ! -d "$TREE/.git" ]]; then
  echo "ttmetal_build: clone -> $TREE"
  if [[ -n "$REF" && -d "$REF/.git" ]]; then
    git clone --reference "$REF" "$REMOTE" "$TREE" || ttm_die "clone (ref) failed"
  else
    git clone "$REMOTE" "$TREE" || ttm_die "clone failed"
  fi
fi
cd "$TREE" || ttm_die "cd $TREE"
git cat-file -t "$COMMIT" >/dev/null 2>&1 || git fetch origin "$COMMIT" 2>&1 | tail -2
git checkout -f "$COMMIT" 2>&1 | tail -3 || ttm_die "checkout failed"
[[ "$(ttm_head "$TREE")" == "$COMMIT" ]] || ttm_die "HEAD != $COMMIT after checkout"
git -c submodule.fetchJobs=8 submodule update --init --recursive 2>&1 | tail -6 || ttm_die "submodule update failed"

export TT_METAL_HOME="$TREE"
echo "ttmetal_build: build_metal.sh"
./build_metal.sh --build-type Release --enable-ccache ${JOBS:+--jobs "$JOBS"} 2>&1 | tail -30
rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || ttm_die "build_metal failed rc=$rc"
echo "ttmetal_build: create_venv.sh"
./create_venv.sh 2>&1 | tail -15
rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || ttm_die "create_venv failed rc=$rc"

# Stamp: records the built-lib commit so the guard and the serve preflight can attest it.
ttm_stamp "$TREE" "$COMMIT"
"$GUARD" check "$TREE" || ttm_die "post-build verdict not COHERENT"
echo "ttmetal_build: DONE"
echo "TT_METAL_HOME=$TREE"; echo "TT_METAL_RUNTIME_ROOT=$TREE"
