#!/bin/bash
# ttmetal_guard.sh - refuse to corrupt a live serve's tt-metal tree, and detect the
# source-HEAD vs built-lib mismatch class BEFORE it breaks a serve.
#
# Subcommands:
#   check <tree>                 Print the coherence verdict for a tree. Exit nonzero if
#                                not COHERENT (3=CORRUPT, 4=UNSTAMPED, 5=NOBUILD, 2=NOTREE).
#   check-pair <headsha> <idsha> Pure verdict on explicit source-HEAD vs built-lib shas
#                                (used to prove the exact mismatch class in tests/CI).
#   live-serves                  List live serves on this node and the tree each depends on.
#   guard-checkout <tree> <commit>
#                                Decide whether it is safe to `git checkout <commit>` and
#                                build inside <tree>. Prints ALLOW/REFUSE and exits:
#                                  0  ALLOW
#                                  10 REFUSE: a live serve depends on this exact tree and
#                                     <commit> is not what the tree coherently provides
#                                     (a checkout/build here would break that serve)
#                                  11 REFUSE: tree is an immutable per-commit tree already
#                                     coherent at a DIFFERENT commit -- relocate to the
#                                     per-commit path for <commit> instead
#   stamp <tree> <commit>        Write .ttq-runtime-identity.json for a freshly built tree.
#
# Run this ON the node where the build/checkout would happen (live serves are node-local).

set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ttmetal_lib.sh
source "$HERE/ttmetal_lib.sh"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 64; }

cmd="${1:-}"; shift || true
case "$cmd" in
  check)
    tree="${1:?usage: check <tree>}"
    ttm_verdict "$tree"; exit $?
    ;;

  check-pair)
    head="${1:?usage: check-pair <headsha> <identitysha>}"; ident="${2:-}"
    ttm_verdict_pair "$head" "$ident"; exit $?
    ;;

  live-serves)
    got=0
    while read -r pid tree; do
      [[ -z "$pid" ]] && continue
      got=1
      v="$(ttm_verdict "$tree")"
      printf 'pid=%s tree=%s  ::  %s\n' "$pid" "$tree" "$v"
    done < <(ttm_live_serve_trees)
    [[ "$got" == 0 ]] && echo "(no live tt-metal serves on $(hostname))"
    exit 0
    ;;

  guard-checkout)
    tree="${1:?usage: guard-checkout <tree> <commit>}"
    commit="${2:?usage: guard-checkout <tree> <commit>}"
    [[ "$commit" =~ $_TTM_SHA_RE ]] || ttm_die "commit must be 40-hex"
    rtree="$(readlink -f "$tree" 2>/dev/null || echo "$tree")"

    # 1) Is a live serve on this node bound to this exact tree?
    while read -r pid ltree; do
      [[ "$ltree" == "$rtree" ]] || continue
      # A serve is live here. Safe ONLY if the tree already coherently provides <commit>.
      head="$(ttm_head "$rtree")"; ident="$(ttm_identity_rev "$rtree")"
      if [[ "$head" == "$commit" && "$ident" == "$commit" ]]; then
        echo "ALLOW: $rtree already coherent at $commit; live serve pid=$pid unaffected"
        exit 0
      fi
      echo "REFUSE: live serve pid=$pid depends on $rtree (head=$head identity=$ident);" \
           "checking out/building $commit here would break its JIT. Build into" \
           "$(ttm_tree_for "$commit") instead." >&2
      exit 10
    done < <(ttm_live_serve_trees)

    # 2) No live serve, but is this an immutable per-commit tree pinned elsewhere?
    v="$(ttm_verdict "$rtree")"; rc=$?
    head="$(ttm_head "$rtree")"; ident="$(ttm_identity_rev "$rtree")"
    if [[ $rc -eq 0 && "$ident" != "$commit" ]]; then
      echo "REFUSE: $rtree is an immutable tree coherent at $ident, not $commit." \
           "Build into $(ttm_tree_for "$commit") instead." >&2
      exit 11
    fi
    echo "ALLOW: no live serve depends on $rtree; safe to checkout/build $commit ($v)"
    exit 0
    ;;

  stamp)
    tree="${1:?usage: stamp <tree> <commit>}"; commit="${2:?usage: stamp <tree> <commit>}"
    ttm_stamp "$tree" "$commit"
    ;;

  ""|-h|--help) usage ;;
  *) echo "unknown subcommand: $cmd" >&2; usage ;;
esac
