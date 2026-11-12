#!/bin/bash
# Build + push the TTS blaze media-server image (tt-media-server/Dockerfile.blaze)
# from a tt-inference-server commit and a nanicicTT/nanicic-TTS-llm-engine commit.
#
#   build_tts_image.sh resolve [--engine REF] [--server REF] [--tag TAG]
#       Fetches both repos, resolves REFs to full SHAs (defaults: latest
#       commit of nanicicTT/TTS_Scheduler and of ljovanovic/tts_stable_branch),
#       prints KEY=VALUE lines plus the commits since the previous recorded
#       build, and whether TAG already exists in the registry. Changes nothing.
#
#   build_tts_image.sh build --engine SHA --server SHA --tag TAG [--no-push]
#       Builds from a clean worktree of the server commit (never the live
#       working tree, so uncommitted edits cannot leak into the image), retries
#       transient clone failures, pushes, and verifies the registry digest.
#       Long-running: start it with nohup and poll the log it names.
#
# Needs GitHub access from this host (gh token in ~/.config/gh/hosts.yml, or
# GITHUB_TOKEN) and docker logged in to ghcr.io.
set -o pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$REPO_ROOT" || exit 1

IMAGE=ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze
ENGINE_SLUG=nanicicTT/nanicic-TTS-llm-engine
ENGINE_URL=https://github.com/${ENGINE_SLUG}.git
ENGINE_DEFAULT_REF=nanicicTT/TTS_Scheduler
SERVER_DEFAULT_REF=ljovanovic/tts_stable_branch
RECORD=tt-media-server/docs/tts_image_builds.md
WORK=${TTS_IMAGE_BUILD_DIR:-$(dirname "$REPO_ROOT")/tts-image-builds}
ENGINE_CACHE=${WORK}/engine.git

die() { echo "ERROR: $*" >&2; exit 1; }

token() {
  if [ -n "${GITHUB_TOKEN:-}" ]; then echo "$GITHUB_TOKEN"; return; fi
  awk '/oauth_token:/{print $2; exit}' ~/.config/gh/hosts.yml 2>/dev/null
}

# Never let the token reach the terminal or a log.
redact() { local t; t=$(token); if [ -n "$t" ]; then sed "s/${t}/***/g"; else cat; fi; }

# Run git against GitHub. The token comes from a credential helper that reads
# it from the environment, so it is never in a URL, on a command line (ps), or
# saved in any config. HOME is an empty dir so a global
# `url.git@github.com:.insteadOf=https://github.com/` (common on these hosts,
# and needing an ssh key the host may not have) cannot reroute to ssh; ssh
# submodule URLs (tt-metal) are fetched over https instead. -c settings and
# the environment carry over into submodule fetches.
gitnet() {
  local h rc; h=$(mktemp -d)
  HOME=$h GIT_TERMINAL_PROMPT=0 TTS_GH_TOKEN=$(token) timeout "${GITNET_TIMEOUT:-1800}" git \
    -c credential.helper= \
    -c 'credential.helper=!f() { test "$1" = get || exit 0; echo username=x-access-token; echo "password=$TTS_GH_TOKEN"; }; f' \
    -c url.https://github.com/.insteadOf=git@github.com: \
    -c url.https://github.com/.insteadOf=ssh://git@github.com/ \
    "$@"
  rc=$?; rm -rf "$h"; return $rc
}

# Fetch into the engine cache. Blobless (partial clone): history for the
# change log without the file contents.
efetch() {
  GITNET_TIMEOUT=600 gitnet -C "$ENGINE_CACHE" fetch -q --filter=blob:none origin "$@" 2>&1 | redact >&2
  return "${PIPESTATUS[0]}"
}

ENGINE_REF="" SERVER_REF="" TAG="" PUSH=1
MODE=$1; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE_REF=$2; shift 2 ;;
    --server) SERVER_REF=$2; shift 2 ;;
    --tag)    TAG=$2; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

is_sha() { [[ "$1" =~ ^[0-9a-f]{7,40}$ ]]; }

valid_tag() { [[ "$1" =~ ^[a-z0-9][a-z0-9_.-]{0,127}$ ]]; }

# Fail in seconds, not after git's two-minute connect timeout.
need_github() {
  timeout 15 bash -c '</dev/tcp/github.com/443' 2>/dev/null \
    || die "cannot reach github.com:443 from $(hostname); this host needs internet to build"
}

resolve_engine() {
  [ -n "$(token)" ] || die "no GitHub token (gh auth login on this host, or export GITHUB_TOKEN)"
  need_github
  mkdir -p "$WORK"
  if [ ! -d "$ENGINE_CACHE" ]; then
    git init -q --bare "$ENGINE_CACHE"
    git -C "$ENGINE_CACHE" remote add origin "$ENGINE_URL"
    # git 2.25 only accepts --filter on a fetch from a remote marked as a
    # partial-clone promisor.
    git -C "$ENGINE_CACHE" config core.repositoryformatversion 1
    git -C "$ENGINE_CACHE" config extensions.partialClone origin
  fi
  # Caches made by older versions of this script use a placeholder URL.
  git -C "$ENGINE_CACHE" remote set-url origin "$ENGINE_URL"
  local ref=${ENGINE_REF:-$ENGINE_DEFAULT_REF}
  # Blobless: history for the change log, without downloading file contents.
  if is_sha "$ref"; then
    efetch "+refs/heads/${ENGINE_DEFAULT_REF}:refs/heads/${ENGINE_DEFAULT_REF}" \
      || die "cannot fetch engine branch ${ENGINE_DEFAULT_REF} from ${ENGINE_SLUG} (no network?)"
    if ! git -C "$ENGINE_CACHE" cat-file -e "${ref}^{commit}" 2>/dev/null; then
      [ ${#ref} -eq 40 ] || die "engine commit $ref is not on ${ENGINE_DEFAULT_REF}; give the full 40-char SHA"
      efetch "$ref" || die "engine commit $ref not found in ${ENGINE_SLUG}"
    fi
    git -C "$ENGINE_CACHE" rev-parse "${ref}^{commit}"
  else
    efetch "+refs/heads/${ref}:refs/heads/${ref}" \
      || die "cannot fetch engine branch ${ref} from ${ENGINE_SLUG} (no network?)"
    git -C "$ENGINE_CACHE" rev-parse "refs/heads/${ref}"
  fi
}

resolve_server() {
  local ref=${SERVER_REF:-$SERVER_DEFAULT_REF}
  if is_sha "$ref"; then
    git rev-parse --verify -q "${ref}^{commit}" || die "server commit $ref not found locally"
    return
  fi
  timeout 120 git fetch -q origin "$ref" 2>/dev/null || echo "WARN: could not fetch origin/${ref}; using local state" >&2
  local local_tip remote_tip
  local_tip=$(git rev-parse --verify -q "refs/heads/${ref}")
  remote_tip=$(git rev-parse --verify -q "refs/remotes/origin/${ref}")
  [ -n "$local_tip$remote_tip" ] || die "server branch ${ref} not found locally or on origin"
  if [ -z "$remote_tip" ]; then echo "$local_tip"; return; fi
  if [ -z "$local_tip" ]; then echo "$remote_tip"; return; fi
  # Latest = whichever tip contains the other; diverged branches need a human.
  if git merge-base --is-ancestor "$remote_tip" "$local_tip"; then echo "$local_tip"
  elif git merge-base --is-ancestor "$local_tip" "$remote_tip"; then echo "$remote_tip"
  else die "local ${ref} and origin/${ref} have diverged; pass --server <sha>"
  fi
}

# Last recorded build: "| date | tag | engine | server | digest | notes |"
prev_build_field() {
  [ -f "$RECORD" ] || return
  grep -E '^\| [0-9]{4}-' "$RECORD" | tail -1 | awk -F'|' -v n="$1" '{gsub(/[ `]/,"",$n); print $n}'
}

# yes / no / unknown. "unknown" (registry unreachable, not logged in) must not
# be read as "free", or a picked tag could silently overwrite an image.
tag_state() {
  local err
  if err=$(timeout 60 docker manifest inspect "${IMAGE}:$1" 2>&1 >/dev/null); then
    echo yes
  elif echo "$err" | grep -qiE "no such manifest|manifest unknown|not found"; then
    echo no
  else
    echo unknown
  fi
}

cmd_resolve() {
  local engine server prev_engine prev_server
  engine=$(resolve_engine) || exit 1
  server=$(resolve_server) || exit 1
  prev_engine=$(prev_build_field 4)
  prev_server=$(prev_build_field 5)
  echo "ENGINE=$engine"
  echo "ENGINE_REF=${ENGINE_REF:-$ENGINE_DEFAULT_REF}"
  echo "ENGINE_SUBJECT=$(git -C "$ENGINE_CACHE" log -1 --format='%h %cs %s' "$engine")"
  echo "SERVER=$server"
  echo "SERVER_REF=${SERVER_REF:-$SERVER_DEFAULT_REF}"
  echo "SERVER_SUBJECT=$(git log -1 --format='%h %cs %s' "$server")"
  echo "PREV_TAG=$(prev_build_field 3)"
  echo "PREV_ENGINE=$prev_engine"
  echo "PREV_SERVER=$prev_server"
  if [ -n "$TAG" ]; then
    valid_tag "$TAG" || die "invalid tag '$TAG' (lowercase letters, digits, _ . -)"
    echo "TAG_EXISTS=$(tag_state "$TAG")"
  fi
  local dirty
  dirty=$(git status --porcelain --untracked-files=no -- tt-media-server | wc -l)
  echo "UNCOMMITTED_SERVER_FILES=$dirty"
  echo
  echo "=== engine commits since previous build (${prev_engine:-none recorded}) ==="
  if [ -n "$prev_engine" ] && ! git -C "$ENGINE_CACHE" cat-file -e "${prev_engine}^{commit}" 2>/dev/null; then
    # The previous engine may live on another branch; fetch it to compare.
    efetch "$prev_engine" >/dev/null 2>&1 || true
  fi
  if [ -n "$prev_engine" ] && git -C "$ENGINE_CACHE" cat-file -e "${prev_engine}^{commit}" 2>/dev/null; then
    echo "new:"
    git -C "$ENGINE_CACHE" log --format='  %h %cs %s' "${prev_engine}..${engine}" | head -30
    # Non-empty when the previous engine was on a different branch: these
    # commits were in the last image and will NOT be in this one.
    local dropped
    dropped=$(git -C "$ENGINE_CACHE" log --format='  %h %cs %s' "${engine}..${prev_engine}" | head -30)
    if [ -n "$dropped" ]; then
      echo "DROPPED (in the previous image, not in this one):"
      echo "$dropped"
    fi
  else
    echo "(previous engine commit not available; latest 10)"
    git -C "$ENGINE_CACHE" log --format='  %h %cs %s' -10 "$engine"
  fi
  echo
  echo "=== server commits since previous build (${prev_server:-none recorded}) ==="
  if [ -n "$prev_server" ] && git cat-file -e "${prev_server}^{commit}" 2>/dev/null; then
    git log --format='%h %cs %s' "${prev_server}..${server}" -- . ':!tt-media-server/docs/tts_image_builds.md' | head -30
  else
    git log --format='%h %cs %s' -10 "$server"
  fi
}

cmd_build() {
  [ -n "$ENGINE_REF" ] && [ -n "$SERVER_REF" ] && [ -n "$TAG" ] || die "build needs --engine, --server and --tag"
  [ ${#ENGINE_REF} -eq 40 ] || die "build needs the full engine SHA (run resolve first)"
  valid_tag "$TAG" || die "invalid tag '$TAG'"
  local server; server=$(git rev-parse --verify -q "${SERVER_REF}^{commit}") || die "server commit not found"
  [ -n "$(token)" ] || die "no GitHub token"
  need_github
  mkdir -p "$WORK"
  local log="${WORK}/${TAG}.log" src="${WORK}/src-${TAG}"
  local full="${IMAGE}:${TAG}" short; short=$(git rev-parse --short=9 "$server")
  : > "$log"
  {
    echo "=== build started $(date -Is) ==="
    echo "=== engine: ${ENGINE_REF} (${ENGINE_SLUG}) ==="
    echo "=== server: ${server} ==="
    echo "=== tag:    ${full} ==="
  } | tee -a "$log"

  # Clean source tree at exactly the server commit. Dockerfile.blaze COPYs the
  # working tree, so building from the live checkout would ship whatever is
  # uncommitted there.
  git worktree remove --force "$src" 2>/dev/null; rm -rf "$src"
  git worktree add -q --detach "$src" "$server" 2>&1 | tee -a "$log" || die "worktree add failed"
  local eng="${WORK}/engine-${TAG}"
  trap 'git -C "$REPO_ROOT" worktree remove --force "$src" 2>/dev/null; rm -rf "$eng"' EXIT

  # Engine checkout with submodules, made here on the host and handed to
  # docker as the `tt-llm-engine` build context. The token is used only by
  # these host-side fetches; docker build never sees it.
  local erc=1 eattempt
  for eattempt in 1 2 3; do
    echo "=== engine checkout attempt ${eattempt} $(date -Is) ===" | tee -a "$log"
    rm -rf "$eng"
    { git init -q "$eng" \
      && git -C "$eng" remote add origin "$ENGINE_URL" \
      && gitnet -C "$eng" fetch -q --depth 1 origin "$ENGINE_REF" \
      && git -C "$eng" -c advice.detachedHead=false checkout -q FETCH_HEAD \
      && gitnet -C "$eng" submodule update -q --init --recursive --depth 1 --jobs 4; } 2>&1 | redact | tee -a "$log"
    erc=${PIPESTATUS[0]}
    [ "$erc" -eq 0 ] && break
    sleep 60
  done
  [ "$erc" -eq 0 ] || die "engine checkout of ${ENGINE_REF} failed (see $log)"
  [ "$(git -C "$eng" rev-parse HEAD)" = "$ENGINE_REF" ] || die "engine checkout is not at ${ENGINE_REF}"
  if grep -rIlE 'x-access-token|gh[opsu]_[A-Za-z0-9]{30,}|github_pat_' "$eng/.git" 2>/dev/null | head -1 | grep -q .; then
    die "a GitHub credential ended up in the engine checkout's git config; refusing to build"
  fi

  local rc=1 attempt
  for attempt in 1 2 3; do
    echo "=== docker build attempt ${attempt} $(date -Is) ===" | tee -a "$log"
    (cd "$src" && DOCKER_BUILDKIT=1 docker build --progress=plain \
        --build-context tt-llm-engine="$eng" \
        --build-arg TT_INFERENCE_SERVER_COMMIT_SHA_OR_TAG="$short" \
        --build-arg TT_LLM_ENGINE_COMMIT="$ENGINE_REF" \
        -f tt-media-server/Dockerfile.blaze -t "$full" .) 2>&1 | redact | tee -a "$log" >/dev/null
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break
    # Only network-shaped failures are worth retrying; a compile error is not.
    if ! tail -400 "$log" | grep -qE "Failed to clone|Could not resolve host|Connection (reset|timed out)|early EOF|RPC failed|TLS|502|503|504"; then
      break
    fi
    echo "=== transient failure, retrying in 5 min ===" | tee -a "$log"; sleep 300
  done
  if [ "$rc" -ne 0 ]; then
    { echo "=== BUILD FAILED $(date -Is) exit=${rc} ==="
      grep -E 'fatal error|CMake Error|^ERROR:|error: ' "$log" | tail -20
      echo "RESULT: BUILD_FAILED"; } | tee -a "$log"
    exit 1
  fi

  local local_id remote_id
  local_id=$(docker inspect --format '{{.Id}}' "$full")
  echo "=== BUILD OK, image ${local_id} ===" | tee -a "$log"
  if [ "$PUSH" -eq 0 ]; then
    echo "RESULT: BUILT_NOT_PUSHED tag=${full} id=${local_id} engine=${ENGINE_REF} server=${server}" | tee -a "$log"
    exit 0
  fi
  timeout 1800 docker push "$full" 2>&1 | tee -a "$log" >/dev/null
  local prc=${PIPESTATUS[0]}
  remote_id=$(docker manifest inspect "$full" 2>/dev/null \
    | python3 -c "import json,sys;print(json.load(sys.stdin).get('config',{}).get('digest',''))" 2>/dev/null)
  if [ "$prc" -eq 0 ] && [ "$local_id" = "$remote_id" ]; then
    echo "RESULT: SUCCESS tag=${full} digest=${local_id} engine=${ENGINE_REF} server=${server}" | tee -a "$log"
  else
    echo "RESULT: PUSH_FAILED exit=${prc} local=${local_id} remote=${remote_id:-none}" | tee -a "$log"
    exit 1
  fi
}

case "$MODE" in
  resolve) cmd_resolve ;;
  build)   cmd_build ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
