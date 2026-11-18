#!/usr/bin/env bash
# Builds the router for GLM-5.3-Flash on Galaxy: vllm-router 0.1.17 (vllm-project/router commit 3451e66) with this
# directory's patches, into PREFIX/bin, user-level (rustup under PREFIX/rust, no system change). The production
# binaries were built with rustc/cargo 1.99.1 and `cargo build --release --locked`.
#   [PATCHES=placeholders|route3] build-router.sh [PREFIX=$HOME/glm-router]
# PATCHES=placeholders (the default): router-media-placeholders.patch only. This is the router that carries production
#   traffic today; its binary's sha256 there is 2a98718e...
# PATCHES=route3: router-media-placeholders.patch, router-reuseport.patch, router-retry-exclude.patch, in that order.
#   Built and tested but not serving production traffic yet; its binary's sha256 is 68ad05bd...
# Every patch is switched by its own environment variable and is off when it is unset (see glm-router.env.example).
# The binary's sha256 depends on the build path and CARGO_HOME (cargo's metadata hashes them), so a build elsewhere
# gives the same code under a different sha256 than the ones above.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFIX=${1:-$HOME/glm-router}
TOOLCHAIN=${TOOLCHAIN:-1.99.1}
case ${PATCHES:-placeholders} in
    placeholders) PATCH_FILES=(router-media-placeholders.patch) ;;
    route3) PATCH_FILES=(router-media-placeholders.patch router-reuseport.patch router-retry-exclude.patch) ;;
    *) echo "PATCHES must be placeholders or route3, not '${PATCHES}'" >&2; exit 2 ;;
esac
export RUSTUP_HOME=$PREFIX/rust/rustup CARGO_HOME=$PREFIX/rust/cargo
mkdir -p "$PREFIX/bin" "$PREFIX/src"
if [[ ! -x $CARGO_HOME/bin/cargo ]]; then
    curl -sSf https://sh.rustup.rs -o "$PREFIX/rust-init.sh"
    sh "$PREFIX/rust-init.sh" -y --no-modify-path --profile minimal --default-toolchain "$TOOLCHAIN"
fi
if [[ ! -d $PREFIX/src/router ]]; then
    git clone https://github.com/vllm-project/router "$PREFIX/src/router"
fi
cd "$PREFIX/src/router"
git checkout --detach 3451e66
[[ -z $(git status --porcelain) ]] || { echo "the router source has local changes: refusing" >&2; exit 1; }
for p in "${PATCH_FILES[@]}"; do
    git apply --check "$HERE/$p" && git apply "$HERE/$p" && echo "applied $p"
done
"$CARGO_HOME/bin/cargo" +"$TOOLCHAIN" build --release --locked --bin vllm-router
install -m 755 target/release/vllm-router "$PREFIX/bin/vllm-router"
git checkout -- .
sha256sum "$PREFIX/bin/vllm-router"
