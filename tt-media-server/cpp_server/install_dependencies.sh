#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# cpp_server build dependencies.
#
# Default behavior: install everything the build needs into a user-local prefix
# (${HOME}/.local) without requiring root / sudo / apt. Source trees are kept
# under cpp_server/deps/sources/. An env.sh file is written next to them so
# build.sh can pick the prefix up automatically.
#
# If apt is usable (root + writable lists), system packages are used instead
# (the Dockerfile flow); Drogon is still built from source because the apt
# package is not available on every supported distro.
#
# Usage:
#   ./install_dependencies.sh                 # install to ${HOME}/.local
#   ./install_dependencies.sh --prefix /opt/foo
#   ./install_dependencies.sh --kafka         # also build librdkafka
#   ./install_dependencies.sh --force-source  # skip apt even if available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/deps"
SOURCES_DIR="${DEPS_DIR}/sources"
ENV_SH="${DEPS_DIR}/env.sh"

# Pick a directory on a filesystem that has at least ${2} MiB free. If the
# preferred location (${1}) is too tight, fall back to ${3} (which lives under
# cpp_server/deps/ on the same filesystem as the source tree).
pick_roomy_dir() {
    local pref="$1" min_mb="$2" fallback="$3"
    local probe avail
    probe="${pref}"
    [ -d "${probe}" ] || probe="$(dirname "${probe}")"
    [ -d "${probe}" ] || probe="${HOME}"
    avail=$(df -Pm "${probe}" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "${avail}" ] && [ "${avail}" -ge "${min_mb}" ]; then
        echo "${pref}"
    else
        echo "${fallback}"
    fi
}

# Defaults: home if it has room, else the workspace's deps/ (usually on /data).
# ~/.local needs ~1.5 GiB (JsonCpp + libuuid + Drogon). Rustup + cargo can
# balloon to several GiB.
PREFIX="${PREFIX:-$(pick_roomy_dir "${HOME}/.local"  2048 "${DEPS_DIR}/opt")}"
CARGO_HOME_PICK="${CARGO_HOME:-$(pick_roomy_dir "${HOME}/.cargo"  4096 "${DEPS_DIR}/cargo")}"
RUSTUP_HOME_PICK="${RUSTUP_HOME:-$(pick_roomy_dir "${HOME}/.rustup" 4096 "${DEPS_DIR}/rustup")}"

INSTALL_KAFKA=0
FORCE_SOURCE=0

usage() {
    cat <<EOF
Usage: $0 [--prefix DIR] [--kafka] [--force-source]

Installs cpp_server build dependencies (Drogon, JsonCpp, libuuid, Rust).

Options:
  --prefix DIR     Install prefix for source-built deps (default: \$HOME/.local)
  --kafka          Also build librdkafka (enables --kafka in build.sh)
  --force-source   Build from source even if apt is usable

If apt is available (root + writable lists), system packages are used for most
libs and Drogon is still source-built. Otherwise everything missing is built
from source into the prefix. Emits cpp_server/deps/env.sh so build.sh picks it
up automatically on the next run.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)          PREFIX="$2"; shift 2 ;;
        --prefix=*)        PREFIX="${1#*=}"; shift ;;
        --kafka)           INSTALL_KAFKA=1; shift ;;
        --force-source)    FORCE_SOURCE=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

mkdir -p "${SOURCES_DIR}" "${PREFIX}" "${CARGO_HOME_PICK}" "${RUSTUP_HOME_PICK}"

NPROC=$(nproc 2>/dev/null || echo 4)
log()  { echo "[install_deps] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

log "Install prefix:   ${PREFIX}"
log "CARGO_HOME:       ${CARGO_HOME_PICK}"
log "RUSTUP_HOME:      ${RUSTUP_HOME_PICK}"

# Make the prefix's cmake/pkgconfig/linker paths visible to the rest of this
# script (so Drogon can find our freshly-built JsonCpp/libuuid, etc.).
export CMAKE_PREFIX_PATH="${PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${PREFIX}/bin:${PATH}"

# Cargo/rustup state relocated off the full root filesystem when $HOME is tight.
export CARGO_HOME="${CARGO_HOME_PICK}"
export RUSTUP_HOME="${RUSTUP_HOME_PICK}"
export PATH="${CARGO_HOME}/bin:${PATH}"

# ---------------------------------------------------------------------------
# Optional apt path (keeps Docker-style installs working). Falls through to
# source builds if we're a regular user with no passwordless sudo.
# ---------------------------------------------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo && sudo -n true >/dev/null 2>&1; then
    SUDO="sudo -n"
fi

can_use_apt() {
    have apt-get || return 1
    { [ "$(id -u)" -eq 0 ] || [ -n "${SUDO}" ]; } || return 1
    $SUDO apt-get check >/dev/null 2>&1
}

APT_OK=0
if [ "${FORCE_SOURCE}" != 1 ] && can_use_apt; then
    APT_PKGS=(
        build-essential cmake g++ pkg-config curl git wget
        libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libboost-all-dev
    )
    [ "${INSTALL_KAFKA}" = 1 ] && APT_PKGS+=(librdkafka-dev)

    log "apt available — installing: ${APT_PKGS[*]}"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

    # clang-format-20 for the linter; optional, don't fail if llvm.sh is down
    if ! have clang-format-20; then
        LLVM_SH="/tmp/llvm.sh"
        if curl -fsSL -o "${LLVM_SH}" https://apt.llvm.org/llvm.sh; then
            chmod +x "${LLVM_SH}"
            $SUDO "${LLVM_SH}" 20 || true
            rm -f "${LLVM_SH}"
            $SUDO apt-get install -y --no-install-recommends clang-format-20 || true
        fi
    fi
    $SUDO rm -rf /var/lib/apt/lists/*
    APT_OK=1
else
    log "apt unavailable — building missing libs from source into ${PREFIX}"
fi

# ---------------------------------------------------------------------------
# Rust (cargo) — needed by mlc-ai/tokenizers-cpp fetched at configure time.
# tokenizers-cpp pulls `monostate` which requires rustc >= 1.79, so the stock
# Ubuntu rust (1.75) isn't enough. If we find an old rustc, shadow it with a
# rustup toolchain under $HOME/.cargo.
# ---------------------------------------------------------------------------
RUST_MIN_MAJOR=1
RUST_MIN_MINOR=79

rust_version_ok() {
    have rustc || return 1
    local v major minor
    v="$(rustc --version 2>/dev/null | awk '{print $2}')"
    major="${v%%.*}"; minor="${v#*.}"; minor="${minor%%.*}"
    [ -n "${major}" ] && [ -n "${minor}" ] || return 1
    if [ "${major}" -gt "${RUST_MIN_MAJOR}" ]; then return 0; fi
    [ "${major}" -eq "${RUST_MIN_MAJOR}" ] && [ "${minor}" -ge "${RUST_MIN_MINOR}" ]
}

install_rustup() {
    log "Installing Rust via rustup (stable) into:"
    log "  CARGO_HOME=${CARGO_HOME}"
    log "  RUSTUP_HOME=${RUSTUP_HOME}"
    # RUSTUP_INIT_SKIP_PATH_CHECK=yes lets rustup install alongside a system
    # rust (e.g. Ubuntu's rustc 1.75) without refusing with
    # "cannot install while Rust is installed". We shadow the system one by
    # putting ${CARGO_HOME}/bin first on PATH.
    RUSTUP_INIT_SKIP_PATH_CHECK=yes \
    CARGO_HOME="${CARGO_HOME}" \
    RUSTUP_HOME="${RUSTUP_HOME}" \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | RUSTUP_INIT_SKIP_PATH_CHECK=yes \
          CARGO_HOME="${CARGO_HOME}" \
          RUSTUP_HOME="${RUSTUP_HOME}" \
          sh -s -- -y --no-modify-path --default-toolchain stable
    [ -f "${CARGO_HOME}/env" ] && . "${CARGO_HOME}/env"
}

# Prefer an existing rustup install if one is already on disk under the
# chosen CARGO_HOME.
[ -f "${CARGO_HOME}/env" ] && . "${CARGO_HOME}/env"

if ! rust_version_ok; then
    if have rustc; then
        log "Found rustc $(rustc --version 2>/dev/null | awk '{print $2}') but need >= ${RUST_MIN_MAJOR}.${RUST_MIN_MINOR}."
    fi
    install_rustup
    export PATH="${CARGO_HOME}/bin:${PATH}"
    if ! rust_version_ok; then
        log "ERROR: rustup install did not produce rustc >= ${RUST_MIN_MAJOR}.${RUST_MIN_MINOR}"
        log "  got: $(rustc --version 2>/dev/null || echo '<missing>')"
        exit 1
    fi
fi
log "Using rustc $(rustc --version | awk '{print $2}') from $(command -v rustc)"

# Pre-install the rustup components tokenizers-cpp / huggingface/tokenizers
# can ask for (e.g. via rust-toolchain.toml). Doing this once, serially, up
# front avoids the known race where several parallel `cargo build` workers all
# race to rustup-auto-install the same component and stomp each other's
# download.partial files:
#   error: component download failed for rust-src: could not rename ...
if have rustup; then
    for _c in rust-src rustfmt clippy; do
        rustup component add "${_c}" >/dev/null 2>&1 || \
            log "  note: rustup component add ${_c} failed (continuing)"
    done
fi

# ---------------------------------------------------------------------------
# Source-build helpers
# ---------------------------------------------------------------------------
fetch_git() {
    # fetch_git <url> <tag> <dir> [marker-path-inside-dir]
    local url="$1" tag="$2" dir="$3" marker="${4:-.git}"
    local out="${SOURCES_DIR}/${dir}"
    if [ -d "${out}" ] && [ ! -e "${out}/${marker}" ]; then
        log "  ${dir}/ exists but looks incomplete — removing and re-cloning"
        rm -rf "${out}"
    fi
    if [ ! -d "${out}" ]; then
        git clone --depth 1 --branch "${tag}" --recurse-submodules "${url}" "${out}"
    fi
}

fetch_tarball() {
    # fetch_tarball <url> <dir> <topdir-inside-tarball> [marker-path-inside-dir]
    local url="$1" dir="$2" top="$3" marker="${4:-configure}" tarball
    local out="${SOURCES_DIR}/${dir}"
    tarball="${SOURCES_DIR}/${dir}.tar"

    # Recover from a partial/stale previous run (e.g. empty ${dir}, leftover
    # tarball, or an unrenamed topdir).
    if [ -d "${out}" ] && [ ! -e "${out}/${marker}" ]; then
        log "  ${dir}/ exists but looks incomplete — removing and re-extracting"
        rm -rf "${out}"
    fi
    if [ ! -d "${out}" ] && [ -d "${SOURCES_DIR}/${top}" ]; then
        mv "${SOURCES_DIR}/${top}" "${out}"
    fi

    if [ ! -d "${out}" ]; then
        [ -f "${tarball}" ] || curl -fsSL "${url}" -o "${tarball}"
        tar -xf "${tarball}" -C "${SOURCES_DIR}"
        [ -d "${SOURCES_DIR}/${top}" ] && mv "${SOURCES_DIR}/${top}" "${out}"
        rm -f "${tarball}"
    fi
}

# libuuid (util-linux). Headers + libuuid.so. Release tarballs ship with
# pre-generated configure so libtoolize is not required.
need_libuuid() {
    [ -f "${PREFIX}/include/uuid/uuid.h" ] && return 1
    [ -f /usr/include/uuid/uuid.h ]        && return 1
    return 0
}
build_libuuid() {
    need_libuuid || { log "libuuid: already available, skipping"; return 0; }
    log "Building libuuid (util-linux 2.39.3) from source..."
    local ver="2.39.3"
    fetch_tarball \
        "https://mirrors.edge.kernel.org/pub/linux/utils/util-linux/v2.39/util-linux-${ver}.tar.xz" \
        "util-linux" "util-linux-${ver}"
    cd "${SOURCES_DIR}/util-linux"
    ./configure --prefix="${PREFIX}" \
        --disable-all-programs --enable-libuuid \
        --without-systemd --without-python --without-ncurses --without-ncursesw \
        --disable-makeinstall-chown --disable-makeinstall-setuid \
        --disable-nls >/dev/null
    make -j"${NPROC}"
    make install
    cd "${SCRIPT_DIR}"
}

# JsonCpp — CMakeLists.txt uses `find_package(jsoncpp REQUIRED)`.
need_jsoncpp() {
    [ -f "${PREFIX}/lib/cmake/jsoncpp/jsoncppConfig.cmake" ] && return 1
    [ -f "${PREFIX}/lib64/cmake/jsoncpp/jsoncppConfig.cmake" ] && return 1
    pkg-config --exists jsoncpp 2>/dev/null && return 1
    return 0
}
build_jsoncpp() {
    need_jsoncpp || { log "JsonCpp: already available, skipping"; return 0; }
    log "Building JsonCpp 1.9.5 from source..."
    fetch_git https://github.com/open-source-parsers/jsoncpp.git 1.9.5 jsoncpp
    cmake -S "${SOURCES_DIR}/jsoncpp" -B "${SOURCES_DIR}/jsoncpp/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
        -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
        -DJSONCPP_WITH_EXAMPLE=OFF
    cmake --build "${SOURCES_DIR}/jsoncpp/build" -j"${NPROC}"
    cmake --install "${SOURCES_DIR}/jsoncpp/build"
}

# Drogon. Always source-built because the apt package is not available on every
# supported distro and we want a known-good version (matches build.sh).
need_drogon() {
    pkg-config --exists drogon 2>/dev/null && return 1
    for p in \
        "${PREFIX}/lib/cmake/Drogon/DrogonConfig.cmake" \
        "${PREFIX}/lib64/cmake/Drogon/DrogonConfig.cmake" \
        /usr/local/lib/cmake/Drogon/DrogonConfig.cmake \
        /usr/lib/cmake/Drogon/DrogonConfig.cmake \
        /opt/homebrew/lib/cmake/Drogon/DrogonConfig.cmake
    do
        [ -f "$p" ] && return 1
    done
    return 0
}
build_drogon() {
    need_drogon || { log "Drogon: already available, skipping"; return 0; }
    log "Building Drogon v1.9.12 from source..."
    fetch_git https://github.com/drogonframework/drogon.git v1.9.12 drogon
    cmake -S "${SOURCES_DIR}/drogon" -B "${SOURCES_DIR}/drogon/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_YAML_CONFIG=OFF
    cmake --build "${SOURCES_DIR}/drogon/build" -j"${NPROC}"
    cmake --install "${SOURCES_DIR}/drogon/build"
}

# librdkafka (optional; enable with --kafka)
need_rdkafka() {
    pkg-config --exists rdkafka 2>/dev/null && return 1
    [ -f "${PREFIX}/lib/pkgconfig/rdkafka.pc" ] && return 1
    return 0
}
build_rdkafka() {
    [ "${INSTALL_KAFKA}" = 1 ] || return 0
    need_rdkafka || { log "librdkafka: already available, skipping"; return 0; }
    log "Building librdkafka v2.6.1 from source..."
    fetch_git https://github.com/confluentinc/librdkafka.git v2.6.1 librdkafka
    cd "${SOURCES_DIR}/librdkafka"
    ./configure --prefix="${PREFIX}" --install-deps
    make -j"${NPROC}"
    make install
    cd "${SCRIPT_DIR}"
}

# Build only what is actually missing. build_libuuid / build_jsoncpp are no-ops
# if the apt path already put them on the system.
build_libuuid
build_jsoncpp
build_drogon
build_rdkafka

# ---------------------------------------------------------------------------
# Emit env.sh — build.sh sources this to pick up the prefix
# ---------------------------------------------------------------------------
cat > "${ENV_SH}" <<EOF
# Generated by install_dependencies.sh — do not edit by hand.
# Sourced by build.sh so cmake/pkg-config/ld/cargo pick up the local deps.

_TT_CPP_DEPS_PREFIX="${PREFIX}"
if [ -d "\${_TT_CPP_DEPS_PREFIX}" ]; then
    case ":\${CMAKE_PREFIX_PATH:-}:" in
        *":\${_TT_CPP_DEPS_PREFIX}:"*) : ;;
        *) export CMAKE_PREFIX_PATH="\${_TT_CPP_DEPS_PREFIX}\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}" ;;
    esac
    export PKG_CONFIG_PATH="\${_TT_CPP_DEPS_PREFIX}/lib/pkgconfig:\${_TT_CPP_DEPS_PREFIX}/lib64/pkgconfig\${PKG_CONFIG_PATH:+:\${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="\${_TT_CPP_DEPS_PREFIX}/lib:\${_TT_CPP_DEPS_PREFIX}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    export PATH="\${_TT_CPP_DEPS_PREFIX}/bin:\${PATH}"
fi
unset _TT_CPP_DEPS_PREFIX

# Cargo/rustup state (may live off \$HOME when the home filesystem is tight).
export CARGO_HOME="${CARGO_HOME}"
export RUSTUP_HOME="${RUSTUP_HOME}"
if [ -d "\${CARGO_HOME}/bin" ]; then
    case ":\${PATH}:" in
        *":\${CARGO_HOME}/bin:"*) : ;;
        *) export PATH="\${CARGO_HOME}/bin:\${PATH}" ;;
    esac
fi
EOF

log "Done. Prefix: ${PREFIX}"
log "Wrote env file: ${ENV_SH}"
log "Next: ./build.sh"
