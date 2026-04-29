#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# cpp_server build dependencies. Installs Drogon, JsonCpp, libuuid, and a
# Rust toolchain (>= 1.79, for monostate in tokenizers-cpp) into a
# user-writable prefix — no sudo required. If apt is usable (Docker flow),
# system packages are used instead. Optional: --kafka for librdkafka.
#
# Writes cpp_server/deps/env.sh; build.sh sources it automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/deps"
SRC_DIR="${DEPS_DIR}/sources"
ENV_SH="${DEPS_DIR}/env.sh"

usage() {
    cat <<EOF
Usage: $0 [--prefix DIR] [--kafka] [--force-source]
  --prefix DIR     Install prefix (default: \$HOME/.local, or deps/opt if / is tight)
  --kafka          Also build librdkafka (enables ./build.sh --kafka)
  --force-source   Skip apt even if usable
EOF
}

INSTALL_KAFKA=0 FORCE_SOURCE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)        PREFIX="$2"; shift 2 ;;
        --prefix=*)      PREFIX="${1#*=}"; shift ;;
        --kafka)         INSTALL_KAFKA=1; shift ;;
        --force-source)  FORCE_SOURCE=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

log()  { echo "[install_deps] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Return $pref if its nearest existing ancestor has >= $min_mb free, else $fallback.
roomy() {
    local pref="$1" min_mb="$2" fallback="$3" probe avail
    probe="${pref}"
    while [ ! -d "${probe}" ]; do probe="$(dirname "${probe}")"; done
    avail=$(df -Pm "${probe}" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "${avail:-}" ] && [ "${avail}" -ge "${min_mb}" ]; then
        echo "${pref}"
    else
        echo "${fallback}"
    fi
}

# Install locations — fall back to cpp_server/deps (on /data) when $HOME is tight.
PREFIX="${PREFIX:-$(roomy "${HOME}/.local"  2048 "${DEPS_DIR}/opt")}"
export CARGO_HOME="${CARGO_HOME:-$(roomy  "${HOME}/.cargo"  4096 "${DEPS_DIR}/cargo")}"
export RUSTUP_HOME="${RUSTUP_HOME:-$(roomy "${HOME}/.rustup" 4096 "${DEPS_DIR}/rustup")}"
NPROC=$(nproc 2>/dev/null || echo 4)
mkdir -p "${SRC_DIR}" "${PREFIX}" "${CARGO_HOME}" "${RUSTUP_HOME}"

log "PREFIX=${PREFIX}"
log "CARGO_HOME=${CARGO_HOME}"
log "RUSTUP_HOME=${RUSTUP_HOME}"

# Make prefix visible to cmake/pkg-config/ld; prefer rustup's cargo.
export CMAKE_PREFIX_PATH="${PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${CARGO_HOME}/bin:${PREFIX}/bin:${PATH}"

# --- apt path (Docker flow) ------------------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo && sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
fi
can_apt() {
    have apt-get || return 1
    [ "$(id -u)" -eq 0 ] || [ -n "${SUDO}" ] || return 1
    $SUDO apt-get check >/dev/null 2>&1
}
if [ "${FORCE_SOURCE}" != 1 ] && can_apt; then
    PKGS=(build-essential cmake g++ pkg-config curl git wget
          libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libboost-all-dev)
    [ "${INSTALL_KAFKA}" = 1 ] && PKGS+=(librdkafka-dev)
    log "apt: installing ${PKGS[*]}"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends "${PKGS[@]}"
    $SUDO rm -rf /var/lib/apt/lists/*
else
    log "No apt access — source-building missing libs into ${PREFIX}"
fi

# --- Rust (>= 1.79; tokenizers-cpp/monostate) ------------------------------
rust_ok() {
    have rustc || return 1
    local v mj mn
    v="$(rustc --version 2>/dev/null | awk '{print $2}')"
    mj="${v%%.*}"; mn="${v#*.}"; mn="${mn%%.*}"
    [ -n "${mj}" ] && [ -n "${mn}" ] || return 1
    [ "${mj}" -gt 1 ] || { [ "${mj}" -eq 1 ] && [ "${mn}" -ge 79 ]; }
}
[ -f "${CARGO_HOME}/env" ] && . "${CARGO_HOME}/env"
if ! rust_ok; then
    have rustc && log "rustc $(rustc --version | awk '{print $2}') too old (need >= 1.79)"
    log "Installing rustup (stable) into ${CARGO_HOME}"
    # Skip rustup's "system rust already installed" refusal — we shadow via PATH.
    RUSTUP_INIT_SKIP_PATH_CHECK=yes \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | RUSTUP_INIT_SKIP_PATH_CHECK=yes \
          sh -s -- -y --no-modify-path --default-toolchain stable
    . "${CARGO_HOME}/env"
    rust_ok || { log "ERROR: rustup did not produce rustc >= 1.79"; exit 1; }
fi
log "Using rustc $(rustc --version | awk '{print $2}') ($(command -v rustc))"

# Pre-install components serially so parallel `cargo build` workers during the
# tokenizers-cpp build don't race rustup on auto-component-download.
if have rustup; then
    for c in rust-src rustfmt clippy; do
        rustup component add "$c" >/dev/null 2>&1 || true
    done
fi

# --- Source-build helpers ---------------------------------------------------
fetch_git() {
    # fetch_git <url> <tag> <name>
    local url="$1" tag="$2" out="${SRC_DIR}/$3"
    if [ ! -d "${out}/.git" ]; then
        rm -rf "${out}"
        git clone --depth 1 --branch "${tag}" --recurse-submodules "${url}" "${out}"
    fi
}

fetch_tar() {
    # fetch_tar <url> <name> <topdir-inside-tarball>
    local url="$1" name="$2" top="$3"
    local out="${SRC_DIR}/${name}" tarball="${SRC_DIR}/${name}.tar"
    [ -f "${out}/configure" ] && return 0
    rm -rf "${out}"
    if [ ! -d "${SRC_DIR}/${top}" ]; then
        [ -f "${tarball}" ] || curl -fsSL "${url}" -o "${tarball}"
        tar -xf "${tarball}" -C "${SRC_DIR}"
        rm -f "${tarball}"
    fi
    mv "${SRC_DIR}/${top}" "${out}"
}

cmake_install() {
    # cmake_install <src-dir> [extra-cmake-flags...]
    local src="$1"; shift
    cmake -S "${src}" -B "${src}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        "$@"
    cmake --build   "${src}/build" -j"${NPROC}"
    cmake --install "${src}/build"
}

# --- Individual deps --------------------------------------------------------
build_libuuid() {
    if [ -f "${PREFIX}/include/uuid/uuid.h" ] || [ -f /usr/include/uuid/uuid.h ]; then
        log "libuuid: present"; return 0
    fi
    log "Building libuuid (util-linux 2.39.3)"
    fetch_tar \
        "https://mirrors.edge.kernel.org/pub/linux/utils/util-linux/v2.39/util-linux-2.39.3.tar.xz" \
        "util-linux" "util-linux-2.39.3"
    (
        cd "${SRC_DIR}/util-linux"
        ./configure --prefix="${PREFIX}" \
            --disable-all-programs --enable-libuuid \
            --without-systemd --without-python \
            --without-ncurses --without-ncursesw \
            --disable-makeinstall-chown --disable-makeinstall-setuid \
            --disable-nls >/dev/null
        make -j"${NPROC}"
        make install
    )
}

build_jsoncpp() {
    if [ -f "${PREFIX}/lib/cmake/jsoncpp/jsoncppConfig.cmake" ] \
       || [ -f "${PREFIX}/lib64/cmake/jsoncpp/jsoncppConfig.cmake" ] \
       || pkg-config --exists jsoncpp 2>/dev/null; then
        log "JsonCpp: present"; return 0
    fi
    log "Building JsonCpp 1.9.5"
    fetch_git https://github.com/open-source-parsers/jsoncpp.git 1.9.5 jsoncpp
    cmake_install "${SRC_DIR}/jsoncpp" \
        -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
        -DJSONCPP_WITH_TESTS=OFF \
        -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
        -DJSONCPP_WITH_EXAMPLE=OFF
}

build_drogon() {
    if pkg-config --exists drogon 2>/dev/null; then
        log "Drogon: present"; return 0
    fi
    local p
    for p in "${PREFIX}/lib/cmake/Drogon/DrogonConfig.cmake" \
             "${PREFIX}/lib64/cmake/Drogon/DrogonConfig.cmake" \
             /usr/local/lib/cmake/Drogon/DrogonConfig.cmake \
             /usr/lib/cmake/Drogon/DrogonConfig.cmake \
             /opt/homebrew/lib/cmake/Drogon/DrogonConfig.cmake; do
        [ -f "$p" ] && { log "Drogon: present"; return 0; }
    done
    log "Building Drogon v1.9.12"
    fetch_git https://github.com/drogonframework/drogon.git v1.9.12 drogon
    cmake_install "${SRC_DIR}/drogon" \
        -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_YAML_CONFIG=OFF
}

build_rdkafka() {
    [ "${INSTALL_KAFKA}" = 1 ] || return 0
    if pkg-config --exists rdkafka 2>/dev/null \
       || [ -f "${PREFIX}/lib/pkgconfig/rdkafka.pc" ]; then
        log "librdkafka: present"; return 0
    fi
    log "Building librdkafka v2.6.1"
    fetch_git https://github.com/confluentinc/librdkafka.git v2.6.1 librdkafka
    (
        cd "${SRC_DIR}/librdkafka"
        ./configure --prefix="${PREFIX}" --install-deps
        make -j"${NPROC}"
        make install
    )
}

build_libuuid
build_jsoncpp
build_drogon
build_rdkafka

# --- env.sh — consumed by build.sh -----------------------------------------
cat > "${ENV_SH}" <<EOF
# Generated by install_dependencies.sh — do not edit by hand.
_p="${PREFIX}"
if [ -d "\$_p" ]; then
    case ":\${CMAKE_PREFIX_PATH:-}:" in
        *":\$_p:"*) : ;;
        *) export CMAKE_PREFIX_PATH="\$_p\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}" ;;
    esac
    export PKG_CONFIG_PATH="\$_p/lib/pkgconfig:\$_p/lib64/pkgconfig\${PKG_CONFIG_PATH:+:\${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="\$_p/lib:\$_p/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    export PATH="\$_p/bin:\${PATH}"
fi
unset _p
export CARGO_HOME="${CARGO_HOME}" RUSTUP_HOME="${RUSTUP_HOME}"
if [ -d "\${CARGO_HOME}/bin" ]; then
    case ":\${PATH}:" in
        *":\${CARGO_HOME}/bin:"*) : ;;
        *) export PATH="\${CARGO_HOME}/bin:\${PATH}" ;;
    esac
fi
EOF

log "Done. Prefix=${PREFIX}. Wrote ${ENV_SH}. Next: ./build.sh"
