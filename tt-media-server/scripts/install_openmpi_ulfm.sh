#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# install_openmpi_ulfm.sh — build OpenMPI 5 from source with ULFM
# (User-Level Failure Mitigation, --with-ft=ulfm) and install it under
# /opt/openmpi-v<version>-ulfm. Matches the on-host layout the migration
# launchers (migration_verify_launcher.sh, launch_migration_endpoints.sh)
# and the disaggregation/migration CMakeLists hard-code at MPI_HOME.
#
# Why ULFM? PRRTE's cross-job MPI_Comm_accept/connect, used by the migration
# layer to pair endpoints, requires OpenMPI 5; OpenMPI 4's ORTE has the
# multi-host cross-job bring-up bug ompi/#6818. ULFM also re-aligns the worker
# with libtt_metal.so (tt-metal is built against the same ULFM v5).
#
# Network access (configure downloads release tarballs) is required.
#
# Environment overrides:
#   OPENMPI_VERSION   default: 5.0.7
#   PREFIX            default: /opt/openmpi-v${OPENMPI_VERSION}-ulfm
#   JOBS              default: $(nproc)
#   PRTE_VERSION      (optional) checkout PRRTE submodule at a specific tag

set -euo pipefail

OPENMPI_VERSION="${OPENMPI_VERSION:-5.0.7}"
PREFIX="${PREFIX:-/opt/openmpi-v${OPENMPI_VERSION}-ulfm}"
JOBS="${JOBS:-$(nproc)}"

OPENMPI_MAJOR_MINOR="${OPENMPI_VERSION%.*}"           # e.g. "5.0"
TARBALL="openmpi-${OPENMPI_VERSION}.tar.bz2"
URL="https://download.open-mpi.org/release/open-mpi/v${OPENMPI_MAJOR_MINOR}/${TARBALL}"

echo "==> Installing OpenMPI ${OPENMPI_VERSION} (with ULFM) to ${PREFIX}"
echo "==> JOBS=${JOBS}"

WORKDIR="$(mktemp -d -t openmpi-ulfm-build.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

cd "${WORKDIR}"
echo "==> Downloading ${URL}"
curl -fsSL "${URL}" -o "${TARBALL}"
tar -xjf "${TARBALL}"
cd "openmpi-${OPENMPI_VERSION}"

echo "==> configure --with-ft=ulfm"
# --with-ft=ulfm    : enable User-Level Failure Mitigation (MPIX_Comm_revoke etc.).
# --enable-mpi-ext  : also enable ULFM's MPI extensions API used by libtt_metal.
# --enable-prte-prefix-by-default
#                   : the PRRTE launcher embeds PREFIX so prte/prun/prted resolve
#                     their lib paths without LD_LIBRARY_PATH — required for
#                     ssh-launched daemons on bare-metal hosts. The migration
#                     launchers use prte/prun directly (workflow_exabox-deploy
#                     also pins PRTE_MCA_plm_ssh_pass_libpath at this prefix).
# --without-cuda    : we don't need CUDA bindings for this image.
./configure \
    --prefix="${PREFIX}" \
    --with-ft=ulfm \
    --enable-mpi-ext=ftmpi \
    --enable-prte-prefix-by-default \
    --without-cuda \
    --disable-silent-rules

echo "==> make -j${JOBS}"
make -j"${JOBS}"

echo "==> make install"
make install

echo "==> ompi_info sanity check"
"${PREFIX}/bin/ompi_info" --version
"${PREFIX}/bin/ompi_info" 2>/dev/null | grep -E "Open MPI:|Fault Tolerance" || true

# A tiny smoke check: ULFM symbol must be exported in libmpi so that
# downstream consumers (libtt_metal.so, mpi4py-ulfm) link cleanly.
LIBMPI="${PREFIX}/lib/libmpi.so"
if [[ -e "${LIBMPI}" ]] && command -v nm >/dev/null 2>&1; then
    if ! nm --dynamic --defined-only "${LIBMPI}" 2>/dev/null | grep -q MPIX_Comm_revoke; then
        echo "ERROR: MPIX_Comm_revoke missing from ${LIBMPI} — ULFM was not enabled." >&2
        exit 1
    fi
    echo "==> verified: MPIX_Comm_revoke is exported by libmpi.so"
fi

echo "==> Done. PRRTE/OpenMPI installed at ${PREFIX}"
