# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#!/usr/bin/env bash
# Installs mpi4py compiled against the ULFM-enabled OpenMPI on this host.
#
# ULFM (User-Level Failure Mitigation) is required because libtt_metal.so is
# compiled against an ULFM-enabled OpenMPI and exports MPIX_Comm_revoke. If
# mpi4py links against a non-ULFM build (e.g. system OpenMPI 4.1.2), the
# dynamic linker fails with "undefined symbol: MPIX_Comm_revoke" at import.
#
# Assumes the python environment is already active (uv available on PATH).
#
# Usage:
#   1. source your environment (e.g. source python_env/bin/activate)
#   2. run this script: $ bash scripts/install_mpi4py_ulfm.sh

set -euo pipefail

find_ulfm_prefix() {
    for ompi_info_bin in /opt/*/bin/ompi_info; do
        [[ -x "$ompi_info_bin" ]] || continue
        if "$ompi_info_bin" 2>/dev/null | grep -q "Fault Tolerance support: yes"; then
            echo "${ompi_info_bin%/bin/ompi_info}"
            return 0
        fi
    done
    return 1
}

echo "Searching for ULFM-enabled OpenMPI under /opt/ ..."
if ! ULFM_PREFIX=$(find_ulfm_prefix); then
    echo "ERROR: No ULFM-enabled OpenMPI found under /opt/." >&2
    echo "       Install one with --with-ft=ulfm and retry." >&2
    exit 1
fi

echo "Found ULFM OpenMPI: $ULFM_PREFIX"
"$ULFM_PREFIX/bin/ompi_info" 2>/dev/null | grep --color=never -E "Open MPI:|Fault Tolerance"

MPICC="$ULFM_PREFIX/bin/mpicc"

echo ""
echo "Installing mpi4py from source against $MPICC ..."
MPICC="$MPICC" uv pip install --no-binary mpi4py mpi4py --reinstall

echo ""
echo "Verifying linkage ..."
MPI4PY_SO=$(python -c "import mpi4py, os; print(os.path.dirname(mpi4py.__file__))" 2>/dev/null)
MPI4PY_SO=$(find "$MPI4PY_SO" -name "MPI.cpython-*.so" 2>/dev/null | head -1)
if [[ -z "$MPI4PY_SO" ]]; then
    echo "WARNING: Could not find mpi4py MPI.so to verify." >&2
else
    echo "ldd $MPI4PY_SO | grep mpi:"
    ldd "$MPI4PY_SO" | grep -i mpi
fi

echo ""
echo "Done. mpi4py is linked against the ULFM OpenMPI at $ULFM_PREFIX"
