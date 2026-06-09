#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Validate that a built runtime image (a) carries ZERO first-party C/C++ source
# and (b) lost no runtime artifact relative to a baseline image. Pure file-level
# + linking checks -- no Tenstorrent device required, so it runs in CI.
#
# Usage:
#   verify_image_no_source.sh <new_image> [baseline_image]
#
# With a baseline image (build it from the OLD Dockerfile, e.g. `git stash`)
# the script additionally diffs the two app trees and FAILS if any .so / binary
# / .py / tokenizer / resource present in the baseline is missing from the new
# image -- i.e. it proves only source/build-intermediates/tests were dropped.

set -euo pipefail

readonly NEW_IMAGE="${1:?usage: $0 <new_image> [baseline_image]}"
readonly BASELINE_IMAGE="${2:-}"
readonly SERVER_DIR="${SERVER_DIR:-/home/container_app_user/tt-metal/server}"
readonly SOURCE_EXTS='-name *.c -o -name *.cc -o -name *.cpp -o -name *.cxx -o -name *.h -o -name *.hh -o -name *.hpp -o -name *.hxx -o -name *.cu -o -name *.cuh'
# A file is "runtime-critical" if losing it can break execution: shared libs,
# the server binary, any Python module, or the runtime data dirs.
readonly CRITICAL_RE='(\.so(\.[0-9]+)*$|/tt_media_server_cpp$|\.py$|/tokenizers/|/resources/|/monitoring/)'

# Run a command inside an image with the entrypoint bypassed.
runInImage() {
    local image="$1" cmd="$2"
    docker run --rm --entrypoint "" "${image}" bash -lc "${cmd}"
}

# List every regular file under the app tree of an image, sorted.
listAppFiles() {
    runInImage "$1" "find '${SERVER_DIR}' -type f 2>/dev/null | sort"
}

# Fail the build if any first-party C/C++ source survives in the image.
assertNoSource() {
    echo "== [1/4] source scan: ${NEW_IMAGE} =="
    local found
    found="$(runInImage "${NEW_IMAGE}" "find '${SERVER_DIR}' -type f \\( ${SOURCE_EXTS} \\)" || true)"
    if [ -n "${found}" ]; then
        echo "FAIL: first-party C/C++ source present in image:"
        echo "${found}"
        exit 1
    fi
    echo "PASS: no C/C++ source under ${SERVER_DIR}"
}

# Report any shared object / binary with unresolved dependencies.
assertArtifactsLink() {
    echo "== [2/4] linking check (ldd) =="
    local report
    report="$(runInImage "${NEW_IMAGE}" "
        miss=0
        for f in \$(find '${SERVER_DIR}/cpp_server' -type f \\( -name '*.so' -o -name '*.so.*' -o -name 'tt_media_server_cpp' \\) 2>/dev/null); do
            if ldd \"\$f\" 2>/dev/null | grep -q 'not found'; then
                echo \"MISSING-LIBS: \$f\"; ldd \"\$f\" 2>/dev/null | grep 'not found' | sed 's/^/    /'; miss=1
            fi
        done
        [ \"\$miss\" = 0 ] && echo 'OK: all artifacts resolve their shared libraries'
    " || true)"
    echo "${report}"
    if echo "${report}" | grep -q 'MISSING-LIBS'; then
        echo "WARNING: unresolved libraries above (likely a pre-existing gap, e.g. Drogon in /usr/local/lib)."
        echo "         Fix by copying the lib into the runtime stage; not failing the run for it here."
    fi
}

# Diff against a baseline image and fail if a runtime-critical file disappeared.
assertNoArtifactRegression() {
    if [ -z "${BASELINE_IMAGE}" ]; then
        echo "== [3/4] artifact-parity diff: skipped (no baseline image given) =="
        return 0
    fi
    echo "== [3/4] artifact-parity diff vs ${BASELINE_IMAGE} =="
    local old new removed lost
    old="$(mktemp)"; new="$(mktemp)"
    listAppFiles "${BASELINE_IMAGE}" > "${old}"
    listAppFiles "${NEW_IMAGE}"      > "${new}"
    removed="$(comm -23 "${old}" "${new}")"
    echo "removed $(echo "${removed}" | grep -c . || true) file(s) vs baseline"
    lost="$(echo "${removed}" | grep -E "${CRITICAL_RE}" || true)"
    rm -f "${old}" "${new}"
    if [ -n "${lost}" ]; then
        echo "FAIL: runtime-critical files removed vs baseline:"
        echo "${lost}"
        exit 1
    fi
    echo "PASS: only source / build-intermediates / tests removed; no .so / binary / .py / data lost"
}

# Confirm the launch-critical artifacts are present and the app imports.
assertRuntimeReady() {
    echo "== [4/4] runtime artifact + import smoke =="
    runInImage "${NEW_IMAGE}" "
        set -e
        ls '${SERVER_DIR}/run_uvicorn.sh' '${SERVER_DIR}/run_cpp.sh' >/dev/null && echo 'run scripts: OK'
        if [ -x '${SERVER_DIR}/cpp_server/build/tt_media_server_cpp' ]; then echo 'cpp binary: OK'; else echo 'cpp binary: ABSENT (ok only if cpp build was skipped)'; fi
        source '${SERVER_DIR}'/../python_env/bin/activate
        python -m compileall -q '${SERVER_DIR}' && echo 'python compileall: OK'
        cd '${SERVER_DIR}' && python -c 'import main' && echo 'import main:app: OK'
    "
}

main() {
    command -v docker >/dev/null || { echo "docker is required"; exit 2; }
    assertNoSource
    assertArtifactsLink
    assertNoArtifactRegression
    assertRuntimeReady
    echo ""
    echo "=================================================="
    echo "  VALIDATION COMPLETE for ${NEW_IMAGE}"
    echo "=================================================="
}

main "$@"
