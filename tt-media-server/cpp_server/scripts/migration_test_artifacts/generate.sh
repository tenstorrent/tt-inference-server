#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# generate.sh — host-local generation of a synthetic KV chunk table + device map
# for a single Galaxy mesh (1 mesh × 32 chips). Run this on the reserved host
# that owns the chips; no SSH, Kafka, or worker deployment.
#
# See README.md in this directory for prerequisites and usage.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CPP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_MIG="${CPP_ROOT}/tt-llm-engine/disaggregation/migration"

OUTPUT_DIR=""
FORCE=0
LAYERS=61
SLOTS=1
MAX_SEQ_LEN=60000
CHUNK_N_TOKENS=32
CHUNK_SIZE_BYTES=19584
DRAM_BASE="0x100000"
DEVICES_PER_RANK=32
GROUP_RANK=0
BUILD_DIR=""
MIG_DIR="${MIG_DIR:-${DEFAULT_MIG}}"
SKIP_BUILD=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --output-dir DIR [options]

Required:
  --output-dir DIR         directory for kv_chunk_table.pb and device_map.txt

Options:
  --force                  overwrite existing outputs
  --layers N               (default: ${LAYERS})
  --slots N                (default: ${SLOTS})
  --max-seq-len N          token positions; must be divisible by --chunk-n-tokens
                           (default: ${MAX_SEQ_LEN})
  --chunk-n-tokens N       (default: ${CHUNK_N_TOKENS})
  --chunk-size-bytes N     BFP8 chunk size (default: ${CHUNK_SIZE_BYTES})
  --dram-base HEX|DEC      first chunk offset (default: ${DRAM_BASE})
  --devices-per-rank N     chips owned by the single host (default: ${DEVICES_PER_RANK})
  --group-rank N           make_test_table --groups value / host-<N> (default: ${GROUP_RANK})
  --build-dir DIR          cmake build dir (default: <migration>/build-test-artifacts)
  --mig-dir DIR            path to disaggregation/migration (default: ${DEFAULT_MIG})
  --skip-build             reuse existing make_test_table / print_local_device_map
  -h, --help               this help

Environment:
  TT_METAL_HOME            built tt-metal tree (default: <tt-llm-engine>/tt-metal)
  TT_MESH_GRAPH_DESC_PATH  BH Galaxy mesh descriptor (default: migration's
                           single_bh_galaxy_torus_x_relaxed.textproto)
  TT_METAL_RUNTIME_ROOT    defaults to TT_METAL_HOME when unset
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

fail_collect() {
  cat >&2 <<EOF

ERROR: $*

Collect for debugging:
  * cmake / build log above
  * print_local_device_map stderr
  * echo "\$TT_METAL_HOME" "\$TT_MESH_GRAPH_DESC_PATH"
  * tt-smi -ls   (or: tt-smi -s) to confirm 32 visible chips
  * Confirm the host has a Galaxy reservation and no other process holds the chips
EOF
  exit 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --force) FORCE=1; shift ;;
      --layers) LAYERS="$2"; shift 2 ;;
      --slots) SLOTS="$2"; shift 2 ;;
      --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
      --chunk-n-tokens) CHUNK_N_TOKENS="$2"; shift 2 ;;
      --chunk-size-bytes) CHUNK_SIZE_BYTES="$2"; shift 2 ;;
      --dram-base) DRAM_BASE="$2"; shift 2 ;;
      --devices-per-rank) DEVICES_PER_RANK="$2"; shift 2 ;;
      --group-rank) GROUP_RANK="$2"; shift 2 ;;
      --build-dir) BUILD_DIR="$2"; shift 2 ;;
      --mig-dir) MIG_DIR="$2"; shift 2 ;;
      --skip-build) SKIP_BUILD=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

validate_shape() {
  is_positive_int "${LAYERS}" || die "--layers must be a positive integer"
  is_positive_int "${SLOTS}" || die "--slots must be a positive integer"
  is_positive_int "${MAX_SEQ_LEN}" || die "--max-seq-len must be a positive integer"
  is_positive_int "${CHUNK_N_TOKENS}" || die "--chunk-n-tokens must be a positive integer"
  is_positive_int "${CHUNK_SIZE_BYTES}" || die "--chunk-size-bytes must be a positive integer"
  is_positive_int "${DEVICES_PER_RANK}" || die "--devices-per-rank must be a positive integer"
  is_nonneg_int "${GROUP_RANK}" || die "--group-rank must be a non-negative integer"

  if (( MAX_SEQ_LEN % CHUNK_N_TOKENS != 0 )); then
    die "--max-seq-len (${MAX_SEQ_LEN}) must be divisible by --chunk-n-tokens (${CHUNK_N_TOKENS})"
  fi

  local chunks_per_layer=$((MAX_SEQ_LEN / CHUNK_N_TOKENS))
  local total_chunks=$((LAYERS * SLOTS * chunks_per_layer))
  local dram_base_dec
  dram_base_dec="$(printf '%d' "${DRAM_BASE}")"
  local final_exclusive=$((dram_base_dec + total_chunks * CHUNK_SIZE_BYTES))
  local max_u32=$((1 << 32))
  if (( final_exclusive > max_u32 )); then
    die "table address range ends at ${final_exclusive} (> 4 GiB). Lower --layers/--max-seq-len/--chunk-size-bytes or --dram-base"
  fi

  CHUNKS_PER_LAYER="${chunks_per_layer}"
  TOTAL_CHUNKS="${total_chunks}"
  LOGICAL_BYTES=$((total_chunks * CHUNK_SIZE_BYTES))
  FINAL_EXCLUSIVE="${final_exclusive}"
}

resolve_paths() {
  [[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

  [[ -d "${MIG_DIR}" ]] || die "migration tree not found: ${MIG_DIR}
Initialize the submodule:
  git submodule update --init --recursive tt-media-server/cpp_server/tt-llm-engine"
  MIG_DIR="$(cd "${MIG_DIR}" && pwd)"

  if [[ -z "${TT_METAL_HOME:-}" ]]; then
    if [[ -d "${CPP_ROOT}/tt-llm-engine/tt-metal" ]]; then
      export TT_METAL_HOME="$(cd "${CPP_ROOT}/tt-llm-engine/tt-metal" && pwd)"
    else
      die "TT_METAL_HOME is unset and ${CPP_ROOT}/tt-llm-engine/tt-metal is missing"
    fi
  fi
  [[ -d "${TT_METAL_HOME}" ]] || die "TT_METAL_HOME is not a directory: ${TT_METAL_HOME}"
  export TT_METAL_HOME
  export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME}}"

  local default_mesh="${MIG_DIR}/single_bh_galaxy_torus_x_relaxed.textproto"
  if [[ -z "${TT_MESH_GRAPH_DESC_PATH:-}" && -f "${default_mesh}" ]]; then
    export TT_MESH_GRAPH_DESC_PATH="${default_mesh}"
  fi

  if [[ -z "${BUILD_DIR}" ]]; then
    BUILD_DIR="${MIG_DIR}/build-test-artifacts"
  fi
  mkdir -p "${BUILD_DIR}"
  BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

  MAKE_TEST_TABLE="${BUILD_DIR}/test/make_test_table"
  PRINT_LOCAL_DEVICE_MAP="${BUILD_DIR}/test/print_local_device_map"
  TABLE_PATH="${OUTPUT_DIR}/kv_chunk_table.pb"
  DEVICE_MAP_PATH="${OUTPUT_DIR}/device_map.txt"
  RAW_MAP_PATH="${OUTPUT_DIR}/print_local_device_map.raw.txt"
}

check_overwrite() {
  local existing=()
  [[ -e "${TABLE_PATH}" ]] && existing+=("${TABLE_PATH}")
  [[ -e "${DEVICE_MAP_PATH}" ]] && existing+=("${DEVICE_MAP_PATH}")
  if ((${#existing[@]} > 0)) && (( FORCE == 0 )); then
    die "refusing to overwrite: ${existing[*]}
Pass --force to replace existing artifacts"
  fi
}

ensure_tools() {
  if (( SKIP_BUILD )); then
    [[ -x "${MAKE_TEST_TABLE}" ]] || die "--skip-build but missing executable: ${MAKE_TEST_TABLE}"
    [[ -x "${PRINT_LOCAL_DEVICE_MAP}" ]] || die "--skip-build but missing executable: ${PRINT_LOCAL_DEVICE_MAP}"
    return 0
  fi

  local metal_build=""
  for candidate in \
    "${TT_METAL_HOME}/build_Release" \
    "${TT_METAL_HOME}/build_RelWithDebInfo" \
    "${TT_METAL_HOME}/build"; do
    if [[ -f "${candidate}/CMakeCache.txt" || -e "${candidate}/lib/libtt_metal.so" || -e "${candidate}/tt_metal/libtt_metal.so" ]]; then
      metal_build="${candidate}"
      break
    fi
  done
  [[ -n "${metal_build}" ]] || die "no built tt-metal under TT_METAL_HOME=${TT_METAL_HOME}
Build tt-metal (build_Release or build_RelWithDebInfo) before generating artifacts"

  echo "[artifacts] configuring migration tools in ${BUILD_DIR} (tt-metal build=${metal_build})"
  cmake -S "${MIG_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DWORKER_SYNTHETIC_ONLY=ON \
    -DBUILD_DEVICE_TESTS=ON \
    -DTT_METAL_DIR="${TT_METAL_HOME}" \
    -DTT_METAL_BUILD_DIR="${metal_build}" \
    || fail_collect "cmake configure failed for ${MIG_DIR}"

  echo "[artifacts] building make_test_table and print_local_device_map"
  cmake --build "${BUILD_DIR}" --target make_test_table print_local_device_map -j"$(nproc)" \
    || fail_collect "cmake build of make_test_table / print_local_device_map failed"

  [[ -x "${MAKE_TEST_TABLE}" ]] || fail_collect "make_test_table not produced at ${MAKE_TEST_TABLE}"
  [[ -x "${PRINT_LOCAL_DEVICE_MAP}" ]] || fail_collect "print_local_device_map not produced at ${PRINT_LOCAL_DEVICE_MAP}
BUILD_DEVICE_TESTS=ON requires a built libtt_metal under TT_METAL_HOME"
}

generate_device_map() {
  echo "[artifacts] running print_local_device_map (TT_METAL_HOME=${TT_METAL_HOME})"
  local tmp
  tmp="$(mktemp "${OUTPUT_DIR}/print_local_device_map.XXXXXX")"
  if ! "${PRINT_LOCAL_DEVICE_MAP}" >"${tmp}" 2>"${OUTPUT_DIR}/print_local_device_map.stderr"; then
    cat "${OUTPUT_DIR}/print_local_device_map.stderr" >&2 || true
    rm -f "${tmp}"
    fail_collect "print_local_device_map failed"
  fi
  mv -f "${tmp}" "${RAW_MAP_PATH}"

  # print_local_device_map emits its "<mesh> <chip> <umd>" tuples on stdout, but
  # MetalContext init loguru + UMD status lines land there too ("2026-... | info | ..."
  # header/footer). Keep only strict tuple lines in the consumed device_map.txt so
  # the worker's send_device_map parser (and our validator) don't trip on log noise.
  local tuple_re='^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]*$'
  if ! grep -E "${tuple_re}" "${RAW_MAP_PATH}" >"${DEVICE_MAP_PATH}"; then
    rm -f "${DEVICE_MAP_PATH}"
    fail_collect "print_local_device_map produced no '<mesh> <chip> <umd>' tuple lines (see ${RAW_MAP_PATH})"
  fi
}

validate_device_map() {
  local line mesh chip umd
  local -A meshes=() chips=() umds=()
  local count=0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    # shellcheck disable=SC2034
    read -r mesh chip umd _ <<<"${line}" || die "malformed device-map line: ${line}"
    [[ -n "${mesh}" && -n "${chip}" && -n "${umd}" ]] \
      || die "device-map line must be 'mesh chip umd_chip_id': ${line}"
    meshes["${mesh}"]=1
    if [[ -n "${chips[${chip}]:-}" ]]; then
      die "duplicate chip id ${chip} in device map"
    fi
    chips["${chip}"]=1
    if [[ -n "${umds[${umd}]:-}" ]]; then
      die "duplicate ASIC unique id ${umd} in device map"
    fi
    umds["${umd}"]=1
    count=$((count + 1))
  done <"${DEVICE_MAP_PATH}"

  (( count == DEVICES_PER_RANK )) \
    || die "device map has ${count} entries; expected ${DEVICES_PER_RANK}"
  ((${#meshes[@]} == 1)) \
    || die "device map spans ${#meshes[@]} meshes; expected exactly 1 for this 1-mesh recipe"

  local i
  for ((i = 0; i < DEVICES_PER_RANK; i++)); do
    [[ -n "${chips[${i}]:-}" ]] || die "device map missing chip id ${i}"
  done

  MESH_COUNT=1
  DEVICE_COUNT="${count}"
}

generate_table() {
  echo "[artifacts] generating KV table (${LAYERS} layers, ${SLOTS} slots, ${MAX_SEQ_LEN} positions)"
  "${MAKE_TEST_TABLE}" \
    --output "${TABLE_PATH}" \
    --layers "${LAYERS}" \
    --slots "${SLOTS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --chunk-n-tokens "${CHUNK_N_TOKENS}" \
    --chunk-size-bytes "${CHUNK_SIZE_BYTES}" \
    --groups "${GROUP_RANK}" \
    --devices-per-rank "${DEVICES_PER_RANK}" \
    --dram-base "${DRAM_BASE}" \
    >/dev/null \
    || fail_collect "make_test_table failed"

  [[ -s "${TABLE_PATH}" ]] || die "generated table is empty: ${TABLE_PATH}"
}

print_summary() {
  cat <<EOF
[artifacts] OK
  host=$(hostname -s)
  output_dir=${OUTPUT_DIR}
  table=${TABLE_PATH}
  device_map=${DEVICE_MAP_PATH}
  topology=1 mesh / ${DEVICE_COUNT} chips (validated)
  table_host=host-${GROUP_RANK}
  layers=${LAYERS} slots=${SLOTS} max_seq_len=${MAX_SEQ_LEN}
  chunk_n_tokens=${CHUNK_N_TOKENS} chunk_size_bytes=${CHUNK_SIZE_BYTES}
  chunks_per_layer=${CHUNKS_PER_LAYER} total_chunks=${TOTAL_CHUNKS}
  logical_bytes=${LOGICAL_BYTES}
  dram_base=${DRAM_BASE} final_exclusive_addr=${FINAL_EXCLUSIVE}
  TT_METAL_HOME=${TT_METAL_HOME}
  TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}
EOF
}

main() {
  parse_args "$@"
  validate_shape
  resolve_paths
  check_overwrite
  ensure_tools
  generate_device_map
  validate_device_map
  generate_table
  print_summary
}

main "$@"
