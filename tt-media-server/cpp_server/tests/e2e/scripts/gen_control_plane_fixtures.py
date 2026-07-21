#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Generate tiny synthetic .pb + .devmap fixtures for control-plane CI.

Produces a single-chip, single-chunk prefill/decode table stamped with host
name ``ci-host`` and a matching DeviceMap. No Metal / device hardware needed;
the production worker loads these via ENABLE_KV_TABLE + host-DRAM path.

Usage:
  python3 gen_control_plane_fixtures.py [-o DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "fixtures", "control_plane")
)
# Safelist: --output-dir must resolve under one of these roots.
_ALLOWED_OUTPUT_ROOTS = (
    os.path.realpath(os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "fixtures"))),
    os.path.realpath(os.path.abspath(os.path.join(_CPP_ROOT, "tests", "e2e"))),
    os.path.realpath(tempfile.gettempdir()),
)
_PB2_DIR = os.path.join(_CPP_ROOT, "tt-llm-engine", "tests", "tensors")
sys.path.insert(0, _PB2_DIR)

import kv_chunk_address_table_pb2 as pb  # noqa: E402

HOST_NAME = "ci-host"
CHUNK_N_TOKENS = 32
HEAD_DIM = 32
CHUNK_SIZE_BYTES = CHUNK_N_TOKENS * HEAD_DIM * 4  # uint32 row-major
PREFILL_PB_NAME = "prefill.pb"
DECODE_PB_NAME = "decode.pb"
DEVMAP_NAME = "ci-host.devmap"


def buildTableBytes(chipId: int = 0) -> bytes:
    table = pb.KvChunkAddressTable()
    table.num_layers = 1
    table.max_sequence_length = CHUNK_N_TOKENS
    table.num_slots = 1
    table.chunk_n_tokens = CHUNK_N_TOKENS
    table.chunk_size_bytes = CHUNK_SIZE_BYTES

    fabricNode = table.device_groups.add().fabric_node_ids.add()
    fabricNode.mesh_id = 0
    fabricNode.chip_id = chipId

    host = table.fabric_node_hosts.add()
    host.mesh_id = 0
    host.chip_id = chipId
    host.host_name = HOST_NAME

    entry = table.entries.add()
    entry.slot = 0
    entry.layer = 0
    entry.position = 0
    entry.noc_addr = 0
    entry.size_bytes = CHUNK_SIZE_BYTES
    entry.device_group_index = 0
    return table.SerializeToString()


def isUnderAllowedRoot(resolvedPath: str) -> bool:
    for root in _ALLOWED_OUTPUT_ROOTS:
        if resolvedPath == root or resolvedPath.startswith(root + os.sep):
            return True
    return False


def resolveSafeOutputDir(userPath: str) -> str:
    """Resolve --output-dir and reject paths outside the safelist roots."""
    resolvedPath = os.path.realpath(os.path.abspath(userPath))
    if not isUnderAllowedRoot(resolvedPath):
        allowed = ", ".join(_ALLOWED_OUTPUT_ROOTS)
        raise SystemExit(
            f"ERROR: --output-dir '{userPath}' resolves to '{resolvedPath}', "
            f"which is outside allowed roots: {allowed}"
        )
    return resolvedPath


def safeFixturePath(outputDir: str, fileName: str) -> str:
    """Join a constant fixture name under an already-validated output dir."""
    if fileName not in (PREFILL_PB_NAME, DECODE_PB_NAME, DEVMAP_NAME):
        raise SystemExit(f"ERROR: refusing unexpected fixture name '{fileName}'")
    resolvedPath = os.path.realpath(os.path.join(outputDir, fileName))
    if not resolvedPath.startswith(outputDir + os.sep) and resolvedPath != outputDir:
        raise SystemExit(
            f"ERROR: fixture path '{resolvedPath}' escaped output dir '{outputDir}'"
        )
    return resolvedPath


def writeDevMap(outputDir: str, chipId: int = 0) -> str:
    # mesh chip umd_chip_id — single FabricNode → UMD id 0
    path = safeFixturePath(outputDir, DEVMAP_NAME)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"0 {chipId} 0\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for prefill.pb / decode.pb / ci-host.devmap "
        "(must be under tests/e2e/fixtures, tests/e2e, or the system temp dir)",
    )
    args = parser.parse_args()
    outputDir = resolveSafeOutputDir(args.output_dir)
    os.makedirs(outputDir, exist_ok=True)

    for name in (PREFILL_PB_NAME, DECODE_PB_NAME):
        path = safeFixturePath(outputDir, name)
        with open(path, "wb") as out:
            out.write(buildTableBytes())
        print(f"wrote {path}")

    print(f"wrote {writeDevMap(outputDir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
