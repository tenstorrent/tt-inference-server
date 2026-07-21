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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_PB2_DIR = os.path.join(_CPP_ROOT, "tt-llm-engine", "tests", "tensors")
sys.path.insert(0, _PB2_DIR)

import kv_chunk_address_table_pb2 as pb  # noqa: E402

HOST_NAME = "ci-host"
CHUNK_N_TOKENS = 32
HEAD_DIM = 32
CHUNK_SIZE_BYTES = CHUNK_N_TOKENS * HEAD_DIM * 4  # uint32 row-major


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


def writeDevMap(path: str, chipId: int = 0) -> None:
    # mesh chip umd_chip_id — single FabricNode → UMD id 0
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"0 {chipId} 0\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    defaultOut = os.path.join(_SCRIPT_DIR, "..", "fixtures", "control_plane")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.abspath(defaultOut),
        help="Directory for prefill.pb / decode.pb / ci-host.devmap",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for name in ("prefill.pb", "decode.pb"):
        path = os.path.join(args.output_dir, name)
        with open(path, "wb") as out:
            out.write(buildTableBytes())
        print(f"wrote {path}")

    devmapPath = os.path.join(args.output_dir, "ci-host.devmap")
    writeDevMap(devmapPath)
    print(f"wrote {devmapPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
