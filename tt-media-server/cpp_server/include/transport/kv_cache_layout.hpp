// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Local (per-node) device identifier.
 *
 * On a node the host process reaches every TT device by its UMD chip id. This
 * is the per-node identity used to key device I/O; the cross-node identity
 * (mesh_id, chip_id) of the disaggregation layer's FabricNodeId is resolved
 * down to this when the real KvChunkAddressTable is adapted.
 */
using LocalDeviceId = uint32_t;

/**
 * @brief Physical location of a single KV-cache chunk in device DRAM.
 *
 * A lightweight, dependency-free reflection of the disaggregation layer's
 * KvCacheLocation: which device, the NocAddr (`channel << 32 | local_addr`),
 * and the chunk's byte size. The address layer operates purely on these so it
 * builds and unit-tests with no tt-metal / protobuf dependency.
 */
struct KvChunkLocation {
  LocalDeviceId device = 0;
  NocAddr noc_addr = 0;
  uint64_t size_bytes = 0;
};

}  // namespace tt::transport
