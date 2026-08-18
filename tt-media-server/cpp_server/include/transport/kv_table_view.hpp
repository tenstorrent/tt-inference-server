// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief A fabric node identity (mesh, chip), mirroring the disaggregation
 *        layer's FabricNodeId and the real table's proto message.
 *
 * A decode host holds several meshes, so a chunk's device must be identified by
 * (mesh_id, chip_id), not a bare chip id. `encodeDevice` collapses it to the
 * LocalDeviceId the address layer keys on; both sender and receiver encode the
 * same node identically, so their layouts agree.
 */
struct FabricNode {
  uint32_t mesh_id = 0;
  uint32_t chip_id = 0;

  bool operator==(const FabricNode& other) const {
    return mesh_id == other.mesh_id && chip_id == other.chip_id;
  }

  struct Hash {
    std::size_t operator()(const FabricNode& n) const {
      return (static_cast<std::size_t>(n.mesh_id) << 16) ^ n.chip_id;
    }
  };
};

/**
 * @brief Stable LocalDeviceId for a fabric node.
 *
 * `(mesh_id << 16) | chip_id`: the real table has mesh ids well under 2^16 and
 * chip ids under 8, so this is collision-free and identical on both sides.
 */
inline LocalDeviceId encodeDevice(const FabricNode& node) {
  return (static_cast<LocalDeviceId>(node.mesh_id) << 16) | node.chip_id;
}

/**
 * @brief KV-cache table geometry — the scalar config of the real table.
 */
struct KvTableConfig {
  uint32_t num_layers = 0;
  uint32_t num_slots = 0;
  uint32_t max_sequence_length = 0;  ///< In tokens.
  uint32_t chunk_n_tokens = 32;   ///< Tokens per chunk (position granularity).
  uint32_t chunk_size_bytes = 0;  ///< Physical bytes per chunk.

  /// Number of position chunks per (slot, layer).
  uint32_t numPositionChunks() const {
    return chunk_n_tokens == 0 ? 0 : max_sequence_length / chunk_n_tokens;
  }
};

/**
 * @brief Physical location of one KV chunk — mirrors the proto KvCacheEntry's
 *        address fields. A chunk maps to a device *group* (replicas), so a
 *        single chunk may land on several devices at the same NocAddr.
 */
struct ChunkLoc {
  NocAddr noc_addr = 0;
  uint64_t size_bytes = 0;
  uint32_t device_group_index = 0;
};

/**
 * @brief Read-only view over a KV-cache address table.
 *
 * The minimal surface the address adapter needs, decoupled from the concrete
 * table so the adapter builds and unit-tests with no tt-metal / protobuf
 * dependency. Two implementations:
 *   - InMemoryKvTable — hand-built tables for tests and the reduced config;
 *   - KvChunkAddressTableAdapter — a guarded adapter over the real
 *     KvChunkAddressTable (added with the table build guard) that reuses its
 *     protobuf load and accessors.
 */
class IKvTable {
 public:
  virtual ~IKvTable() = default;

  virtual const KvTableConfig& config() const = 0;

  /// Fabric nodes (replicas) of a device group by index.
  virtual const std::vector<FabricNode>& deviceGroup(uint32_t index) const = 0;

  /// Host name a fabric node lives on (empty if unknown).
  virtual const std::string& hostOf(const FabricNode& node) const = 0;

  /**
   * @brief Chunk at (slot, layer, position-in-tokens), or std::nullopt if
   * absent (out of range, or an entry that was never populated). `position`
   * must be chunk-aligned (a multiple of config().chunk_n_tokens).
   */
  virtual std::optional<ChunkLoc> lookup(uint32_t slot, uint32_t layer,
                                         uint32_t position) const = 0;
};

}  // namespace tt::transport
