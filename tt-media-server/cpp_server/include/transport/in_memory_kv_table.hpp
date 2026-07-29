// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief A hand-built, in-memory KV-cache table.
 *
 * Backs the address-adapter unit tests and the reduced single-host bring-up
 * config: build a small table by hand (a few layers, device groups of 2
 * replicas, multiple channels) without any tt-metal / protobuf dependency. The
 * guarded KvChunkAddressTableAdapter exposes the same IKvTable surface over the
 * real table.
 */
class InMemoryKvTable : public IKvTable {
 public:
  explicit InMemoryKvTable(KvTableConfig config) : config_(config) {}

  /// Register a device group (replica set) and return its index.
  uint32_t addDeviceGroup(std::vector<FabricNode> nodes) {
    const uint32_t index = static_cast<uint32_t>(device_groups_.size());
    device_groups_.push_back(std::move(nodes));
    return index;
  }

  /// Map a fabric node to its host.
  void setHost(const FabricNode& node, std::string host) {
    hosts_[node] = std::move(host);
  }

  /// Place a chunk at (slot, layer, position).
  void setChunk(uint32_t slot, uint32_t layer, uint32_t position,
                ChunkLoc loc) {
    entries_[key(slot, layer, position)] = loc;
  }

  const KvTableConfig& config() const override { return config_; }

  const std::vector<FabricNode>& deviceGroup(uint32_t index) const override {
    return device_groups_.at(index);
  }

  const std::string& hostOf(const FabricNode& node) const override {
    const auto it = hosts_.find(node);
    return it == hosts_.end() ? K_NO_HOST : it->second;
  }

  std::optional<ChunkLoc> lookup(uint32_t slot, uint32_t layer,
                                 uint32_t position) const override {
    const auto it = entries_.find(key(slot, layer, position));
    if (it == entries_.end()) return std::nullopt;
    return it->second;
  }

 private:
  static uint64_t key(uint32_t slot, uint32_t layer, uint32_t position) {
    return (static_cast<uint64_t>(slot) << 48) |
           (static_cast<uint64_t>(layer) << 32) | position;
  }

  KvTableConfig config_;
  std::vector<std::vector<FabricNode>> device_groups_;
  std::unordered_map<FabricNode, std::string, FabricNode::Hash> hosts_;
  std::map<uint64_t, ChunkLoc> entries_;
  inline static const std::string K_NO_HOST{};
};

}  // namespace tt::transport
