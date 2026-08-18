// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief IKvTable adapter over the real disaggregation KvChunkAddressTable.
 *
 * Wraps a KvChunkAddressTable (loaded from its protobuf wire format) and
 * presents it through the dependency-free IKvTable interface, translating the
 * tt-metal types at the boundary (FabricNodeId <-> FabricNode, KvCacheLocation
 * -> ChunkLoc, config fields, arg order). This is the *only* place in the
 * transport code that touches the real table type, so the address adapter and
 * everything above it stay unchanged whether they run against this or the
 * InMemoryKvTable.
 *
 * Built behind the TT_TRANSPORT_WITH_KV_TABLE guard (tt-metal + protobuf +
 * migration table sources). When the guard is off, the factories return nullptr
 * and available() is false, so transport_lib still builds and links in every
 * configuration — callers fall back to InMemoryKvTable. The tt-metal headers
 * are hidden behind a pimpl, so this header has no tt-metal dependency.
 */
class KvChunkAddressTableAdapter : public IKvTable {
 public:
  /// Load from a serialized KvChunkAddressTable protobuf (e.g. a .pb's bytes).
  /// @return the adapter, or nullptr if the guard is off or parsing fails.
  static std::unique_ptr<KvChunkAddressTableAdapter> fromProtobuf(
      const std::string& data);

  /// Load from a serialized KvChunkAddressTable protobuf file path.
  static std::unique_ptr<KvChunkAddressTableAdapter> fromProtobufFile(
      const std::string& path);

  /// True if compiled with the table guard (otherwise the factories no-op).
  static bool available();

  ~KvChunkAddressTableAdapter() override;
  KvChunkAddressTableAdapter(const KvChunkAddressTableAdapter&) = delete;
  KvChunkAddressTableAdapter& operator=(const KvChunkAddressTableAdapter&) =
      delete;

  const KvTableConfig& config() const override;
  const std::vector<FabricNode>& deviceGroup(uint32_t index) const override;
  const std::string& hostOf(const FabricNode& node) const override;
  std::optional<ChunkLoc> lookup(uint32_t slot, uint32_t layer,
                                 uint32_t position) const override;

 private:
  struct Impl;
  explicit KvChunkAddressTableAdapter(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::transport
