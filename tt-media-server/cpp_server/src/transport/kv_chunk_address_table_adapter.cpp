// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_chunk_address_table_adapter.hpp"

#include <utility>

#include "utils/logger.hpp"

#ifdef TT_TRANSPORT_WITH_KV_TABLE
#include <exception>

#include "tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp"
// Protobuf import/export helpers (the build must also compile the matching
// protobuf .cpp + generated .pb.cc and link protobuf — see the helper header).
#include "experimental/disaggregation/kv_chunk_address_table_protobuf.hpp"
#endif

namespace tt::transport {

namespace {
const std::string K_NO_HOST{};
const std::vector<FabricNode> K_NO_GROUP{};
const KvTableConfig K_EMPTY_CONFIG{};
}  // namespace

#ifdef TT_TRANSPORT_WITH_KV_TABLE

namespace dis = tt::tt_metal::experimental::disaggregation;

// Holds the real table plus the caches the IKvTable reference-returning
// accessors need (device groups translated to FabricNode once, config copied).
struct KvChunkAddressTableAdapter::Impl {
  dis::KvChunkAddressTable table;
  KvTableConfig config;
  std::vector<std::vector<FabricNode>> groups;

  explicit Impl(dis::KvChunkAddressTable t) : table(std::move(t)) {
    const auto& c = table.config();
    config = KvTableConfig{c.num_layers, c.num_slots, c.max_sequence_length,
                           c.chunk_n_tokens, c.chunk_size_bytes};
    const std::size_t n = table.num_device_groups();
    groups.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      const dis::DeviceGroup& g =
          table.get_device_group(dis::DeviceGroupIndex{i});
      groups[i].reserve(g.fabric_node_ids.size());
      for (const auto& fnid : g.fabric_node_ids) {
        groups[i].push_back(FabricNode{fnid.mesh_id.get(), fnid.chip_id});
      }
    }
  }

  // Our FabricNode -> tt-metal FabricNodeId for host lookups.
  static tt::tt_fabric::FabricNodeId toFabricNodeId(const FabricNode& n) {
    return tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{n.mesh_id},
                                       n.chip_id);
  }
};

KvChunkAddressTableAdapter::KvChunkAddressTableAdapter(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

KvChunkAddressTableAdapter::~KvChunkAddressTableAdapter() = default;

bool KvChunkAddressTableAdapter::available() { return true; }

std::unique_ptr<KvChunkAddressTableAdapter>
KvChunkAddressTableAdapter::fromProtobuf(const std::string& data) {
  try {
    auto table = dis::import_from_protobuf(data);
    return std::unique_ptr<KvChunkAddressTableAdapter>(
        new KvChunkAddressTableAdapter(
            std::make_unique<Impl>(std::move(table))));
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[KvChunkAddressTableAdapter] import_from_protobuf failed: {}",
                 e.what());
    return nullptr;
  }
}

std::unique_ptr<KvChunkAddressTableAdapter>
KvChunkAddressTableAdapter::fromProtobufFile(const std::string& path) {
  try {
    auto table = dis::import_from_protobuf_file(path);
    return std::unique_ptr<KvChunkAddressTableAdapter>(
        new KvChunkAddressTableAdapter(
            std::make_unique<Impl>(std::move(table))));
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[KvChunkAddressTableAdapter] import_from_protobuf_file({}) failed: {}",
        path, e.what());
    return nullptr;
  }
}

const KvTableConfig& KvChunkAddressTableAdapter::config() const {
  return impl_->config;
}

const std::vector<FabricNode>& KvChunkAddressTableAdapter::deviceGroup(
    uint32_t index) const {
  return index < impl_->groups.size() ? impl_->groups[index] : K_NO_GROUP;
}

const std::string& KvChunkAddressTableAdapter::hostOf(
    const FabricNode& node) const {
  const auto fnid = Impl::toFabricNodeId(node);
  if (!impl_->table.has_host(fnid)) return K_NO_HOST;
  return impl_->table.get_host(fnid);
}

std::optional<ChunkLoc> KvChunkAddressTableAdapter::lookup(
    uint32_t slot, uint32_t layer, uint32_t position) const {
  const KvTableConfig& c = impl_->config;
  // Bounds-check against config so we never trip the real table's arg
  // validation; treat out-of-range / unpopulated entries as absent.
  if (layer >= c.num_layers || slot >= c.num_slots) return std::nullopt;
  if (c.chunk_n_tokens == 0 || position >= c.max_sequence_length) {
    return std::nullopt;
  }
  const dis::KvCacheLocation& loc = impl_->table.lookup(layer, position, slot);
  if (loc.size_bytes == 0) return std::nullopt;  // never populated
  return ChunkLoc{loc.noc_addr, loc.size_bytes, loc.device_group_index.get()};
}

#else  // !TT_TRANSPORT_WITH_KV_TABLE

// Fallback: the real table isn't in the build. Factories no-op so callers fall
// back to InMemoryKvTable; the accessors are defined (for the vtable / link)
// but unreachable since no instance is ever constructed.
struct KvChunkAddressTableAdapter::Impl {};

KvChunkAddressTableAdapter::KvChunkAddressTableAdapter(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

KvChunkAddressTableAdapter::~KvChunkAddressTableAdapter() = default;

bool KvChunkAddressTableAdapter::available() { return false; }

std::unique_ptr<KvChunkAddressTableAdapter>
KvChunkAddressTableAdapter::fromProtobuf(const std::string& /*data*/) {
  TT_LOG_WARN(
      "[KvChunkAddressTableAdapter] fromProtobuf unavailable (built without "
      "TT_TRANSPORT_WITH_KV_TABLE); use InMemoryKvTable");
  return nullptr;
}

std::unique_ptr<KvChunkAddressTableAdapter>
KvChunkAddressTableAdapter::fromProtobufFile(const std::string& /*path*/) {
  TT_LOG_WARN(
      "[KvChunkAddressTableAdapter] fromProtobufFile unavailable (built "
      "without "
      "TT_TRANSPORT_WITH_KV_TABLE); use InMemoryKvTable");
  return nullptr;
}

const KvTableConfig& KvChunkAddressTableAdapter::config() const {
  return K_EMPTY_CONFIG;
}

const std::vector<FabricNode>& KvChunkAddressTableAdapter::deviceGroup(
    uint32_t /*index*/) const {
  return K_NO_GROUP;
}

const std::string& KvChunkAddressTableAdapter::hostOf(
    const FabricNode& /*node*/) const {
  return K_NO_HOST;
}

std::optional<ChunkLoc> KvChunkAddressTableAdapter::lookup(
    uint32_t /*slot*/, uint32_t /*layer*/, uint32_t /*position*/) const {
  return std::nullopt;
}

#endif  // TT_TRANSPORT_WITH_KV_TABLE

}  // namespace tt::transport
