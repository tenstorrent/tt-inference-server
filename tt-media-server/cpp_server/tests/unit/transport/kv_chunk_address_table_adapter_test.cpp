// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Real-table test: loads an actual serialized KvChunkAddressTable (.pb) through
// the guarded KvChunkAddressTableAdapter and checks that real device NoC
// addresses resolve. Only built when ENABLE_KV_TABLE=ON (the CMake target
// defines TT_TRANSPORT_WITH_KV_TABLE and provides KV_TABLE_PB_DEFAULT).
//
// The .pb to read is taken from the KV_TABLE_PB env var if set, else the
// compile-time default (the prefill/decoder table produced by the model
// runner). If neither file is readable the test SKIPs, so it never fails CI on
// a missing artifact while still satisfying the Phase-0 "resolves real NoC
// addresses" criterion when the file is present.

#include "transport/kv_chunk_address_table_adapter.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>

namespace tt::transport {
namespace {

#ifndef KV_TABLE_PB_DEFAULT
#define KV_TABLE_PB_DEFAULT ""
#endif

std::string tablePath() {
  if (const char* env = std::getenv("KV_TABLE_PB"); env && *env) return env;
  return KV_TABLE_PB_DEFAULT;
}

bool readable(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

TEST(KvChunkAddressTableAdapterRealTable, AvailableWhenGuardOn) {
  EXPECT_TRUE(KvChunkAddressTableAdapter::available());
}

TEST(KvChunkAddressTableAdapterRealTable, ResolvesRealNocAddresses) {
  const std::string path = tablePath();
  if (!readable(path)) {
    GTEST_SKIP() << "no readable KV table .pb (set KV_TABLE_PB or place the "
                    "default at "
                 << KV_TABLE_PB_DEFAULT << ")";
  }

  auto table = KvChunkAddressTableAdapter::fromProtobufFile(path);
  ASSERT_NE(table, nullptr) << "failed to load " << path;

  const KvTableConfig& cfg = table->config();
  EXPECT_GT(cfg.num_layers, 0u);
  EXPECT_GT(cfg.num_slots, 0u);
  EXPECT_GT(cfg.max_sequence_length, 0u);
  EXPECT_GT(cfg.chunk_n_tokens, 0u);
  EXPECT_GT(cfg.chunk_size_bytes, 0u);

  // The first chunk (slot 0, layer 0, position 0) must resolve to a populated
  // entry: a real, non-zero NoC address and the table's chunk size.
  auto loc = table->lookup(/*slot=*/0, /*layer=*/0, /*position=*/0);
  ASSERT_TRUE(loc.has_value()) << "slot 0/layer 0/pos 0 unexpectedly absent";
  EXPECT_NE(loc->noc_addr, 0u);
  EXPECT_EQ(loc->size_bytes, cfg.chunk_size_bytes);

  // The chunk's device group must list at least one replica fabric node.
  const std::vector<FabricNode>& group =
      table->deviceGroup(loc->device_group_index);
  EXPECT_FALSE(group.empty())
      << "device_group_index " << loc->device_group_index << " has no nodes";
}

}  // namespace
}  // namespace tt::transport
