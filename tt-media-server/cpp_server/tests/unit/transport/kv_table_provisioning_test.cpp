// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_table_provisioning.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport_test_fakes.hpp"

#ifndef KV_TABLE_PREFILL_PB_DEFAULT
#define KV_TABLE_PREFILL_PB_DEFAULT ""
#endif
#ifndef KV_TABLE_DECODE_PB_DEFAULT
#define KV_TABLE_DECODE_PB_DEFAULT ""
#endif

namespace tt::transport {
namespace {

using namespace std::chrono_literals;
using test::BlockingFakeTransport;
using test::closePipe;
using test::Pipe;

// A connected channel pair over crossed pipes, with fast timings so failure
// paths don't stall the suite.
struct ChannelPair {
  std::shared_ptr<Pipe> ab{std::make_shared<Pipe>()};
  std::shared_ptr<Pipe> ba{std::make_shared<Pipe>()};
  std::shared_ptr<BlockingFakeTransport> senderTp{
      std::make_shared<BlockingFakeTransport>(/*in=*/ba, /*out=*/ab)};
  std::shared_ptr<BlockingFakeTransport> receiverTp{
      std::make_shared<BlockingFakeTransport>(/*in=*/ab, /*out=*/ba)};
  KvControlChannel senderCh{senderTp, 2000ms, 1ms};
  KvControlChannel receiverCh{receiverTp, 2000ms, 1ms};
};

std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string{v} : std::string{fallback};
}

bool readable(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

// ---------------------------------------------------------------------------
// Wire exchange — runs in every build (no real table needed)
// ---------------------------------------------------------------------------

TEST(KvTableProvisioning, ExchangeSwapsBlobsBetweenRoles) {
  ChannelPair p;
  const std::vector<uint8_t> prefillBlob{1, 2, 3, 4};
  const std::vector<uint8_t> decodeBlob{9, 8, 7};

  std::optional<std::vector<uint8_t>> receiverGot;
  std::thread receiver([&] {
    receiverGot =
        exchangeTableBlob(p.receiverCh, TableExchangeRole::Receiver, decodeBlob);
  });
  const auto senderGot =
      exchangeTableBlob(p.senderCh, TableExchangeRole::Sender, prefillBlob);
  receiver.join();

  ASSERT_TRUE(senderGot.has_value());
  EXPECT_EQ(*senderGot, decodeBlob);  // sender obtained the decode blob
  ASSERT_TRUE(receiverGot.has_value());
  EXPECT_EQ(*receiverGot, prefillBlob);  // receiver obtained the prefill blob
}

TEST(KvTableProvisioning, ExchangeReturnsNulloptOnClosedChannel) {
  ChannelPair p;
  closePipe(p.ba);  // the sender's inbound is dead — no peer will ever reply
  const auto got =
      exchangeTableBlob(p.senderCh, TableExchangeRole::Sender, {1, 2, 3});
  EXPECT_FALSE(got.has_value());
}

TEST(KvTableProvisioning, LoadMissingFileReturnsNullopt) {
  EXPECT_FALSE(loadKvTableFile("/nonexistent/path/to/table.pb").has_value());
}

TEST(KvTableProvisioning, DeserializeEmptyBlobReturnsNull) {
  EXPECT_EQ(deserializeKvTable({}), nullptr);
}

// ---------------------------------------------------------------------------
// Real tables — gated on ENABLE_KV_TABLE + the .pb files being present
// ---------------------------------------------------------------------------

TEST(KvTableProvisioningRealTables, LoadExchangeAndResolve) {
  if (!KvChunkAddressTableAdapter::available()) {
    GTEST_SKIP() << "ENABLE_KV_TABLE is OFF; real-table exchange not built";
  }
  const std::string prefillPath =
      envOr("KV_TABLE_PREFILL_PB", KV_TABLE_PREFILL_PB_DEFAULT);
  const std::string decodePath =
      envOr("KV_TABLE_DECODE_PB", KV_TABLE_DECODE_PB_DEFAULT);
  if (!readable(prefillPath) || !readable(decodePath)) {
    GTEST_SKIP() << "missing real .pb files (set KV_TABLE_PREFILL_PB / "
                    "KV_TABLE_DECODE_PB)";
  }

  // Each side loads ONLY its own table from disk.
  auto prefill = loadKvTableFile(prefillPath);
  ASSERT_TRUE(prefill.has_value()) << "load prefill " << prefillPath;
  auto decode = loadKvTableFile(decodePath);
  ASSERT_TRUE(decode.has_value()) << "load decode " << decodePath;

  // Over the control channel, the prefill side obtains the decode table and the
  // decode side obtains the prefill table.
  ChannelPair p;
  std::shared_ptr<const IKvTable> receiverGotPrefill;
  std::thread receiver([&] {
    receiverGotPrefill = provisionPeerTable(
        p.receiverCh, TableExchangeRole::Receiver, decode->blob);
  });
  auto senderGotDecode =
      provisionPeerTable(p.senderCh, TableExchangeRole::Sender, prefill->blob);
  receiver.join();

  ASSERT_NE(senderGotDecode, nullptr) << "sender did not obtain decode table";
  ASSERT_NE(receiverGotPrefill, nullptr) << "receiver did not obtain prefill";

  // The table the sender received over the wire must address identically to the
  // decode table loaded directly from disk.
  auto direct = decode->table->lookup(/*slot=*/0, /*layer=*/0, /*position=*/0);
  auto viaWire =
      senderGotDecode->lookup(/*slot=*/0, /*layer=*/0, /*position=*/0);
  ASSERT_TRUE(direct.has_value());
  ASSERT_TRUE(viaWire.has_value());
  EXPECT_EQ(viaWire->noc_addr, direct->noc_addr);
  EXPECT_EQ(viaWire->size_bytes, direct->size_bytes);
  EXPECT_EQ(viaWire->device_group_index, direct->device_group_index);

  // And the receiver's wire-obtained prefill table matches the direct one.
  auto prefillDirect = prefill->table->lookup(0, 0, 0);
  auto prefillViaWire = receiverGotPrefill->lookup(0, 0, 0);
  if (prefillDirect.has_value()) {
    ASSERT_TRUE(prefillViaWire.has_value());
    EXPECT_EQ(prefillViaWire->noc_addr, prefillDirect->noc_addr);
  }
}

}  // namespace
}  // namespace tt::transport
