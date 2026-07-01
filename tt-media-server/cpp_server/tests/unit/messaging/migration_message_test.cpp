// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/migration_message.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace tt::messaging {
namespace {

using tt::services::KVCacheBlockRef;
using tt::services::MigrationStatus;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Json::Value parseJson(const std::string& payload) {
  Json::Value root;
  Json::CharReaderBuilder builder;
  std::istringstream iss(payload);
  std::string errs;
  EXPECT_TRUE(Json::parseFromStream(builder, iss, &root, &errs))
      << "Failed to reparse produced JSON: " << errs
      << "\nPayload: " << payload;
  return root;
}

MigrationRequestMessage makeRequest() {
  return MigrationRequestMessage{
      .migration_id = 42,
      .src_slot = 1,
      .dst_slot = 2,
      .layer_id = 7,
      .position_start = 100,
      .position_end = 200,
  };
}

MigrationResponseMessage makeResponse(MigrationStatus status) {
  return MigrationResponseMessage{.migration_id = 99, .status = status};
}

// ---------------------------------------------------------------------------
// MigrationRequestMessage
// ---------------------------------------------------------------------------

TEST(MigrationRequestMessageWire, RoundTripPreservesAllFields) {
  const auto in = makeRequest();
  const std::string wire = serialize(in);

  const auto out = parseMigrationRequest(wire);
  ASSERT_TRUE(out.has_value()) << "parseMigrationRequest rejected: " << wire;

  EXPECT_EQ(out->migration_id, in.migration_id);
  EXPECT_EQ(out->src_slot, in.src_slot);
  EXPECT_EQ(out->dst_slot, in.dst_slot);
  EXPECT_EQ(out->layer_id, in.layer_id);
  EXPECT_EQ(out->position_start, in.position_start);
  EXPECT_EQ(out->position_end, in.position_end);
}

TEST(MigrationRequestMessageWire, SerializeEmitsAllExpectedFields) {
  // The wire shape is part of the contract with the Python frontend / consumer
  // tooling — guard it explicitly so a careless field rename can't silently
  // ship.
  const auto wire = serialize(makeRequest());
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("migration_id"));
  EXPECT_TRUE(root.isMember("src_slot"));
  EXPECT_TRUE(root.isMember("dst_slot"));
  EXPECT_TRUE(root.isMember("layer_id"));
  EXPECT_TRUE(root.isMember("position_start"));
  EXPECT_TRUE(root.isMember("position_end"));
  EXPECT_EQ(root.size(), 6u);
}

TEST(MigrationRequestMessageWire, HandlesMaxUint64MigrationId) {
  // migration_id is uint64_t and the JSON encoder must not silently truncate
  // it to a signed int (a regression we'd never spot in normal-size tests).
  MigrationRequestMessage in = makeRequest();
  in.migration_id = std::numeric_limits<uint64_t>::max();

  const auto out = parseMigrationRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->migration_id, std::numeric_limits<uint64_t>::max());
}

TEST(MigrationRequestMessageWire, HandlesMaxUint32Slots) {
  MigrationRequestMessage in = makeRequest();
  in.src_slot = std::numeric_limits<uint32_t>::max();
  in.dst_slot = std::numeric_limits<uint32_t>::max();
  in.layer_id = std::numeric_limits<uint32_t>::max();
  in.position_start = std::numeric_limits<uint32_t>::max();
  in.position_end = std::numeric_limits<uint32_t>::max();

  const auto out = parseMigrationRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->src_slot, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->dst_slot, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->layer_id, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->position_start, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->position_end, std::numeric_limits<uint32_t>::max());
}

TEST(MigrationRequestMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseMigrationRequest("not json at all").has_value());
  EXPECT_FALSE(parseMigrationRequest("").has_value());
  EXPECT_FALSE(parseMigrationRequest("{ unterminated").has_value());
}

TEST(MigrationRequestMessageParse, RejectsMissingRequiredField) {
  // Every required field, dropped one at a time.
  for (const char* dropped : {"migration_id", "src_slot", "dst_slot",
                              "layer_id", "position_start", "position_end"}) {
    Json::Value root;
    root["migration_id"] = 1;
    root["src_slot"] = 2;
    root["dst_slot"] = 3;
    root["layer_id"] = 4;
    root["position_start"] = 5;
    root["position_end"] = 6;
    root.removeMember(dropped);

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    const std::string payload = Json::writeString(w, root);
    EXPECT_FALSE(parseMigrationRequest(payload).has_value())
        << "Expected rejection when dropping field: " << dropped;
  }
}

TEST(MigrationRequestMessageParse, RejectsNonIntegralField) {
  // Strings where numbers are required must be rejected, not coerced.
  Json::Value root;
  root["migration_id"] = "not-a-number";
  root["src_slot"] = 2;
  root["dst_slot"] = 3;
  root["layer_id"] = 4;
  root["position_start"] = 5;
  root["position_end"] = 6;

  Json::StreamWriterBuilder w;
  w["indentation"] = "";
  EXPECT_FALSE(parseMigrationRequest(Json::writeString(w, root)).has_value());
}

// ---------------------------------------------------------------------------
// MigrationResponseMessage
// ---------------------------------------------------------------------------

class MigrationResponseStatusWire
    : public ::testing::TestWithParam<MigrationStatus> {};

TEST_P(MigrationResponseStatusWire, RoundTripPreservesStatus) {
  const auto in = makeResponse(GetParam());
  const auto out = parseMigrationResponse(serialize(in));

  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->migration_id, in.migration_id);
  EXPECT_EQ(out->status, in.status);
}

INSTANTIATE_TEST_SUITE_P(AllStatuses, MigrationResponseStatusWire,
                         ::testing::Values(MigrationStatus::UNKNOWN,
                                           MigrationStatus::IN_PROGRESS,
                                           MigrationStatus::SUCCESSFUL,
                                           MigrationStatus::FAILED));

TEST(MigrationResponseMessageWire, SerializeEmitsExpectedFields) {
  const auto wire = serialize(makeResponse(MigrationStatus::SUCCESSFUL));
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  ASSERT_TRUE(root.isMember("migration_id"));
  ASSERT_TRUE(root.isMember("status"));
  // status is the human-readable enum name on the wire, not its numeric value.
  ASSERT_TRUE(root["status"].isString());
  EXPECT_EQ(root["status"].asString(), "SUCCESSFUL");
  EXPECT_EQ(root.size(), 2u);
}

TEST(MigrationResponseMessageWire, HandlesMaxUint64MigrationId) {
  MigrationResponseMessage in = makeResponse(MigrationStatus::SUCCESSFUL);
  in.migration_id = std::numeric_limits<uint64_t>::max();

  const auto out = parseMigrationResponse(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->migration_id, std::numeric_limits<uint64_t>::max());
}

TEST(MigrationResponseMessageWire, StatusUsesEnumNameOnTheWire) {
  // Cross-language consumers (Python) read this field as a string; if the C++
  // side ever switches to numeric encoding it'll break them silently.
  const auto inProgressWire =
      parseJson(serialize(makeResponse(MigrationStatus::IN_PROGRESS)));
  EXPECT_EQ(inProgressWire["status"].asString(), "IN_PROGRESS");

  const auto failedWire =
      parseJson(serialize(makeResponse(MigrationStatus::FAILED)));
  EXPECT_EQ(failedWire["status"].asString(), "FAILED");
}

TEST(MigrationResponseMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseMigrationResponse("not json").has_value());
  EXPECT_FALSE(parseMigrationResponse("").has_value());
}

TEST(MigrationResponseMessageParse, RejectsMissingFields) {
  // missing status
  EXPECT_FALSE(parseMigrationResponse(R"({"migration_id": 1})").has_value());
  // missing migration_id
  EXPECT_FALSE(
      parseMigrationResponse(R"({"status": "SUCCESSFUL"})").has_value());
}

TEST(MigrationResponseMessageParse, RejectsWrongFieldTypes) {
  // migration_id must be integral.
  EXPECT_FALSE(
      parseMigrationResponse(R"({"migration_id": "1", "status": "SUCCESSFUL"})")
          .has_value());
  // status must be a string.
  EXPECT_FALSE(parseMigrationResponse(R"({"migration_id": 1, "status": 0})")
                   .has_value());
}

TEST(MigrationResponseMessageParse, RejectsUnknownStatusString) {
  // A status value that doesn't map to any MigrationStatus enumerator must be
  // rejected; silently degrading to UNKNOWN would mask a real protocol drift.
  EXPECT_FALSE(parseMigrationResponse(
                   R"({"migration_id": 1, "status": "MAYBE_SUCCESSFUL"})")
                   .has_value());
}

// ---------------------------------------------------------------------------
// Download / Offload shared fixtures
// ---------------------------------------------------------------------------

struct BlockVectorCase {
  const char* label;
  std::vector<KVCacheBlockRef> blocks;
};

// Prints the label instead of a numeric index in gtest test names, so the
// generated names look like BlockShapes/EmptyBlocks rather than BlockShapes/0.
struct BlockVectorCaseName {
  std::string operator()(
      const ::testing::TestParamInfo<BlockVectorCase>& info) const {
    return info.param.label;
  }
};

std::vector<BlockVectorCase> blockVectorCases() {
  return {
      {"EmptyBlocks", {}},
      {"SingleBlock",
       {KVCacheBlockRef{
           .blockHash = 0xdeadbeef, .positionId = 0, .tokenCount = 128}}},
      {"MultipleContiguousBlocks",
       {
           KVCacheBlockRef{.blockHash = 1, .positionId = 0, .tokenCount = 128},
           KVCacheBlockRef{
               .blockHash = 2, .positionId = 128, .tokenCount = 32},
           KVCacheBlockRef{
               .blockHash = 3, .positionId = 160, .tokenCount = 32},
       }},
      {"MaxValueBlock",
       {KVCacheBlockRef{
           .blockHash = std::numeric_limits<uint64_t>::max(),
           .positionId = std::numeric_limits<uint32_t>::max(),
           .tokenCount = std::numeric_limits<uint32_t>::max()}}},
  };
}

void expectBlocksEqual(const std::vector<KVCacheBlockRef>& expected,
                       const std::vector<KVCacheBlockRef>& actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].blockHash, expected[i].blockHash) << "block " << i;
    EXPECT_EQ(actual[i].positionId, expected[i].positionId) << "block " << i;
    EXPECT_EQ(actual[i].tokenCount, expected[i].tokenCount) << "block " << i;
  }
}

DownloadRequestMessage makeDownloadRequest(
    std::vector<KVCacheBlockRef> blocks) {
  return DownloadRequestMessage{
      .id = 42, .dst_slot = 3, .blocks = std::move(blocks)};
}

OffloadRequestMessage makeOffloadRequest(std::vector<KVCacheBlockRef> blocks) {
  return OffloadRequestMessage{
      .id = 42, .src_slot = 3, .blocks = std::move(blocks)};
}

DownloadResponseMessage makeDownloadResponse(MigrationStatus status) {
  return DownloadResponseMessage{
      .id = 99, .status = status, .usable_prefix_count = 5};
}

OffloadResponseMessage makeOffloadResponse(MigrationStatus status) {
  return OffloadResponseMessage{.id = 99, .status = status};
}

// ---------------------------------------------------------------------------
// DownloadRequestMessage
// ---------------------------------------------------------------------------

class DownloadRequestBlocksWire
    : public ::testing::TestWithParam<BlockVectorCase> {};

TEST_P(DownloadRequestBlocksWire, RoundTripPreservesAllFields) {
  const auto in = makeDownloadRequest(GetParam().blocks);
  const auto wire = serialize(in);

  const auto out = parseDownloadRequest(wire);
  ASSERT_TRUE(out.has_value()) << "parseDownloadRequest rejected: " << wire;

  EXPECT_EQ(out->id, in.id);
  EXPECT_EQ(out->dst_slot, in.dst_slot);
  expectBlocksEqual(in.blocks, out->blocks);
}

INSTANTIATE_TEST_SUITE_P(BlockShapes, DownloadRequestBlocksWire,
                         ::testing::ValuesIn(blockVectorCases()),
                         BlockVectorCaseName{});

TEST(DownloadRequestMessageWire, SerializeEmitsAllExpectedFields) {
  const auto wire = serialize(makeDownloadRequest(
      {KVCacheBlockRef{.blockHash = 7, .positionId = 0, .tokenCount = 128}}));
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("id"));
  EXPECT_TRUE(root.isMember("dst_slot"));
  EXPECT_TRUE(root.isMember("blocks"));
  EXPECT_EQ(root.size(), 3u);

  ASSERT_TRUE(root["blocks"].isArray());
  ASSERT_EQ(root["blocks"].size(), 1u);
  const auto& block = root["blocks"][0];
  EXPECT_TRUE(block.isMember("block_hash"));
  EXPECT_TRUE(block.isMember("position_id"));
  EXPECT_TRUE(block.isMember("token_count"));
}

TEST(DownloadRequestMessageWire, HandlesMaxUint64Id) {
  auto in = makeDownloadRequest({});
  in.id = std::numeric_limits<uint64_t>::max();

  const auto out = parseDownloadRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, std::numeric_limits<uint64_t>::max());
}

TEST(DownloadRequestMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseDownloadRequest("not json").has_value());
  EXPECT_FALSE(parseDownloadRequest("").has_value());
  EXPECT_FALSE(parseDownloadRequest("{ unterminated").has_value());
}

TEST(DownloadRequestMessageParse, RejectsMissingRequiredField) {
  for (const char* dropped : {"id", "dst_slot", "blocks"}) {
    Json::Value root;
    root["id"] = 1;
    root["dst_slot"] = 2;
    root["blocks"] = Json::arrayValue;
    root.removeMember(dropped);

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    EXPECT_FALSE(parseDownloadRequest(Json::writeString(w, root)).has_value())
        << "Expected rejection when dropping field: " << dropped;
  }
}

TEST(DownloadRequestMessageParse, RejectsWrongFieldTypes) {
  EXPECT_FALSE(
      parseDownloadRequest(R"({"id": "x", "dst_slot": 1, "blocks": []})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadRequest(R"({"id": 1, "dst_slot": "x", "blocks": []})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadRequest(R"({"id": 1, "dst_slot": 1, "blocks": {}})")
          .has_value());
}

// ---------------------------------------------------------------------------
// DownloadResponseMessage
// ---------------------------------------------------------------------------

class DownloadResponseStatusWire
    : public ::testing::TestWithParam<MigrationStatus> {};

TEST_P(DownloadResponseStatusWire, RoundTripPreservesAllFields) {
  const auto in = makeDownloadResponse(GetParam());
  const auto out = parseDownloadResponse(serialize(in));

  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, in.id);
  EXPECT_EQ(out->status, in.status);
  EXPECT_EQ(out->usable_prefix_count, in.usable_prefix_count);
}

INSTANTIATE_TEST_SUITE_P(AllStatuses, DownloadResponseStatusWire,
                         ::testing::Values(MigrationStatus::UNKNOWN,
                                           MigrationStatus::IN_PROGRESS,
                                           MigrationStatus::SUCCESSFUL,
                                           MigrationStatus::FAILED));

TEST(DownloadResponseMessageWire, SerializeEmitsExpectedFields) {
  const auto wire =
      serialize(makeDownloadResponse(MigrationStatus::SUCCESSFUL));
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("id"));
  EXPECT_TRUE(root.isMember("status"));
  EXPECT_TRUE(root.isMember("usable_prefix_count"));
  ASSERT_TRUE(root["status"].isString());
  EXPECT_EQ(root["status"].asString(), "SUCCESSFUL");
  EXPECT_EQ(root.size(), 3u);
}

TEST(DownloadResponseMessageWire, HandlesMaxValues) {
  auto in = makeDownloadResponse(MigrationStatus::SUCCESSFUL);
  in.id = std::numeric_limits<uint64_t>::max();
  in.usable_prefix_count = std::numeric_limits<uint32_t>::max();

  const auto out = parseDownloadResponse(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(out->usable_prefix_count, std::numeric_limits<uint32_t>::max());
}

TEST(DownloadResponseMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseDownloadResponse("not json").has_value());
  EXPECT_FALSE(parseDownloadResponse("").has_value());
}

TEST(DownloadResponseMessageParse, RejectsMissingFields) {
  EXPECT_FALSE(
      parseDownloadResponse(
          R"({"status": "SUCCESSFUL", "usable_prefix_count": 0})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadResponse(R"({"id": 1, "usable_prefix_count": 0})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadResponse(R"({"id": 1, "status": "SUCCESSFUL"})").has_value());
}

TEST(DownloadResponseMessageParse, RejectsWrongFieldTypes) {
  EXPECT_FALSE(
      parseDownloadResponse(
          R"({"id": "x", "status": "SUCCESSFUL", "usable_prefix_count": 0})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadResponse(
          R"({"id": 1, "status": 0, "usable_prefix_count": 0})")
          .has_value());
  EXPECT_FALSE(
      parseDownloadResponse(
          R"({"id": 1, "status": "SUCCESSFUL", "usable_prefix_count": "x"})")
          .has_value());
}

TEST(DownloadResponseMessageParse, RejectsUnknownStatusString) {
  EXPECT_FALSE(
      parseDownloadResponse(
          R"({"id": 1, "status": "MAYBE_SUCCESSFUL", "usable_prefix_count": 0})")
          .has_value());
}

// ---------------------------------------------------------------------------
// OffloadRequestMessage
// ---------------------------------------------------------------------------

class OffloadRequestBlocksWire
    : public ::testing::TestWithParam<BlockVectorCase> {};

TEST_P(OffloadRequestBlocksWire, RoundTripPreservesAllFields) {
  const auto in = makeOffloadRequest(GetParam().blocks);
  const auto wire = serialize(in);

  const auto out = parseOffloadRequest(wire);
  ASSERT_TRUE(out.has_value()) << "parseOffloadRequest rejected: " << wire;

  EXPECT_EQ(out->id, in.id);
  EXPECT_EQ(out->src_slot, in.src_slot);
  expectBlocksEqual(in.blocks, out->blocks);
}

INSTANTIATE_TEST_SUITE_P(BlockShapes, OffloadRequestBlocksWire,
                         ::testing::ValuesIn(blockVectorCases()),
                         BlockVectorCaseName{});

TEST(OffloadRequestMessageWire, SerializeEmitsAllExpectedFields) {
  const auto wire = serialize(makeOffloadRequest(
      {KVCacheBlockRef{.blockHash = 7, .positionId = 0, .tokenCount = 128}}));
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("id"));
  EXPECT_TRUE(root.isMember("src_slot"));
  EXPECT_TRUE(root.isMember("blocks"));
  EXPECT_EQ(root.size(), 3u);

  ASSERT_TRUE(root["blocks"].isArray());
  ASSERT_EQ(root["blocks"].size(), 1u);
  const auto& block = root["blocks"][0];
  EXPECT_TRUE(block.isMember("block_hash"));
  EXPECT_TRUE(block.isMember("position_id"));
  EXPECT_TRUE(block.isMember("token_count"));
}

TEST(OffloadRequestMessageWire, HandlesMaxUint64Id) {
  auto in = makeOffloadRequest({});
  in.id = std::numeric_limits<uint64_t>::max();

  const auto out = parseOffloadRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, std::numeric_limits<uint64_t>::max());
}

TEST(OffloadRequestMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseOffloadRequest("not json").has_value());
  EXPECT_FALSE(parseOffloadRequest("").has_value());
  EXPECT_FALSE(parseOffloadRequest("{ unterminated").has_value());
}

TEST(OffloadRequestMessageParse, RejectsMissingRequiredField) {
  for (const char* dropped : {"id", "src_slot", "blocks"}) {
    Json::Value root;
    root["id"] = 1;
    root["src_slot"] = 2;
    root["blocks"] = Json::arrayValue;
    root.removeMember(dropped);

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    EXPECT_FALSE(parseOffloadRequest(Json::writeString(w, root)).has_value())
        << "Expected rejection when dropping field: " << dropped;
  }
}

TEST(OffloadRequestMessageParse, RejectsWrongFieldTypes) {
  EXPECT_FALSE(
      parseOffloadRequest(R"({"id": "x", "src_slot": 1, "blocks": []})")
          .has_value());
  EXPECT_FALSE(
      parseOffloadRequest(R"({"id": 1, "src_slot": "x", "blocks": []})")
          .has_value());
  EXPECT_FALSE(
      parseOffloadRequest(R"({"id": 1, "src_slot": 1, "blocks": {}})")
          .has_value());
}

// ---------------------------------------------------------------------------
// OffloadResponseMessage
// ---------------------------------------------------------------------------

class OffloadResponseStatusWire
    : public ::testing::TestWithParam<MigrationStatus> {};

TEST_P(OffloadResponseStatusWire, RoundTripPreservesAllFields) {
  const auto in = makeOffloadResponse(GetParam());
  const auto out = parseOffloadResponse(serialize(in));

  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, in.id);
  EXPECT_EQ(out->status, in.status);
}

INSTANTIATE_TEST_SUITE_P(AllStatuses, OffloadResponseStatusWire,
                         ::testing::Values(MigrationStatus::UNKNOWN,
                                           MigrationStatus::IN_PROGRESS,
                                           MigrationStatus::SUCCESSFUL,
                                           MigrationStatus::FAILED));

TEST(OffloadResponseMessageWire, SerializeEmitsExpectedFields) {
  const auto wire = serialize(makeOffloadResponse(MigrationStatus::SUCCESSFUL));
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("id"));
  EXPECT_TRUE(root.isMember("status"));
  ASSERT_TRUE(root["status"].isString());
  EXPECT_EQ(root["status"].asString(), "SUCCESSFUL");
  EXPECT_EQ(root.size(), 2u);
}

TEST(OffloadResponseMessageWire, HandlesMaxUint64Id) {
  auto in = makeOffloadResponse(MigrationStatus::SUCCESSFUL);
  in.id = std::numeric_limits<uint64_t>::max();

  const auto out = parseOffloadResponse(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->id, std::numeric_limits<uint64_t>::max());
}

TEST(OffloadResponseMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseOffloadResponse("not json").has_value());
  EXPECT_FALSE(parseOffloadResponse("").has_value());
}

TEST(OffloadResponseMessageParse, RejectsMissingFields) {
  EXPECT_FALSE(parseOffloadResponse(R"({"id": 1})").has_value());
  EXPECT_FALSE(parseOffloadResponse(R"({"status": "SUCCESSFUL"})").has_value());
}

TEST(OffloadResponseMessageParse, RejectsWrongFieldTypes) {
  EXPECT_FALSE(parseOffloadResponse(R"({"id": "x", "status": "SUCCESSFUL"})")
                   .has_value());
  EXPECT_FALSE(
      parseOffloadResponse(R"({"id": 1, "status": 0})").has_value());
}

TEST(OffloadResponseMessageParse, RejectsUnknownStatusString) {
  EXPECT_FALSE(
      parseOffloadResponse(R"({"id": 1, "status": "MAYBE_SUCCESSFUL"})")
          .has_value());
}

}  // namespace
}  // namespace tt::messaging
