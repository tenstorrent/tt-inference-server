// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/migration_message.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace tt::messaging {
namespace {

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
      .layer_begin = 0,
      .layer_end = 32,
      .src_position_begin = 100,
      .src_position_end = 200,
      .dst_position_begin = 100,
      .dst_position_end = 200,
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
  EXPECT_EQ(out->layer_begin, in.layer_begin);
  EXPECT_EQ(out->layer_end, in.layer_end);
  EXPECT_EQ(out->src_position_begin, in.src_position_begin);
  EXPECT_EQ(out->src_position_end, in.src_position_end);
  EXPECT_EQ(out->dst_position_begin, in.dst_position_begin);
  EXPECT_EQ(out->dst_position_end, in.dst_position_end);
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
  EXPECT_TRUE(root.isMember("layer_begin"));
  EXPECT_TRUE(root.isMember("layer_end"));
  EXPECT_TRUE(root.isMember("src_position_begin"));
  EXPECT_TRUE(root.isMember("src_position_end"));
  EXPECT_TRUE(root.isMember("dst_position_begin"));
  EXPECT_TRUE(root.isMember("dst_position_end"));
  EXPECT_EQ(root.size(), 9u);
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
  in.layer_begin = std::numeric_limits<uint32_t>::max();
  in.layer_end = std::numeric_limits<uint32_t>::max();
  in.src_position_begin = std::numeric_limits<uint32_t>::max();
  in.src_position_end = std::numeric_limits<uint32_t>::max();
  in.dst_position_begin = std::numeric_limits<uint32_t>::max();
  in.dst_position_end = std::numeric_limits<uint32_t>::max();

  const auto out = parseMigrationRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->src_slot, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->dst_slot, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->layer_begin, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->layer_end, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->src_position_begin, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->src_position_end, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->dst_position_begin, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(out->dst_position_end, std::numeric_limits<uint32_t>::max());
}

TEST(MigrationRequestMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseMigrationRequest("not json at all").has_value());
  EXPECT_FALSE(parseMigrationRequest("").has_value());
  EXPECT_FALSE(parseMigrationRequest("{ unterminated").has_value());
}

TEST(MigrationRequestMessageParse, RejectsMissingRequiredField) {
  // Every required field, dropped one at a time.
  for (const char* dropped :
       {"migration_id", "src_slot", "dst_slot", "layer_begin", "layer_end",
        "src_position_begin", "src_position_end", "dst_position_begin",
        "dst_position_end"}) {
    Json::Value root;
    root["migration_id"] = 1;
    root["src_slot"] = 2;
    root["dst_slot"] = 3;
    root["layer_begin"] = 0;
    root["layer_end"] = 32;
    root["src_position_begin"] = 100;
    root["src_position_end"] = 200;
    root["dst_position_begin"] = 100;
    root["dst_position_end"] = 200;
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
  root["layer_begin"] = 0;
  root["layer_end"] = 32;
  root["src_position_begin"] = 100;
  root["src_position_end"] = 200;
  root["dst_position_begin"] = 100;
  root["dst_position_end"] = 200;

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

}  // namespace
}  // namespace tt::messaging
