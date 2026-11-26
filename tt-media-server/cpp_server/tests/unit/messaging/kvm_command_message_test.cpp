// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/kvm_command_message.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace tt::messaging {
namespace {

using tt::services::MigrationStatus;

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

KvmCommandMessage makeCommand() {
  return KvmCommandMessage{
      .command_id = 42,
      .migration_id = 1001,
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

KvmResponseMessage makeResponse(MigrationStatus status) {
  return KvmResponseMessage{
      .command_id = 99, .migration_id = 1001, .status = status};
}

TEST(KvmCommandMessageWire, RoundTripPreservesAllFields) {
  const auto in = makeCommand();
  const auto out = parseKvmCommand(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->command_id, in.command_id);
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

TEST(KvmCommandMessageWire, SerializeMatchesKvManagerSchema) {
  // kv_manager's parser insists on `migration_id`, geometry fields, and
  // accepts `kind` as a string. Our wire must include all of them so a
  // stock kv_manager can parse it byte-for-byte.
  const auto root = parseJson(serialize(makeCommand()));
  ASSERT_TRUE(root.isObject());
  for (const char* key :
       {"command_id", "migration_id", "src_slot", "dst_slot", "layer_begin",
        "layer_end", "src_position_begin", "src_position_end",
        "dst_position_begin", "dst_position_end", "kind"}) {
    EXPECT_TRUE(root.isMember(key)) << "missing key: " << key;
  }
  EXPECT_EQ(root["kind"].asString(), "migrate");
}

TEST(KvmCommandMessageWire, HandlesMaxUint64Ids) {
  auto in = makeCommand();
  in.command_id = std::numeric_limits<uint64_t>::max();
  in.migration_id = std::numeric_limits<uint64_t>::max();
  const auto out = parseKvmCommand(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->command_id, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(out->migration_id, std::numeric_limits<uint64_t>::max());
}

TEST(KvmCommandMessageWire, HandlesMaxUint32Geometry) {
  auto in = makeCommand();
  const auto kMax = std::numeric_limits<uint32_t>::max();
  in.src_slot = kMax;
  in.dst_slot = kMax;
  in.layer_begin = kMax;
  in.layer_end = kMax;
  in.src_position_begin = kMax;
  in.src_position_end = kMax;
  in.dst_position_begin = kMax;
  in.dst_position_end = kMax;
  const auto out = parseKvmCommand(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->src_slot, kMax);
  EXPECT_EQ(out->layer_end, kMax);
  EXPECT_EQ(out->dst_position_end, kMax);
}

TEST(KvmCommandMessageParse, DefaultsMissingCommandIdToZero) {
  // kv_manager treats command_id as optional (defaults to 0); mirror that
  // so a hand-crafted kv_manager reply we ever see the other way round
  // still parses.
  Json::Value root;
  root["migration_id"] = 55;
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
  const auto out = parseKvmCommand(Json::writeString(w, root));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->command_id, 0u);
  EXPECT_EQ(out->migration_id, 55u);
}

TEST(KvmCommandMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseKvmCommand("not json at all").has_value());
  EXPECT_FALSE(parseKvmCommand("").has_value());
  EXPECT_FALSE(parseKvmCommand("{ unterminated").has_value());
}

TEST(KvmCommandMessageParse, RejectsMissingRequiredField) {
  for (const char* dropped :
       {"migration_id", "src_slot", "dst_slot", "layer_begin", "layer_end",
        "src_position_begin", "src_position_end", "dst_position_begin",
        "dst_position_end"}) {
    Json::Value root;
    root["command_id"] = 1;
    root["migration_id"] = 2;
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
    EXPECT_FALSE(parseKvmCommand(Json::writeString(w, root)).has_value())
        << "Expected rejection when dropping: " << dropped;
  }
}

class KvmResponseStatusWire : public ::testing::TestWithParam<MigrationStatus> {
};

TEST_P(KvmResponseStatusWire, RoundTripPreservesStatus) {
  const auto in = makeResponse(GetParam());
  const auto out = parseKvmResponse(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->command_id, in.command_id);
  EXPECT_EQ(out->migration_id, in.migration_id);
  EXPECT_EQ(out->status, in.status);
}

INSTANTIATE_TEST_SUITE_P(AllStatuses, KvmResponseStatusWire,
                         ::testing::Values(MigrationStatus::UNKNOWN,
                                           MigrationStatus::IN_PROGRESS,
                                           MigrationStatus::SUCCESSFUL,
                                           MigrationStatus::FAILED));

TEST(KvmResponseMessageWire, SerializeEmitsExpectedFields) {
  const auto root =
      parseJson(serialize(makeResponse(MigrationStatus::SUCCESSFUL)));
  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("command_id"));
  EXPECT_TRUE(root.isMember("migration_id"));
  EXPECT_TRUE(root.isMember("status"));
  EXPECT_EQ(root["status"].asString(), "SUCCESSFUL");
  EXPECT_EQ(root.size(), 3u);
}

TEST(KvmResponseMessageParse, RejectsMissingFields) {
  EXPECT_FALSE(parseKvmResponse(R"({"command_id": 1})").has_value());
  EXPECT_FALSE(parseKvmResponse(R"({"status": "SUCCESSFUL"})").has_value());
  EXPECT_FALSE(
      parseKvmResponse(R"({"command_id": 1, "migration_id": 2})").has_value());
}

TEST(KvmResponseMessageParse, RejectsUnknownStatus) {
  EXPECT_FALSE(parseKvmResponse(
                   R"({"command_id": 1, "migration_id": 2, "status": "MAYBE"})")
                   .has_value());
}

}  // namespace
}  // namespace tt::messaging
