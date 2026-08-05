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
      .kafka_request_id = 42,
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

MigrationResponseMessage makeResponse(MigrationStatus status) {
  return MigrationResponseMessage{
      .kafka_request_id = 99, .migration_id = 1001, .status = status};
}

TEST(MigrationRequestMessageWire, RoundTripPreservesAllFields) {
  const auto in = makeRequest();
  const std::string wire = serialize(in);

  const auto out = parseMigrationRequest(wire);
  ASSERT_TRUE(out.has_value()) << "parseMigrationRequest rejected: " << wire;

  EXPECT_EQ(out->kafka_request_id, in.kafka_request_id);
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
  const auto wire = serialize(makeRequest());
  const auto root = parseJson(wire);

  ASSERT_TRUE(root.isObject());
  EXPECT_TRUE(root.isMember("kafka_request_id"));
  EXPECT_TRUE(root.isMember("migration_id"));
  EXPECT_TRUE(root.isMember("src_slot"));
  EXPECT_TRUE(root.isMember("dst_slot"));
  EXPECT_TRUE(root.isMember("layer_begin"));
  EXPECT_TRUE(root.isMember("layer_end"));
  EXPECT_TRUE(root.isMember("src_position_begin"));
  EXPECT_TRUE(root.isMember("src_position_end"));
  EXPECT_TRUE(root.isMember("dst_position_begin"));
  EXPECT_TRUE(root.isMember("dst_position_end"));
  EXPECT_FALSE(root.isMember("migration_id"));
  EXPECT_EQ(root.size(), 10u);
}

TEST(MigrationRequestMessageWire, OmitsMigrationIdWhenUnset) {
  MigrationRequestMessage in = makeRequest();
  in.migration_id = std::nullopt;
  const auto root = parseJson(serialize(in));
  EXPECT_FALSE(root.isMember("migration_id"));
  EXPECT_EQ(root.size(), 9u);
}

TEST(MigrationRequestMessageWire, HandlesMaxUint64KafkaRequestId) {
  MigrationRequestMessage in = makeRequest();
  in.kafka_request_id = std::numeric_limits<uint64_t>::max();

  const auto out = parseMigrationRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, std::numeric_limits<uint64_t>::max());
}

TEST(MigrationRequestMessageWire, HandlesMaxUint32Slots) {
  MigrationRequestMessage in = makeRequest();
  const auto kMax = std::numeric_limits<uint32_t>::max();
  in.src_slot = kMax;
  in.dst_slot = kMax;
  in.layer_begin = kMax;
  in.layer_end = kMax;
  in.src_position_begin = kMax;
  in.src_position_end = kMax;
  in.dst_position_begin = kMax;
  in.dst_position_end = kMax;

  const auto out = parseMigrationRequest(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->src_slot, kMax);
  EXPECT_EQ(out->dst_slot, kMax);
  EXPECT_EQ(out->layer_begin, kMax);
  EXPECT_EQ(out->layer_end, kMax);
  EXPECT_EQ(out->src_position_begin, kMax);
  EXPECT_EQ(out->src_position_end, kMax);
  EXPECT_EQ(out->dst_position_begin, kMax);
  EXPECT_EQ(out->dst_position_end, kMax);
}

TEST(MigrationRequestMessageParse, AcceptsLegacyMigrationIdAsKafkaRequestId) {
  // Old producers only emitted migration_id, and it was the per-request id.
  Json::Value root;
  root["migration_id"] = 77;
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
  const auto out = parseMigrationRequest(Json::writeString(w, root));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, 77u);
  EXPECT_FALSE(out->migration_id.has_value());
}

TEST(MigrationRequestMessageParse, DualFieldsKeepMigrationIdAsParent) {
  Json::Value root;
  root["kafka_request_id"] = 42;
  root["migration_id"] = 1001;
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
  const auto out = parseMigrationRequest(Json::writeString(w, root));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, 42u);
  ASSERT_TRUE(out->migration_id.has_value());
  EXPECT_EQ(*out->migration_id, 1001u);
}

TEST(MigrationRequestMessageParse, RejectsMalformedJson) {
  EXPECT_FALSE(parseMigrationRequest("not json at all").has_value());
  EXPECT_FALSE(parseMigrationRequest("").has_value());
  EXPECT_FALSE(parseMigrationRequest("{ unterminated").has_value());
}

TEST(MigrationRequestMessageParse, RejectsMissingRequiredField) {
  for (const char* dropped :
       {"kafka_request_id", "src_slot", "dst_slot", "layer_begin", "layer_end",
        "src_position_begin", "src_position_end", "dst_position_begin",
        "dst_position_end"}) {
    Json::Value root;
    root["kafka_request_id"] = 1;
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
  Json::Value root;
  root["kafka_request_id"] = "not-a-number";
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

class MigrationResponseStatusWire
    : public ::testing::TestWithParam<MigrationStatus> {};

TEST_P(MigrationResponseStatusWire, RoundTripPreservesStatus) {
  const auto in = makeResponse(GetParam());
  const auto out = parseMigrationResponse(serialize(in));

  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, in.kafka_request_id);
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
  ASSERT_TRUE(root.isMember("kafka_request_id"));
  ASSERT_TRUE(root.isMember("migration_id"));
  ASSERT_TRUE(root.isMember("status"));
  ASSERT_TRUE(root["status"].isString());
  EXPECT_EQ(root["status"].asString(), "SUCCESSFUL");
  EXPECT_EQ(root.size(), 3u);
}

TEST(MigrationResponseMessageWire, HandlesMaxUint64KafkaRequestId) {
  MigrationResponseMessage in = makeResponse(MigrationStatus::SUCCESSFUL);
  in.kafka_request_id = std::numeric_limits<uint64_t>::max();

  const auto out = parseMigrationResponse(serialize(in));
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, std::numeric_limits<uint64_t>::max());
}

TEST(MigrationResponseMessageWire, StatusUsesEnumNameOnTheWire) {
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
  EXPECT_FALSE(
      parseMigrationResponse(R"({"kafka_request_id": 1})").has_value());
  EXPECT_FALSE(
      parseMigrationResponse(R"({"status": "SUCCESSFUL"})").has_value());
}

TEST(MigrationResponseMessageParse, AcceptsLegacyMigrationIdKey) {
  const auto out =
      parseMigrationResponse(R"({"migration_id": 55, "status": "SUCCESSFUL"})");
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->kafka_request_id, 55u);
  EXPECT_FALSE(out->migration_id.has_value());
}

TEST(MigrationResponseMessageParse, RejectsWrongFieldTypes) {
  EXPECT_FALSE(parseMigrationResponse(
                   R"({"kafka_request_id": "1", "status": "SUCCESSFUL"})")
                   .has_value());
  EXPECT_FALSE(parseMigrationResponse(R"({"kafka_request_id": 1, "status": 0})")
                   .has_value());
}

TEST(MigrationResponseMessageParse, RejectsUnknownStatusString) {
  EXPECT_FALSE(parseMigrationResponse(
                   R"({"kafka_request_id": 1, "status": "MAYBE_SUCCESSFUL"})")
                   .has_value());
}

}  // namespace
}  // namespace tt::messaging
