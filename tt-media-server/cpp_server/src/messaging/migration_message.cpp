// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/migration_message.hpp"

#include <json/json.h>
#include <json/value.h>

#include <sstream>
#include <string_view>

#include "utils/logger.hpp"

namespace tt::messaging {

namespace {

constexpr std::string_view K_STATUS_UNKNOWN = "UNKNOWN";
constexpr std::string_view K_STATUS_IN_PROGRESS = "IN_PROGRESS";
constexpr std::string_view K_STATUS_SUCCESSFUL = "SUCCESSFUL";
constexpr std::string_view K_STATUS_FAILED = "FAILED";

constexpr const char* K_KAFKA_REQUEST_ID = "kafka_request_id";
constexpr const char* K_MIGRATION_ID = "migration_id";

std::string_view toWire(tt::services::MigrationStatus status) {
  using Status = tt::services::MigrationStatus;
  switch (status) {
    case Status::UNKNOWN:
      return K_STATUS_UNKNOWN;
    case Status::IN_PROGRESS:
      return K_STATUS_IN_PROGRESS;
    case Status::SUCCESSFUL:
      return K_STATUS_SUCCESSFUL;
    case Status::FAILED:
      return K_STATUS_FAILED;
    default:
      return K_STATUS_UNKNOWN;
  }
}

std::optional<tt::services::MigrationStatus> fromWire(std::string_view status) {
  using Status = tt::services::MigrationStatus;
  if (status == K_STATUS_UNKNOWN) return Status::UNKNOWN;
  if (status == K_STATUS_IN_PROGRESS) return Status::IN_PROGRESS;
  if (status == K_STATUS_SUCCESSFUL) return Status::SUCCESSFUL;
  if (status == K_STATUS_FAILED) return Status::FAILED;

  return std::nullopt;
}

bool parse(std::string_view payload, Json::Value& root) {
  Json::CharReaderBuilder builder;
  std::istringstream iss{std::string(payload)};
  std::string errs;

  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    TT_LOG_ERROR("[migration_message] JSON parse failed: {}", errs);
    return false;
  }

  return true;
}

std::string write(const Json::Value& root) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, root);
}

// New wire: kafka_request_id is the per-request id.
// Legacy wire: only migration_id existed, and it was the per-request id.
std::optional<uint64_t> parseKafkaRequestId(const Json::Value& root) {
  if (root.isMember(K_KAFKA_REQUEST_ID) &&
      root[K_KAFKA_REQUEST_ID].isIntegral()) {
    return root[K_KAFKA_REQUEST_ID].asUInt64();
  }
  if (!root.isMember(K_KAFKA_REQUEST_ID) && root.isMember(K_MIGRATION_ID) &&
      root[K_MIGRATION_ID].isIntegral()) {
    return root[K_MIGRATION_ID].asUInt64();
  }
  return std::nullopt;
}

// Parent burst id is migration_id only in the new dual-field format.
std::optional<uint64_t> parseParentMigrationId(const Json::Value& root) {
  if (!root.isMember(K_KAFKA_REQUEST_ID) || !root.isMember(K_MIGRATION_ID)) {
    return std::nullopt;
  }
  if (!root[K_MIGRATION_ID].isIntegral()) {
    TT_LOG_ERROR("[migration_message] non-integral field: migration_id");
    return std::nullopt;
  }
  return root[K_MIGRATION_ID].asUInt64();
}

}  // namespace

std::string serialize(const MigrationRequestMessage& mrm) {
  Json::Value root;
  root[K_KAFKA_REQUEST_ID] = static_cast<Json::UInt64>(mrm.kafka_request_id);
  if (mrm.migration_id.has_value()) {
    root[K_MIGRATION_ID] = static_cast<Json::UInt64>(*mrm.migration_id);
  }
  root["src_slot"] = mrm.src_slot;
  root["dst_slot"] = mrm.dst_slot;
  root["layer_begin"] = mrm.layer_begin;
  root["layer_end"] = mrm.layer_end;
  root["src_position_begin"] = mrm.src_position_begin;
  root["src_position_end"] = mrm.src_position_end;
  root["dst_position_begin"] = mrm.dst_position_begin;
  root["dst_position_end"] = mrm.dst_position_end;

  return write(root);
}

std::string serialize(const MigrationResponseMessage& mrm) {
  Json::Value root;
  root[K_KAFKA_REQUEST_ID] = static_cast<Json::UInt64>(mrm.kafka_request_id);
  if (mrm.migration_id.has_value()) {
    root[K_MIGRATION_ID] = static_cast<Json::UInt64>(*mrm.migration_id);
  }
  root["status"] = std::string(toWire(mrm.status));

  return write(root);
}

std::optional<MigrationRequestMessage> parseMigrationRequest(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;

  const auto kafkaRequestId = parseKafkaRequestId(root);
  if (!kafkaRequestId.has_value()) {
    TT_LOG_ERROR(
        "[migration_message] Request missing/non-integral: kafka_request_id "
        "(or legacy migration_id)");
    return std::nullopt;
  }

  if (root.isMember(K_KAFKA_REQUEST_ID) && root.isMember(K_MIGRATION_ID) &&
      !root[K_MIGRATION_ID].isIntegral()) {
    TT_LOG_ERROR("[migration_message] Request non-integral: migration_id");
    return std::nullopt;
  }

  for (const char* field :
       {"src_slot", "dst_slot", "layer_begin", "layer_end",
        "src_position_begin", "src_position_end", "dst_position_begin",
        "dst_position_end"}) {
    if (!root.isMember(field) || !root[field].isIntegral()) {
      TT_LOG_ERROR("[migration_message] Request missing/non-integral: {}",
                   field);
      return std::nullopt;
    }
  }

  MigrationRequestMessage out{};
  out.kafka_request_id = *kafkaRequestId;
  out.migration_id = parseParentMigrationId(root);
  out.src_slot = root["src_slot"].asUInt();
  out.dst_slot = root["dst_slot"].asUInt();
  out.layer_begin = root["layer_begin"].asUInt();
  out.layer_end = root["layer_end"].asUInt();
  out.src_position_begin = root["src_position_begin"].asUInt();
  out.src_position_end = root["src_position_end"].asUInt();
  out.dst_position_begin = root["dst_position_begin"].asUInt();
  out.dst_position_end = root["dst_position_end"].asUInt();

  return out;
}

std::optional<MigrationResponseMessage> parseMigrationResponse(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;

  const auto kafkaRequestId = parseKafkaRequestId(root);
  if (!kafkaRequestId.has_value() || !root.isMember("status") ||
      !root["status"].isString()) {
    TT_LOG_ERROR("[migration_message] Response missing required fields");
    return std::nullopt;
  }

  if (root.isMember(K_KAFKA_REQUEST_ID) && root.isMember(K_MIGRATION_ID) &&
      !root[K_MIGRATION_ID].isIntegral()) {
    TT_LOG_ERROR("[migration_message] Response non-integral: migration_id");
    return std::nullopt;
  }

  auto status = fromWire(root["status"].asString());
  if (!status.has_value()) {
    TT_LOG_ERROR("[migration_message] Unknown status string: {}",
                 root["status"].asString());
    return std::nullopt;
  }

  MigrationResponseMessage out{};
  out.kafka_request_id = *kafkaRequestId;
  out.migration_id = parseParentMigrationId(root);
  out.status = *status;

  return out;
}

}  // namespace tt::messaging
