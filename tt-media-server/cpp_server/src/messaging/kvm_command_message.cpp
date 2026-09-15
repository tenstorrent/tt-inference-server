// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/kvm_command_message.hpp"

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

// kv_manager keys the migration wire with these. Do not rename — the
// contract lives in `tt-d-gen/kv_manager/src/control_plane/command/types.cpp`.
constexpr const char* K_COMMAND_ID = "command_id";
constexpr const char* K_MIGRATION_ID = "migration_id";
constexpr const char* K_KIND = "kind";
constexpr const char* K_KIND_MIGRATE = "migrate";

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
    TT_LOG_ERROR("[kvm_command_message] JSON parse failed: {}", errs);
    return false;
  }
  return true;
}

std::string write(const Json::Value& root) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, root);
}

}  // namespace

std::string serialize(const KvmCommandMessage& msg) {
  Json::Value root;
  root[K_COMMAND_ID] = static_cast<Json::UInt64>(msg.command_id);
  root[K_MIGRATION_ID] = static_cast<Json::UInt64>(msg.migration_id);
  root["src_slot"] = msg.src_slot;
  root["dst_slot"] = msg.dst_slot;
  root["layer_begin"] = msg.layer_begin;
  root["layer_end"] = msg.layer_end;
  root["src_position_begin"] = msg.src_position_begin;
  root["src_position_end"] = msg.src_position_end;
  root["dst_position_begin"] = msg.dst_position_begin;
  root["dst_position_end"] = msg.dst_position_end;
  // Always MIGRATE — SLOT_COPY / DRAIN belong to the loopback path handled
  // by the composite client, not this transport.
  root[K_KIND] = K_KIND_MIGRATE;
  return write(root);
}

std::string serialize(const KvmResponseMessage& msg) {
  Json::Value root;
  root[K_COMMAND_ID] = static_cast<Json::UInt64>(msg.command_id);
  root[K_MIGRATION_ID] = static_cast<Json::UInt64>(msg.migration_id);
  root["status"] = std::string(toWire(msg.status));
  return write(root);
}

std::optional<KvmCommandMessage> parseKvmCommand(const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;

  for (const char* field :
       {K_MIGRATION_ID, "src_slot", "dst_slot", "layer_begin", "layer_end",
        "src_position_begin", "src_position_end", "dst_position_begin",
        "dst_position_end"}) {
    if (!root.isMember(field) || !root[field].isIntegral()) {
      TT_LOG_ERROR("[kvm_command_message] Request missing/non-integral: {}",
                   field);
      return std::nullopt;
    }
  }

  KvmCommandMessage out{};
  // command_id is nominally required for us, but kv_manager treats it as
  // optional (defaults to 0). Match that so we can round-trip legacy or
  // hand-crafted messages in tests.
  out.command_id =
      root.isMember(K_COMMAND_ID) ? root[K_COMMAND_ID].asUInt64() : 0;
  out.migration_id = root[K_MIGRATION_ID].asUInt64();
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

std::optional<KvmResponseMessage> parseKvmResponse(const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;

  if (!root.isMember(K_MIGRATION_ID) || !root[K_MIGRATION_ID].isIntegral() ||
      !root.isMember("status") || !root["status"].isString()) {
    TT_LOG_ERROR("[kvm_command_message] Response missing required fields");
    return std::nullopt;
  }

  auto status = fromWire(root["status"].asString());
  if (!status.has_value()) {
    TT_LOG_ERROR("[kvm_command_message] Unknown status: {}",
                 root["status"].asString());
    return std::nullopt;
  }

  KvmResponseMessage out{};
  out.command_id =
      root.isMember(K_COMMAND_ID) ? root[K_COMMAND_ID].asUInt64() : 0;
  out.migration_id = root[K_MIGRATION_ID].asUInt64();
  out.status = *status;
  return out;
}

}  // namespace tt::messaging
