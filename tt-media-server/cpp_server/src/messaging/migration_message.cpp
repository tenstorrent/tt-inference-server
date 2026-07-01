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
}  // namespace

std::string serialize(const MigrationRequestMessage& mrm) {
  Json::Value root;
  root["migration_id"] = static_cast<Json::UInt64>(mrm.migration_id);
  root["src_slot"] = mrm.src_slot;
  root["dst_slot"] = mrm.dst_slot;
  root["layer_id"] = mrm.layer_id;
  root["position_start"] = mrm.position_start;
  root["position_end"] = mrm.position_end;

  return write(root);
}

std::string serialize(const MigrationResponseMessage& mrm) {
  Json::Value root;
  root["migration_id"] = static_cast<Json::UInt64>(mrm.migration_id);
  root["status"] = std::string(toWire(mrm.status));

  return write(root);
}

std::string serialize(const DownloadRequestMessage& drm) {
  Json::Value root;
  root["id"] = static_cast<Json::UInt64>(drm.id);
  root["dst_slot"] = drm.dst_slot;
  root["blocks"] = Json::arrayValue;
  for (const auto& block : drm.blocks) {
    Json::Value blockRoot;
    blockRoot["block_hash"] = static_cast<Json::UInt64>(block.blockHash);
    blockRoot["position_id"] = block.positionId;
    blockRoot["token_count"] = static_cast<Json::UInt>(block.tokenCount);
    root["blocks"].append(blockRoot);
  }

  return write(root);
}

std::string serialize(const DownloadResponseMessage& drm) {
  Json::Value root;
  root["id"] = static_cast<Json::UInt64>(drm.id);
  root["status"] = std::string(toWire(drm.status));
  root["downloaded_block_hashes"] = Json::arrayValue;
  for (const auto& hash : drm.downloaded_block_hashes) {
    root["downloaded_block_hashes"].append(static_cast<Json::UInt64>(hash));
  }

  return write(root);
}

std::string serialize(const OffloadRequestMessage& orm) {
  Json::Value root;
  root["id"] = static_cast<Json::UInt64>(orm.id);
  root["src_slot"] = orm.src_slot;
  root["blocks"] = Json::arrayValue;
  for (const auto& block : orm.blocks) {
    Json::Value blockRoot;
    blockRoot["block_hash"] = static_cast<Json::UInt64>(block.blockHash);
    blockRoot["position_id"] = block.positionId;
    blockRoot["token_count"] = static_cast<Json::UInt>(block.tokenCount);
    root["blocks"].append(blockRoot);
  }

  return write(root);
}

std::string serialize(const OffloadResponseMessage& orm) {
  Json::Value root;
  root["id"] = static_cast<Json::UInt64>(orm.id);
  root["status"] = std::string(toWire(orm.status));

  return write(root);
}

std::optional<DownloadRequestMessage> parseDownloadRequest(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  if (!root.isMember("id") || !root["id"].isIntegral() ||
      !root.isMember("dst_slot") || !root["dst_slot"].isIntegral() ||
      !root.isMember("blocks") || !root["blocks"].isArray()) {
    TT_LOG_ERROR("[migration_message] Request missing required fields");
    return std::nullopt;
  }

  DownloadRequestMessage out{};
  out.id = root["id"].asUInt64();
  out.dst_slot = root["dst_slot"].asUInt();
  for (const auto& block : root["blocks"]) {
    out.blocks.push_back(tt::services::KVCacheBlockRef{
        .blockHash = block["block_hash"].asUInt64(),
        .positionId = block["position_id"].asUInt(),
        .tokenCount = block["token_count"].asUInt()});
  }

  return out;
}

std::optional<DownloadResponseMessage> parseDownloadResponse(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  if (!root.isMember("id") || !root["id"].isIntegral() ||
      !root.isMember("status") || !root["status"].isString() ||
      !root.isMember("downloaded_block_hashes") ||
      !root["downloaded_block_hashes"].isArray()) {
    TT_LOG_ERROR("[migration_message] Response missing required fields");
    return std::nullopt;
  }

  for (const auto& hash : root["downloaded_block_hashes"]) {
    if (!hash.isIntegral()) {
      TT_LOG_ERROR(
          "[migration_message] downloaded_block_hashes contains non-integral "
          "entry");
      return std::nullopt;
    }
  }

  auto status = fromWire(root["status"].asString());
  if (!status.has_value()) {
    TT_LOG_ERROR("[migration_message] Unknown status string: {}",
                 root["status"].asString());
    return std::nullopt;
  }

  DownloadResponseMessage out{};
  out.id = root["id"].asUInt64();
  out.status = *status;
  out.downloaded_block_hashes.reserve(root["downloaded_block_hashes"].size());
  for (const auto& hash : root["downloaded_block_hashes"]) {
    out.downloaded_block_hashes.push_back(hash.asUInt64());
  }

  return out;
}

std::optional<OffloadRequestMessage> parseOffloadRequest(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  if (!root.isMember("id") || !root["id"].isIntegral() ||
      !root.isMember("src_slot") || !root["src_slot"].isIntegral() ||
      !root.isMember("blocks") || !root["blocks"].isArray()) {
    TT_LOG_ERROR("[migration_message] Request missing required fields");
    return std::nullopt;
  }

  OffloadRequestMessage out{};
  out.id = root["id"].asUInt64();
  out.src_slot = root["src_slot"].asUInt();
  for (const auto& block : root["blocks"]) {
    out.blocks.push_back(tt::services::KVCacheBlockRef{
        .blockHash = block["block_hash"].asUInt64(),
        .positionId = block["position_id"].asUInt(),
        .tokenCount = block["token_count"].asUInt()});
  }

  return out;
}

std::optional<OffloadResponseMessage> parseOffloadResponse(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  if (!root.isMember("id") || !root["id"].isIntegral() ||
      !root.isMember("status") || !root["status"].isString()) {
    TT_LOG_ERROR("[migration_message] Response missing required fields");
    return std::nullopt;
  }

  auto status = fromWire(root["status"].asString());
  if (!status.has_value()) {
    TT_LOG_ERROR("[migration_message] Unknown status string: {}",
                 root["status"].asString());
    return std::nullopt;
  }

  OffloadResponseMessage out{};
  out.id = root["id"].asUInt64();
  out.status = *status;

  return out;
}

std::optional<MigrationRequestMessage> parseMigrationRequest(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  for (const char* field : {"migration_id", "src_slot", "dst_slot", "layer_id",
                            "position_start", "position_end"}) {
    if (!root.isMember(field) || !root[field].isIntegral()) {
      TT_LOG_ERROR("[migration_message] Request missing/non-integral: {}",
                   field);
      return std::nullopt;
    }
  }

  MigrationRequestMessage out{};
  out.migration_id = root["migration_id"].asUInt64();
  out.src_slot = root["src_slot"].asUInt();
  out.dst_slot = root["dst_slot"].asUInt();
  out.layer_id = root["layer_id"].asUInt();
  out.position_start = root["position_start"].asUInt();
  out.position_end = root["position_end"].asUInt();

  return out;
}

std::optional<MigrationResponseMessage> parseMigrationResponse(
    const std::string& json) {
  Json::Value root;
  if (!parse(json, root)) return std::nullopt;
  if (!root.isMember("migration_id") || !root["migration_id"].isIntegral() ||
      !root.isMember("status") || !root["status"].isString()) {
    TT_LOG_ERROR("[migration_message] Response missing required fields");
    return std::nullopt;
  }

  auto status = fromWire(root["status"].asString());
  if (!status.has_value()) {
    TT_LOG_ERROR("[migration_message] Unknown status string: {}",
                 root["status"].asString());
    return std::nullopt;
  }

  MigrationResponseMessage out{};
  out.migration_id = root["migration_id"].asUInt64();
  out.status = *status;

  return out;
}

}  // namespace tt::messaging
