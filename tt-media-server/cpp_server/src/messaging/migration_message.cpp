// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/migration_message.hpp"

#include <sstream>
#include <string_view>

#include <json/json.h>
#include <json/value.h>

#include "utils/logger.hpp"

namespace tt::messaging {

namespace {

constexpr std::string_view K_STATUS_UNKNOWN = "UNKNOWN";
constexpr std::string_view K_STATUS_IN_PROGRESS = "IN_PROGRESS";
constexpr std::string_view K_STATUS_SUCCESSFUL = "SUCCESSFUL";
constexpr std::string_view K_STATUS_FAILED = "FAILED";

std::string_view toWire(tt::services::MigrationStatus status) {
    using Status = tt::services::MigrationStatus;
    switch(status) {
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
} // namespace

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
    root["migration_id"] = mrm.migration_id;
    root["status"] = std::string(toWire(mrm.status));
    
    return write(root);
}

std::optional<MigrationRequestMessage> parseMigrationRequest(
    const std::string& json) {
    Json::Value root;
    if (!parse(json, root)) return std::nullopt;
    for (const char* field :
        {"migration_id", "src_slot", "dst_slot", "layer_id",
        "position_start", "position_end"}) {
        if (!root.isMember(field) || !root[field].isIntegral()) {
            TT_LOG_ERROR("[migration_message] Request missing/non-integral: {}",field);
            return std::nullopt;
        }
    }

    MigrationRequestMessage out{};
    out.migration_id   = root["migration_id"].asUInt64();
    out.src_slot       = root["src_slot"].asUInt();
    out.dst_slot       = root["dst_slot"].asUInt();
    out.layer_id       = root["layer_id"].asUInt();
    out.position_start = root["position_start"].asUInt();
    out.position_end   = root["position_end"].asUInt();
    
    return out;
}

std::optional<MigrationResponseMessage> parseMigrationResponse(const std::string& json) {
    Json::Value root;
    if (!parse(json, root)) return std::nullopt;
    if (!root.isMember("migration_id") || !root["migration_id"].isIntegral() ||
        !root.isMember("status")       || !root["status"].isString()) {
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
    out.status       = *status;

    return out;
}
} // namespace tt::messaging

