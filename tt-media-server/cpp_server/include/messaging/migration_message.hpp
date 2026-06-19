// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <optional>

#include "services/remote_kv_manager.hpp"

namespace tt::messaging {

/**
    Payload for a migration message. Consumed by migration worker.
*/
struct MigrationRequestMessage {
    uint64_t migration_id;
    uint32_t src_slot;
    uint32_t dst_slot;
    uint32_t layer_id;
    uint32_t position_start;
    uint32_t position_end;
};

/**
    Payload for a migration response message. Produced by migration worker.
*/
struct MigrationResponseMessage {
    uint64_t migration_id;
    tt::services::MigrationStatus status;
};

/**
    JSON encode.
*/
std::string serialize(const MigrationRequestMessage& mrm);
std::string serialize(const MigrationResponseMessage& mrm);

/**
    JSON decode.
*/
std::optional<MigrationRequestMessage> parseMigrationRequest(const std::string& json);
std::optional<MigrationResponseMessage> parseMigrationResponse(const std::string& json);

} // namespace tt::messaging