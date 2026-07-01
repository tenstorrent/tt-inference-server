// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "services/remote_kv_manager.hpp"

namespace tt::messaging {

struct MigrationRequestMessage {
  uint64_t migration_id;
  uint32_t src_slot;
  uint32_t dst_slot;
  uint32_t layer_id;
  uint32_t position_start;
  uint32_t position_end;
};

struct MigrationResponseMessage {
  uint64_t migration_id;
  tt::services::MigrationStatus status;
};

struct DownloadRequestMessage {
  uint64_t id;
  uint32_t dst_slot;
  std::vector<tt::services::KVCacheBlockRef> blocks;
};

struct DownloadResponseMessage {
  uint64_t id;
  tt::services::MigrationStatus status;
  std::vector<uint64_t> downloaded_block_hashes;
};

struct OffloadRequestMessage {
  uint64_t id;
  uint32_t src_slot;
  std::vector<tt::services::KVCacheBlockRef> blocks;
};

struct OffloadResponseMessage {
  uint64_t id;
  tt::services::MigrationStatus status;
};

std::string serialize(const MigrationRequestMessage& mrm);
std::string serialize(const MigrationResponseMessage& mrm);
std::string serialize(const DownloadRequestMessage& drm);
std::string serialize(const DownloadResponseMessage& drm);
std::string serialize(const OffloadRequestMessage& orm);
std::string serialize(const OffloadResponseMessage& orm);

std::optional<MigrationRequestMessage> parseMigrationRequest(
    const std::string& json);
std::optional<MigrationResponseMessage> parseMigrationResponse(
    const std::string& json);
std::optional<DownloadRequestMessage> parseDownloadRequest(
    const std::string& json);
std::optional<DownloadResponseMessage> parseDownloadResponse(
    const std::string& json);
std::optional<OffloadRequestMessage> parseOffloadRequest(
    const std::string& json);
std::optional<OffloadResponseMessage> parseOffloadResponse(
    const std::string& json);
}  // namespace tt::messaging