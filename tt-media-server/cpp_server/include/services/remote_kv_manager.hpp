// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <vector>

namespace tt::services {

enum class MigrationStatus {
  UNKNOWN,
  IN_PROGRESS,
  SUCCESSFUL,
  FAILED,
};

struct Migration {
  uint64_t migration_id;
  std::chrono::steady_clock::time_point time_created;
  MigrationStatus status;
};

struct MigrationRequest {
  uint32_t src_slot;
  uint32_t dst_slot;
  uint32_t layer_id;
  uint32_t position_start;
  uint32_t position_end;
};

struct KVCacheBlockRef {
  uint64_t blockHash;
  uint32_t positionId;
  uint32_t tokenCount;
};

struct DownloadKVRequest {
  uint32_t dstSlot;
  std::vector<KVCacheBlockRef> blocks;
};

struct OffloadKVRequest {
  uint32_t srcSlot;
  std::vector<KVCacheBlockRef> blocks;
};

struct DownloadKVResult {
  MigrationStatus status;
  std::vector<uint64_t> downloadedBlockHashes;
};

/**
 * Async client to the pool of migration workers.
 */
class IRemoteKVManager {
 public:
  virtual ~IRemoteKVManager() = default;

  [[nodiscard]] virtual uint64_t migrate(const MigrationRequest& request) = 0;

  virtual MigrationStatus getMigrationStatus(uint64_t migrationId) const = 0;

  [[nodiscard]] virtual uint64_t downloadFromStore(
      const DownloadKVRequest& request) = 0;

  virtual DownloadKVResult getDownloadResult(uint64_t transferId) const = 0;

  [[nodiscard]] virtual uint64_t offloadToStore(
      const OffloadKVRequest& request) = 0;

  virtual MigrationStatus getOffloadStatus(uint64_t transferId) const = 0;
};

}  // namespace tt::services
