// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * In-process fake of IRemoteKVManager for tests and disaggregation dev mode.
 * Thread-safe. IDs are minted from a single monotonically-increasing
 * counter across migrate() / downloadFromStore() / offloadToStore();
 * cross-kind lookups (e.g. getDownloadRequest with a migrate id) return
 * std::nullopt.
 *
 * Defaults: migrate() resolves to SUCCESSFUL on first poll;
 * downloadFromStore() resolves to {COMPLETED, blocks.size()};
 * offloadToStore() is fire-and-forget and only records the request.
 *
 * setPollsBeforeResolution(n) / setDownloadPollsBeforeResolution(n)
 * defer the transition out of IN_PROGRESS by n polls so tests can
 * exercise the polling path. forceStatus / forceDownloadResult pin a
 * known id to a specific terminal value.
 */
class MockRemoteKVManager : public IRemoteKVManager {
 public:
  MockRemoteKVManager() = default;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getMigrationStatus(uint64_t migrationId) const override;
  [[nodiscard]] uint64_t downloadFromStore(
      const DownloadKVRequest& request) override;
  DownloadKVResult getDownloadResult(uint64_t transferId) const override;
  [[nodiscard]] uint64_t offloadToStore(
      const OffloadKVRequest& request) override;
  MigrationStatus getOffloadStatus(uint64_t transferId) const override;

  /// Drop all bookkeeping and reset the id counter. Knob settings are kept.
  void clear();

 private:
  mutable std::mutex mtx;

  mutable std::unordered_map<uint64_t, MigrationRequest> migrations;
  mutable std::unordered_map<uint64_t, DownloadKVRequest> downloads;
  mutable std::unordered_map<uint64_t, OffloadKVRequest> offloads;
};

}  // namespace tt::services
