// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

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
  MigrationStatus getStatus(uint64_t migrationId) const override;
  [[nodiscard]] uint64_t downloadFromStore(
      const DownloadKVRequest& request) override;
  KVTransferResult getDownloadResult(uint64_t transferId) const override;
  void offloadToStore(const OffloadKVRequest& request) override;

  void setDefaultStatus(MigrationStatus status);
  void setPollsBeforeResolution(size_t polls);
  void forceStatus(uint64_t migrationId, MigrationStatus status);

  void setDefaultDownloadResult(KVTransferResult result);
  void setDownloadPollsBeforeResolution(size_t polls);
  void forceDownloadResult(uint64_t transferId, KVTransferResult result);

  /// Drop all bookkeeping and reset the id counter. Knob settings are kept.
  void clear();

  size_t migrationCount() const;
  std::optional<MigrationRequest> getRequest(uint64_t migrationId) const;
  std::optional<Migration> getMigration(uint64_t migrationId) const;

  size_t downloadCount() const;
  size_t offloadCount() const;
  std::optional<DownloadKVRequest> getDownloadRequest(uint64_t transferId) const;
  /// Snapshot of all offloads ever submitted, in submission order.
  std::vector<OffloadKVRequest> offloadRequests() const;

 private:
  struct MigrationEntry {
    Migration migration;
    MigrationRequest request;
    // Polls remaining until status flips to `terminal`. 0 means the next
    // (or current) getStatus() resolves immediately.
    size_t pollsRemaining;
    MigrationStatus terminal;
  };

  struct DownloadEntry {
    DownloadKVRequest request;
    KVTransferStatus status;
    size_t pollsRemaining;
    KVTransferResult terminal;
  };

  mutable std::mutex mtx;
  uint64_t nextId = 1;

  MigrationStatus defaultTerminalStatus = MigrationStatus::SUCCESSFUL;
  size_t initialPollsBeforeResolution = 0;

  // Sentinel: at submit time, a usablePrefixCount of this value is
  // replaced with the request's block count. Lets tests leave the
  // download knob alone and still get "full hit" by default regardless
  // of request size.
  static constexpr uint32_t USE_FULL_PREFIX_SENTINEL =
      static_cast<uint32_t>(-1);
  KVTransferResult defaultTerminalDownloadResult{KVTransferStatus::COMPLETED,
                                                 USE_FULL_PREFIX_SENTINEL};
  size_t initialDownloadPollsBeforeResolution = 0;

  // mutable because getStatus() / getDownloadResult() are const but
  // lazily advance per-entry poll counters on each call.
  mutable std::unordered_map<uint64_t, MigrationEntry> migrations;
  mutable std::unordered_map<uint64_t, DownloadEntry> downloads;

  std::vector<OffloadKVRequest> offloads;
};

}  // namespace tt::services
