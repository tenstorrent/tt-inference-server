// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * In-process fake of IRemoteKVManager for tests and disaggregation dev mode.
 *
 * No Kafka, no worker pool — every migration / download / offload is
 * bookkept in-memory and resolves synchronously according to the configured
 * policy. Thread-safe: callers may submit and poll from any thread.
 *
 * Default behavior:
 *   - migrate()           resolves immediately to SUCCESSFUL.
 *   - downloadFromStore() resolves immediately to
 *                         {COMPLETED, usablePrefixCount = blocks.size()}.
 *   - offloadToStore()    fire-and-forget; just records the call.
 *
 * Test-control knobs — apply to operations submitted AFTER the call:
 *   setDefaultStatus(status)
 *       terminal MigrationStatus for new migrations (default: SUCCESSFUL).
 *   setPollsBeforeResolution(n)
 *       getStatus() returns IN_PROGRESS for the first n polls then
 *       transitions to the configured terminal MigrationStatus.
 *   setDefaultDownloadResult(result)
 *       terminal KVTransferResult for new downloads (default:
 *       {COMPLETED, blocks.size()}). Callers usually set status =
 *       COMPLETED and a specific usablePrefixCount to simulate partial
 *       hits in the store.
 *   setDownloadPollsBeforeResolution(n)
 *       getDownloadResult() returns {IN_PROGRESS, 0} for the first n
 *       polls then transitions to the configured terminal result.
 *
 * Per-id overrides (take precedence over the staged resolution above):
 *   forceStatus(id, status)
 *       pin a known migration to a specific MigrationStatus.
 *   forceDownloadResult(id, result)
 *       pin a known download to a specific KVTransferResult.
 *
 * Inspection helpers (for assertions):
 *   migrationCount() / downloadCount() / offloadCount()
 *       number of submissions of each kind.
 *   getRequest(id) / getDownloadRequest(id) / getOffloadRequest(id)
 *       the originally submitted request, if the id was issued by the
 *       matching submit call.
 *   getMigration(id)
 *       the bookkeeping record for a migrate() call.
 *
 * IDs are minted from a single monotonically-increasing counter across
 * all three submit methods, mirroring the production impl. Cross-kind
 * lookups (e.g. calling getDownloadRequest with a migrate id) return
 * std::nullopt.
 */
class MockRemoteKVManager : public IRemoteKVManager {
 public:
  MockRemoteKVManager() = default;

  // IRemoteKVManager
  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getStatus(uint64_t migrationId) const override;
  [[nodiscard]] uint64_t downloadFromStore(
      const DownloadKVRequest& request) override;
  KVTransferResult getDownloadResult(uint64_t transferId) const override;
  uint64_t offloadToStore(const OffloadKVRequest& request) override;

  // Migration knobs
  void setDefaultStatus(MigrationStatus status);
  void setPollsBeforeResolution(size_t polls);
  void forceStatus(uint64_t migrationId, MigrationStatus status);

  // Download knobs
  void setDefaultDownloadResult(KVTransferResult result);
  void setDownloadPollsBeforeResolution(size_t polls);
  void forceDownloadResult(uint64_t transferId, KVTransferResult result);

  /// Drop all bookkeeping and reset the id counter. Knob settings are kept.
  void clear();

  // Migration inspection
  size_t migrationCount() const;
  std::optional<MigrationRequest> getRequest(uint64_t migrationId) const;
  std::optional<Migration> getMigration(uint64_t migrationId) const;

  // Download / offload inspection
  size_t downloadCount() const;
  size_t offloadCount() const;
  std::optional<DownloadKVRequest> getDownloadRequest(uint64_t transferId) const;
  std::optional<OffloadKVRequest> getOffloadRequest(uint64_t offloadId) const;

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
    // Polls remaining until status flips to terminal. 0 means immediate.
    size_t pollsRemaining;
    KVTransferResult terminal;
  };

  mutable std::mutex mtx;
  uint64_t nextId = 1;

  // Migration knob state
  MigrationStatus defaultTerminalStatus = MigrationStatus::SUCCESSFUL;
  size_t initialPollsBeforeResolution = 0;

  // Download knob state. Default sentinel (status = COMPLETED, count =
  // SIZE_MAX_SENTINEL) means "use the request's block count" at
  // submission time so test code doesn't have to keep re-setting the
  // knob for every request size.
  static constexpr uint32_t USE_FULL_PREFIX_SENTINEL =
      static_cast<uint32_t>(-1);
  KVTransferResult defaultTerminalDownloadResult{KVTransferStatus::COMPLETED,
                                                 USE_FULL_PREFIX_SENTINEL};
  size_t initialDownloadPollsBeforeResolution = 0;

  // mutable because getStatus() / getDownloadResult() are const but
  // lazily advance per-entry poll counters on each call.
  mutable std::unordered_map<uint64_t, MigrationEntry> migrations;
  mutable std::unordered_map<uint64_t, DownloadEntry> downloads;

  // Offload is fire-and-forget; we record only the request for
  // assertion purposes. No status, no poll counter.
  std::unordered_map<uint64_t, OffloadKVRequest> offloads;
};

}  // namespace tt::services
