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
 * No Kafka, no worker pool — every migration is bookkept in-memory and
 * resolves synchronously according to the configured policy. Thread-safe:
 * callers may submit and poll from any thread.
 *
 * Default behavior: every migrate() resolves immediately to SUCCESSFUL, so
 * a test that doesn't care about timing can ignore the knobs entirely.
 *
 * Test-control knobs — apply to migrations submitted AFTER the call:
 *   setDefaultStatus(status)     terminal state for newly submitted migrations
 *                                (default: SUCCESSFUL).
 *   setPollsBeforeResolution(n)  getStatus() returns IN_PROGRESS for the
 *                                first n polls, then transitions to the
 *                                configured terminal state (default: 0 →
 *                                resolve on first poll).
 *
 * Per-id override (takes precedence over the staged resolution above):
 *   forceStatus(id, status)      pin an existing migration to a specific
 *                                status. Useful for failure-path tests.
 *
 * Inspection helpers (for assertions):
 *   migrationCount()             number of migrate() calls received.
 *   getRequest(id)               the originally submitted MigrationRequest.
 *   getMigration(id)             the current Migration bookkeeping record.
 */
class MockRemoteKVManager : public IRemoteKVManager {
 public:
  MockRemoteKVManager() = default;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getStatus(uint64_t migrationId) const override;

  void setDefaultStatus(MigrationStatus status);
  void setPollsBeforeResolution(size_t polls);
  void forceStatus(uint64_t migrationId, MigrationStatus status);

  /// Drop all migrations and reset the id counter. Knob settings are kept.
  void clear();

  size_t migrationCount() const;
  std::optional<MigrationRequest> getRequest(uint64_t migrationId) const;
  std::optional<Migration> getMigration(uint64_t migrationId) const;

 private:
  struct Entry {
    Migration migration;
    MigrationRequest request;
    // Polls remaining until status flips to `terminal`. 0 means the next
    // (or current) getStatus() resolves immediately.
    size_t pollsRemaining;
    MigrationStatus terminal;
  };

  mutable std::mutex mtx;
  uint64_t nextId = 1;
  MigrationStatus defaultTerminalStatus = MigrationStatus::SUCCESSFUL;
  size_t initialPollsBeforeResolution = 0;
  // mutable because getStatus() is const but lazily advances per-entry
  // poll counters on each call.
  mutable std::unordered_map<uint64_t, Entry> entries;
};

}  // namespace tt::services
