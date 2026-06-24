// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/mock_remote_kv_manager.hpp"

#include <chrono>

namespace tt::services {

namespace {

std::time_t nowSeconds() {
  return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

}  // namespace

uint64_t MockRemoteKVManager::migrate(const MigrationRequest& request) {
  std::lock_guard<std::mutex> lock(mtx);
  const uint64_t id = nextId++;

  // If no polling delay was requested we can short-circuit straight to the
  // terminal status — that matches the real worker's "already done by the
  // time we observe it" path and keeps simple tests one-call-and-go.
  const MigrationStatus initialStatus = initialPollsBeforeResolution == 0
                                            ? defaultTerminalStatus
                                            : MigrationStatus::IN_PROGRESS;

  Migration migration{
      /*migration_id=*/id,
      /*time_created=*/nowSeconds(),
      /*status=*/initialStatus,
  };

  entries.emplace(id, Entry{
                          /*migration=*/std::move(migration),
                          /*request=*/request,
                          /*pollsRemaining=*/initialPollsBeforeResolution,
                          /*terminal=*/defaultTerminalStatus,
                      });
  return id;
}

MigrationStatus MockRemoteKVManager::getStatus(uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = entries.find(migrationId);
  if (it == entries.end()) {
    return MigrationStatus::UNKNOWN;
  }

  Entry& e = it->second;
  // Once a migration has reached a terminal state (either the natural
  // terminal or one set by forceStatus) we just report it back.
  if (e.migration.status != MigrationStatus::IN_PROGRESS) {
    return e.migration.status;
  }

  if (e.pollsRemaining > 0) {
    --e.pollsRemaining;
  }
  if (e.pollsRemaining == 0) {
    e.migration.status = e.terminal;
  }

  return e.migration.status;
}

void MockRemoteKVManager::setDefaultStatus(MigrationStatus status) {
  std::lock_guard<std::mutex> lock(mtx);
  defaultTerminalStatus = status;
}

void MockRemoteKVManager::setPollsBeforeResolution(size_t polls) {
  std::lock_guard<std::mutex> lock(mtx);
  initialPollsBeforeResolution = polls;
}

void MockRemoteKVManager::forceStatus(uint64_t migrationId,
                                      MigrationStatus status) {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = entries.find(migrationId);
  if (it == entries.end()) {
    // Silently ignore — id was never issued or has been cleared. We don't
    // throw here because tests routinely call forceStatus with ids they
    // haven't yet observed; the wrong-order case fails loudly via UNKNOWN
    // returns from getStatus().
    return;
  }

  it->second.migration.status = status;
  it->second.terminal = status;
  it->second.pollsRemaining = 0;
}

void MockRemoteKVManager::clear() {
  std::lock_guard<std::mutex> lock(mtx);
  entries.clear();
  nextId = 1;
}

size_t MockRemoteKVManager::migrationCount() const {
  std::lock_guard<std::mutex> lock(mtx);
  return entries.size();
}

std::optional<MigrationRequest> MockRemoteKVManager::getRequest(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = entries.find(migrationId);
  if (it == entries.end()) return std::nullopt;

  return it->second.request;
}

std::optional<Migration> MockRemoteKVManager::getMigration(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = entries.find(migrationId);
  if (it == entries.end()) return std::nullopt;

  return it->second.migration;
}

}  // namespace tt::services
