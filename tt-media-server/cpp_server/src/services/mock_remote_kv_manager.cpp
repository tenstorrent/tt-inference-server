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

  migrations.emplace(id, MigrationEntry{
                             /*migration=*/std::move(migration),
                             /*request=*/request,
                             /*pollsRemaining=*/initialPollsBeforeResolution,
                             /*terminal=*/defaultTerminalStatus,
                         });
  return id;
}

MigrationStatus MockRemoteKVManager::getStatus(uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
    return MigrationStatus::UNKNOWN;
  }

  MigrationEntry& e = it->second;
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
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
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

uint64_t MockRemoteKVManager::downloadFromStore(
    const DownloadKVRequest& request) {
  std::lock_guard<std::mutex> lock(mtx);
  const uint64_t id = nextId++;

  KVTransferResult terminal = defaultTerminalDownloadResult;
  if (terminal.usablePrefixCount == USE_FULL_PREFIX_SENTINEL) {
    terminal.usablePrefixCount = static_cast<uint32_t>(request.blocks.size());
  }

  const KVTransferStatus initialStatus =
      initialDownloadPollsBeforeResolution == 0 ? terminal.status
                                                : KVTransferStatus::IN_PROGRESS;

  downloads.emplace(
      id, DownloadEntry{
              /*request=*/request,
              /*status=*/initialStatus,
              /*pollsRemaining=*/initialDownloadPollsBeforeResolution,
              /*terminal=*/terminal,
          });
  return id;
}

KVTransferResult MockRemoteKVManager::getDownloadResult(
    uint64_t transferId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(transferId);
  if (it == downloads.end()) {
    return KVTransferResult{KVTransferStatus::UNKNOWN, 0};
  }

  DownloadEntry& e = it->second;
  if (e.status != KVTransferStatus::IN_PROGRESS) {
    return KVTransferResult{e.status,
                            e.status == KVTransferStatus::COMPLETED
                                ? e.terminal.usablePrefixCount
                                : 0u};
  }

  if (e.pollsRemaining > 0) {
    --e.pollsRemaining;
  }
  if (e.pollsRemaining == 0) {
    e.status = e.terminal.status;
  }

  return KVTransferResult{e.status,
                          e.status == KVTransferStatus::COMPLETED
                              ? e.terminal.usablePrefixCount
                              : 0u};
}

void MockRemoteKVManager::setDefaultDownloadResult(KVTransferResult result) {
  std::lock_guard<std::mutex> lock(mtx);
  defaultTerminalDownloadResult = result;
}

void MockRemoteKVManager::setDownloadPollsBeforeResolution(size_t polls) {
  std::lock_guard<std::mutex> lock(mtx);
  initialDownloadPollsBeforeResolution = polls;
}

void MockRemoteKVManager::forceDownloadResult(uint64_t transferId,
                                              KVTransferResult result) {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(transferId);
  if (it == downloads.end()) {
    return;
  }
  it->second.status = result.status;
  it->second.terminal = result;
  it->second.pollsRemaining = 0;
}

void MockRemoteKVManager::offloadToStore(const OffloadKVRequest& request) {
  std::lock_guard<std::mutex> lock(mtx);
  offloads.push_back(request);
}

void MockRemoteKVManager::clear() {
  std::lock_guard<std::mutex> lock(mtx);
  migrations.clear();
  downloads.clear();
  offloads.clear();
  nextId = 1;
}

size_t MockRemoteKVManager::migrationCount() const {
  std::lock_guard<std::mutex> lock(mtx);
  return migrations.size();
}

size_t MockRemoteKVManager::downloadCount() const {
  std::lock_guard<std::mutex> lock(mtx);
  return downloads.size();
}

size_t MockRemoteKVManager::offloadCount() const {
  std::lock_guard<std::mutex> lock(mtx);
  return offloads.size();
}

std::optional<MigrationRequest> MockRemoteKVManager::getRequest(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) return std::nullopt;

  return it->second.request;
}

std::optional<Migration> MockRemoteKVManager::getMigration(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) return std::nullopt;

  return it->second.migration;
}

std::optional<DownloadKVRequest> MockRemoteKVManager::getDownloadRequest(
    uint64_t transferId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(transferId);
  if (it == downloads.end()) return std::nullopt;
  return it->second.request;
}

std::vector<OffloadKVRequest> MockRemoteKVManager::offloadRequests() const {
  std::lock_guard<std::mutex> lock(mtx);
  return offloads;
}

}  // namespace tt::services
