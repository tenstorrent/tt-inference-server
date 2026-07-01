// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/mock_remote_kv_manager.hpp"

#include "services/remote_kv_manager.hpp"
#include "utils/id_generator.hpp"

namespace tt::services {

uint64_t MockRemoteKVManager::migrate(const MigrationRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();

  std::lock_guard<std::mutex> lock(mtx);
  migrations.emplace(id, request);
  return id;
}

MigrationStatus MockRemoteKVManager::getMigrationStatus(
    uint64_t migrationId) const {
  return MigrationStatus::IN_PROGRESS;
}

uint64_t MockRemoteKVManager::downloadFromStore(
    const DownloadKVRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();

  std::lock_guard<std::mutex> lock(mtx);
  downloads.emplace(id, request);
  return id;
}

DownloadKVResult MockRemoteKVManager::getDownloadResult(
    uint64_t transferId) const {
  return DownloadKVResult{MigrationStatus::SUCCESSFUL, {}};
}

uint64_t MockRemoteKVManager::offloadToStore(const OffloadKVRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();

  std::lock_guard<std::mutex> lock(mtx);
  offloads.emplace(id, request);
  return id;
}

MigrationStatus MockRemoteKVManager::getOffloadStatus(
    uint64_t transferId) const {
  return MigrationStatus::SUCCESSFUL;
}

void MockRemoteKVManager::clear() {
  std::lock_guard<std::mutex> lock(mtx);
  migrations.clear();
  downloads.clear();
  offloads.clear();
}

}  // namespace tt::services
