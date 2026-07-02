// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "services/remote_kv_manager.hpp"

namespace tt::services {

using MigrationToken = uint64_t;

struct MigrationCompleteEvent {
  MigrationToken token;
  bool ok;
};

struct MigrationReceivedEvent {
  MigrationToken token;
  int32_t src_endpoint_id;
  uint32_t src_slot, dst_slot;
  uint32_t layer_start, layer_end_excl;
  uint32_t src_start_pos, src_end_pos_excl;
  uint32_t dest_start_pos, dest_end_pos_excl;
};

struct EndpointDisconnectedEvent {
  int32_t remote_endpoint_id;
};

struct MigrationFailedEvent {
  MigrationToken token;
  int32_t remote_endpoint_id;
  uint32_t reason;
};

/**
 * Adapter that implements MigrationClientInterface-compatible API using
 * IRemoteKVManager.
 *
 * This allows PrefillScheduler (which expects MigrationClientInterface) to
 * use our Kafka-backed RemoteKVManager for KV cache migrations.
 *
 * Threading: poll() should be called from the scheduler's ack_reader_thread.
 * migrate() and burst methods are thread-safe.
 */
class RemoteKVManagerAdapter {
 public:
  using BurstId = MigrationToken;

  explicit RemoteKVManagerAdapter(std::unique_ptr<IRemoteKVManager> kvManager,
                                  uint32_t layersPerChunk);

  ~RemoteKVManagerAdapter() = default;

  RemoteKVManagerAdapter(const RemoteKVManagerAdapter&) = delete;
  RemoteKVManagerAdapter& operator=(const RemoteKVManagerAdapter&) = delete;

  // --- Burst lifecycle ---
  BurstId start_burst(MigrationToken uuid);

  void enqueue_migration_in_burst(BurstId burst, int remote_endpoint_id,
                                  uint32_t src_slot, uint32_t dst_slot,
                                  uint32_t layer_start,
                                  uint32_t layer_end_exclusive,
                                  uint32_t pos_start,
                                  uint32_t pos_end_exclusive);

  MigrationToken finish_burst(BurstId burst);

  // --- Single migration (for ALLOCATE prefix copies) ---
  MigrationToken migrate(int remote_endpoint_id, uint32_t src_slot,
                         uint32_t dst_slot, uint32_t layer_start,
                         uint32_t layer_end_exclusive, uint32_t pos_start,
                         uint32_t pos_end_exclusive);

  // --- Polling ---
  int poll();

  uint32_t cmd_queue_write_space() const { return UINT32_MAX; }

  // --- Callbacks ---
  void on_migration_complete(
      std::function<void(const MigrationCompleteEvent&)> cb);

  void on_migration_received(
      std::function<void(const MigrationReceivedEvent&)> cb);

  void on_migration_failed(std::function<void(const MigrationFailedEvent&)> cb);

  void on_endpoint_disconnected(
      std::function<void(const EndpointDisconnectedEvent&)> cb);

  void on_connection_received(std::function<void(int remote_endpoint_id)> cb);

  // --- Setup / lifecycle ---
  void connect_to(int remote_endpoint_id, const std::string& role,
                  const std::string& service_name);

  void wait_ready(int timeout_ms = 5000);

  void shutdown(bool drain = true);

 private:
  struct BurstState {
    std::vector<uint64_t> migrationIds;
    bool finished = false;
  };

  struct InFlightMigration {
    MigrationToken token;
    bool isBurstMember = false;
    BurstId burstId = 0;
  };

  std::unique_ptr<IRemoteKVManager> kvManager_;
  uint32_t layersPerChunk_;

  mutable std::mutex mtx_;
  std::atomic<uint64_t> nextToken_{1};

  std::unordered_map<BurstId, BurstState> bursts_;
  std::unordered_map<uint64_t, InFlightMigration> inFlight_;

  std::function<void(const MigrationCompleteEvent&)> onComplete_;
  std::function<void(const MigrationReceivedEvent&)> onReceived_;
  std::function<void(const MigrationFailedEvent&)> onFailed_;
  std::function<void(const EndpointDisconnectedEvent&)> onDisconnected_;
  std::function<void(int)> onConnectionReceived_;
};

}  // namespace tt::services
