// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>
#include <unordered_map>

#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * Adapter that implements MigrationClientInterface using IRemoteKVManager.
 *
 * This allows PrefillScheduler to use our Kafka-backed RemoteKVManager for
 * KV cache migrations.
 *
 * IMPORTANT: This adapter only supports the single migrate() path used for
 * ALLOCATE prefix copies. Burst methods (start_burst,
 * enqueue_migration_in_burst, finish_burst) throw std::runtime_error. This
 * client is NOT a drop-in replacement for MigrationLayerClientAdapter in the
 * general prefill-burst flow. Use MigrationLayerClientAdapter for burst-based
 * migrations. Throwing from a burst method may terminate the calling thread
 * (typically ack_reader_thread). This is by design; wiring this adapter into
 * the burst flow is a programming error and should crash loudly.
 *
 * Threading:
 * - poll() should be called from the scheduler's ack_reader_thread.
 * - migrate() is thread-safe.
 * - shutdown(drain=true) must be called from a thread OTHER than the poll()
 *   owner. The caller must ensure poll() continues to run on its owner thread
 *   until shutdown returns, otherwise the drain will time out.
 *
 * Callback registration: Register all callbacks during setup, before the first
 * migrate() or poll() call. Callbacks are read without synchronization from
 * poll() on the owner thread.
 */
class RemoteKVManagerAdapter
    : public tt_llm_engine::scheduler::MigrationClientInterface {
 public:
  using MigrationToken = tt_llm_engine::scheduler::MigrationToken;
  using MigrationCompleteEvent =
      tt_llm_engine::scheduler::MigrationCompleteEvent;
  using MigrationReceivedEvent =
      tt_llm_engine::scheduler::MigrationReceivedEvent;
  using MigrationFailedEvent = tt_llm_engine::scheduler::MigrationFailedEvent;
  using EndpointDisconnectedEvent =
      tt_llm_engine::scheduler::EndpointDisconnectedEvent;

  /**
   * @param kvManager The underlying KV manager for migrations.
   * @param shutdownTimeout Max time shutdown(drain=true) waits for in-flight
   *   migrations to complete. Default 30s.
   */
  explicit RemoteKVManagerAdapter(
      std::unique_ptr<IRemoteKVManager> kvManager,
      std::chrono::milliseconds shutdownTimeout = std::chrono::seconds(30));

  ~RemoteKVManagerAdapter() override = default;

  RemoteKVManagerAdapter(const RemoteKVManagerAdapter&) = delete;
  RemoteKVManagerAdapter& operator=(const RemoteKVManagerAdapter&) = delete;

  // --- Burst lifecycle (NOT SUPPORTED - throws std::runtime_error) ---
  BurstId start_burst(MigrationToken uuid) override;

  void enqueue_migration_in_burst(BurstId burst, int remote_endpoint_id,
                                  uint32_t src_slot, uint32_t dst_slot,
                                  uint32_t layer_start,
                                  uint32_t layer_end_exclusive,
                                  uint32_t pos_start,
                                  uint32_t pos_end_exclusive) override;

  MigrationToken finish_burst(BurstId burst) override;

  uint32_t cmd_queue_write_space() const override { return UINT32_MAX; }

  // --- Single migration (for ALLOCATE prefix copies) ---
  MigrationToken migrate(int remote_endpoint_id, uint32_t src_slot,
                         uint32_t dst_slot, uint32_t layer_start,
                         uint32_t layer_end_exclusive, uint32_t pos_start,
                         uint32_t pos_end_exclusive) override;

  // --- Polling ---
  int poll() override;

  void on_migration_complete(
      std::function<void(const MigrationCompleteEvent&)> cb) override;

  void on_migration_received(
      std::function<void(const MigrationReceivedEvent&)> cb) override;

  void on_migration_failed(
      std::function<void(const MigrationFailedEvent&)> cb) override;

  void on_endpoint_disconnected(
      std::function<void(const EndpointDisconnectedEvent&)> cb) override;

  void on_connection_received(
      std::function<void(int remote_endpoint_id)> cb) override;

  // --- Setup / lifecycle ---
  void connect_to(int remote_endpoint_id, const std::string& role,
                  const std::string& service_name) override;

  void wait_ready(int timeout_ms = 5000) override;

  void shutdown(bool drain = true) override;

 private:
  struct InFlightMigration {
    MigrationToken token;
  };

  std::unique_ptr<IRemoteKVManager> kvManager_;

  mutable std::mutex mtx_;
  std::condition_variable drainCv_;
  std::unordered_map<uint64_t, InFlightMigration> inFlight_;
  bool shutdownRequested_{false};

  std::chrono::milliseconds shutdownTimeout_;

  std::function<void(const MigrationCompleteEvent&)> onComplete_;
  std::function<void(const MigrationReceivedEvent&)> onReceived_;
  std::function<void(const MigrationFailedEvent&)> onFailed_;
  std::function<void(const EndpointDisconnectedEvent&)> onDisconnected_;
  std::function<void(int)> onConnectionReceived_;
};

}  // namespace tt::services
