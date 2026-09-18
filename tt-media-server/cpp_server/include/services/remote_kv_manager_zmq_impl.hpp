// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "messaging/i_kvm_zmq_transport.hpp"
#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * ZMQ-backed implementation of `IRemoteKVManager`. Analogue of
 * `RemoteKVManagerImpl` (Kafka), tailored to kv_manager's ZMQ command
 * transport.
 *
 * `migrate()` serializes a `KvmCommandMessage` and hands it to
 * `IKvmZmqTransport::send`. A single background drain thread pulls acks
 * from `IKvmZmqTransport::receive`, updates the per-request status map,
 * and periodically sweeps stale IN_PROGRESS entries to FAILED so callers
 * always eventually observe a terminal state — even if kv_manager
 * disappears mid-burst.
 *
 * Notably absent (vs. the Kafka impl): no `LayerToPartition` mapping.
 * kv_manager owns the internal fan-out from the prefill-leader to its
 * peers, so from tt-inference-server's point of view every command goes to
 * a single ZMQ endpoint regardless of `layer_begin`.
 *
 * Thread-safety: `migrate()` and `getMigrationStatus()` are safe to call
 * from any thread.
 */
class RemoteKVManagerZmqImpl : public IRemoteKVManager {
 public:
  RemoteKVManagerZmqImpl(
      std::unique_ptr<tt::messaging::IKvmZmqTransport> transport,
      std::chrono::milliseconds timeout = std::chrono::seconds(60),
      std::chrono::milliseconds sweepInterval = std::chrono::seconds(5),
      int drainPollMs = 100);

  ~RemoteKVManagerZmqImpl() override;

  RemoteKVManagerZmqImpl(const RemoteKVManagerZmqImpl&) = delete;
  RemoteKVManagerZmqImpl& operator=(const RemoteKVManagerZmqImpl&) = delete;

  [[nodiscard]] uint64_t migrate(const MigrationRequest& request) override;
  MigrationStatus getMigrationStatus(uint64_t migrationId) const override;

 private:
  void drainLoop();
  void sweepLocked(std::chrono::steady_clock::time_point now);

  struct MigrationState {
    MigrationStatus status;
    std::chrono::steady_clock::time_point submittedAt;
  };

  std::unique_ptr<tt::messaging::IKvmZmqTransport> transport;
  std::chrono::milliseconds timeout;
  std::chrono::milliseconds sweepInterval;
  int drainPollMs;

  mutable std::mutex mtx;
  // Keyed by command_id (per-request id) — same role kafka_request_id
  // plays in the Kafka path. The parent burst id (`migration_id`) lives on
  // the wire only; correlation back to a burst is the adapter's job.
  std::unordered_map<uint64_t, MigrationState> migrations;

  std::atomic<bool> running{false};
  std::thread drainThread;
  std::chrono::steady_clock::time_point lastSweep{};
};

}  // namespace tt::services
