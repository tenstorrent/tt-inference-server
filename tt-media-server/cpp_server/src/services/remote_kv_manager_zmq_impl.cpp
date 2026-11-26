// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_zmq_impl.hpp"

#include <string>
#include <utility>

#include "messaging/kvm_command_message.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

RemoteKVManagerZmqImpl::RemoteKVManagerZmqImpl(
    std::unique_ptr<tt::messaging::IKvmZmqTransport> transport,
    std::chrono::milliseconds timeout, std::chrono::milliseconds sweepInterval,
    int drainPollMs)
    : transport(std::move(transport)),
      timeout(timeout),
      sweepInterval(sweepInterval),
      drainPollMs(drainPollMs) {
  if (!this->transport) {
    TT_LOG_ERROR(
        "[RemoteKVManagerZmqImpl] null transport; migrate() will fail every "
        "call and acks will never arrive");
  }
  running.store(true, std::memory_order_relaxed);
  lastSweep = std::chrono::steady_clock::now();
  drainThread = std::thread([this] { drainLoop(); });
  TT_LOG_INFO(
      "[RemoteKVManagerZmqImpl] started (timeout={}ms, sweep={}ms, "
      "drainPoll={}ms)",
      this->timeout.count(), this->sweepInterval.count(), this->drainPollMs);
}

RemoteKVManagerZmqImpl::~RemoteKVManagerZmqImpl() {
  running.store(false, std::memory_order_relaxed);
  if (drainThread.joinable()) {
    drainThread.join();
  }
  TT_LOG_INFO("[RemoteKVManagerZmqImpl] stopped");
}

uint64_t RemoteKVManagerZmqImpl::migrate(const MigrationRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    // 64-bit random id collisions are astronomically unlikely; if one ever
    // does occur we keep the older record so the caller can observe its
    // existing state instead of us silently overwriting an in-flight
    // record. Matches RemoteKVManagerImpl.
    auto [it, inserted] = migrations.emplace(
        id, MigrationState{MigrationStatus::IN_PROGRESS, now});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerZmqImpl] id collision on command_id={}; returning "
          "existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  // kv_manager requires migration_id on the wire. When the caller didn't
  // provide one (rare — the adapter always sets it), fall back to the
  // per-request id so downstream logs still have a stable correlation key.
  const uint64_t migrationId = request.migration_id.value_or(id);

  const tt::messaging::KvmCommandMessage msg{
      .command_id = id,
      .migration_id = migrationId,
      .src_slot = request.src_slot,
      .dst_slot = request.dst_slot,
      .layer_begin = request.layer_begin,
      .layer_end = request.layer_end,
      .src_position_begin = request.src_position_begin,
      .src_position_end = request.src_position_end,
      .dst_position_begin = request.dst_position_begin,
      .dst_position_end = request.dst_position_end,
  };
  const std::string payload = tt::messaging::serialize(msg);

  bool sent = false;
  std::string err;
  if (transport) {
    sent = transport->send(payload, &err);
  } else {
    err = "no transport";
  }

  if (!sent) {
    // Roll the migration straight to FAILED so callers don't wait `timeout`
    // for a request that never made it onto the wire.
    std::lock_guard<std::mutex> lock(mtx);
    auto it = migrations.find(id);
    if (it != migrations.end() &&
        it->second.status == MigrationStatus::IN_PROGRESS) {
      it->second.status = MigrationStatus::FAILED;
    }
    TT_LOG_ERROR(
        "[RemoteKVManagerZmqImpl] transport.send failed for command_id={} "
        "migration_id={}: {}",
        id, migrationId, err);
  } else {
    TT_LOG_DEBUG(
        "[RemoteKVManagerZmqImpl] published command_id={} migration_id={}", id,
        migrationId);
  }

  return id;
}

MigrationStatus RemoteKVManagerZmqImpl::getMigrationStatus(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
    return MigrationStatus::UNKNOWN;
  }
  return it->second.status;
}

void RemoteKVManagerZmqImpl::drainLoop() {
  TT_LOG_INFO("[RemoteKVManagerZmqImpl] drain loop entered");

  while (running.load(std::memory_order_relaxed)) {
    if (transport) {
      auto msg = transport->receive(drainPollMs);
      if (msg.has_value()) {
        auto parsed = tt::messaging::parseKvmResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerZmqImpl] dropping unparsable ack payload: {}",
              *msg);
        } else {
          std::lock_guard<std::mutex> lock(mtx);
          auto it = migrations.find(parsed->command_id);
          if (it == migrations.end()) {
            TT_LOG_WARN(
                "[RemoteKVManagerZmqImpl] ack for unknown command_id={} "
                "migration_id={}; ignoring",
                parsed->command_id, parsed->migration_id);
          } else if (it->second.status != MigrationStatus::IN_PROGRESS) {
            TT_LOG_DEBUG(
                "[RemoteKVManagerZmqImpl] ack for already-terminal "
                "command_id={} status={}; ignoring",
                parsed->command_id, static_cast<int>(it->second.status));
          } else {
            it->second.status = parsed->status;
          }
        }
      }
    } else {
      // No transport: still respect the poll cadence so the loop doesn't spin.
      std::this_thread::sleep_for(std::chrono::milliseconds(drainPollMs));
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - lastSweep >= sweepInterval) {
      std::lock_guard<std::mutex> lock(mtx);
      sweepLocked(now);
      lastSweep = now;
    }
  }

  TT_LOG_INFO("[RemoteKVManagerZmqImpl] drain loop exited");
}

void RemoteKVManagerZmqImpl::sweepLocked(
    std::chrono::steady_clock::time_point now) {
  size_t timedOut = 0;
  for (auto& [id, state] : migrations) {
    if (state.status == MigrationStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = MigrationStatus::FAILED;
      ++timedOut;
      TT_LOG_WARN(
          "[RemoteKVManagerZmqImpl] command_id={} timed out after {}ms; "
          "marked FAILED",
          id, timeout.count());
    }
  }
  if (timedOut > 0) {
    TT_LOG_INFO("[RemoteKVManagerZmqImpl] sweeper timed out {} migration(s)",
                timedOut);
  }
}

}  // namespace tt::services
