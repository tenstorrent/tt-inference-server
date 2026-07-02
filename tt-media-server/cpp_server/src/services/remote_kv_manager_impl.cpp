// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/remote_kv_manager_impl.hpp"

#include <utility>

#include "messaging/migration_message.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services {

RemoteKVManagerImpl::RemoteKVManagerImpl(
    std::unique_ptr<tt::messaging::IKafkaProducer> requestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> ackConsumer,
    std::size_t migrationWorkerPoolSize, std::chrono::milliseconds timeout,
    std::chrono::milliseconds sweepInterval, int drainPollMs,
    std::unique_ptr<tt::messaging::IKafkaProducer> downloadRequestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> downloadAckConsumer,
    std::unique_ptr<tt::messaging::IKafkaProducer> offloadRequestProducer,
    std::unique_ptr<tt::messaging::IKafkaConsumer> offloadAckConsumer)
    : requestProducer(std::move(requestProducer)),
      ackConsumer(std::move(ackConsumer)),
      downloadRequestProducer(std::move(downloadRequestProducer)),
      downloadAckConsumer(std::move(downloadAckConsumer)),
      offloadRequestProducer(std::move(offloadRequestProducer)),
      offloadAckConsumer(std::move(offloadAckConsumer)),
      migrationWorkerPoolSize(migrationWorkerPoolSize),
      timeout(timeout),
      sweepInterval(sweepInterval),
      drainPollMs(drainPollMs) {
  if (!this->requestProducer) {
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] null requestProducer; migrate() will fail every "
        "call");
  }
  if (!this->ackConsumer) {
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] null ackConsumer; statuses will never "
        "transition out of IN_PROGRESS via ack");
  }
  if (!this->downloadRequestProducer) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] null downloadRequestProducer; "
        "downloadFromStore() will roll straight to FAILED");
  }
  if (!this->downloadAckConsumer) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] null downloadAckConsumer; downloads will "
        "only reach terminal state via the timeout sweeper");
  }
  if (!this->offloadRequestProducer) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] null offloadRequestProducer; "
        "offloadToStore() will silently drop payloads");
  }
  if (!this->offloadAckConsumer) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] null offloadAckConsumer; offloads will "
        "only reach terminal state via the timeout sweeper");
  }
  if (this->migrationWorkerPoolSize == 0) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] migrationWorkerPoolSize=0; clamping to 1. "
        "Downloads would never reach COMPLETED with pool size 0.");
    this->migrationWorkerPoolSize = 1;
  }
  running.store(true, std::memory_order_relaxed);
  lastSweep = std::chrono::steady_clock::now();
  drainThread = std::thread([this] { drainLoop(); });
  TT_LOG_INFO(
      "[RemoteKVManagerImpl] started (workerPool={}, timeout={}ms, sweep={}ms, "
      "drainPoll={}ms)",
      this->migrationWorkerPoolSize, this->timeout.count(),
      this->sweepInterval.count(), this->drainPollMs);
}

RemoteKVManagerImpl::~RemoteKVManagerImpl() {
  running.store(false, std::memory_order_relaxed);
  if (drainThread.joinable()) {
    drainThread.join();
  }
  TT_LOG_INFO("[RemoteKVManagerImpl] stopped");
}

uint64_t RemoteKVManagerImpl::migrate(const MigrationRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    // 64-bit random id collisions are astronomically unlikely; if one ever
    // occurs we keep the older record (insert is a no-op) and the caller
    // will observe whatever state that older migration is in. This is safer
    // than overwriting an in-flight record.
    auto [it, inserted] = migrations.emplace(
        id, MigrationState{MigrationStatus::IN_PROGRESS, now});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] id collision on migration_id={}; returning "
          "existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  const tt::messaging::MigrationRequestMessage msg{
      .migration_id = id,
      .src_slot = request.src_slot,
      .dst_slot = request.dst_slot,
      .layer_id = request.layer_id,
      .position_start = request.position_start,
      .position_end = request.position_end,
  };
  const std::string payload = tt::messaging::serialize(msg);

  bool sent = false;
  std::string err;
  if (requestProducer) {
    sent = requestProducer->send(payload, &err);
  } else {
    err = "no producer";
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
        "[RemoteKVManagerImpl] producer.send failed for migration_id={}: {}",
        id, err);
  }

  return id;
}

MigrationStatus RemoteKVManagerImpl::getMigrationStatus(
    uint64_t migrationId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = migrations.find(migrationId);
  if (it == migrations.end()) {
    return MigrationStatus::UNKNOWN;
  }
  return it->second.status;
}

uint64_t RemoteKVManagerImpl::downloadFromStore(
    const DownloadKVRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    auto [it, inserted] =
        downloads.emplace(id, DownloadState{MigrationStatus::IN_PROGRESS,
                                            /*downloadedBlockHashes=*/{}, now,
                                            /*successfulAcksReceived=*/0});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] id collision on download transfer_id={}; "
          "returning existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  tt::messaging::DownloadRequestMessage msg{
      .id = id,
      .dst_slot = request.dstSlot,
      .blocks = request.blocks,
  };
  const std::string payload = tt::messaging::serialize(msg);

  bool sent = false;
  std::string err;
  if (downloadRequestProducer) {
    sent = downloadRequestProducer->send(payload, &err);
  } else {
    err = "no downloadRequestProducer";
  }

  if (!sent) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = downloads.find(id);
    if (it != downloads.end() &&
        it->second.status == MigrationStatus::IN_PROGRESS) {
      it->second.status = MigrationStatus::FAILED;
      it->second.downloadedBlockHashes.clear();
    }
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] downloadRequestProducer.send failed for "
        "transfer_id={}: {}",
        id, err);
  } else {
    TT_LOG_INFO(
        "[RemoteKVManagerImpl] downloadFromStore published transfer_id={} "
        "dst_slot={} blocks={} pool={}",
        id, request.dstSlot, request.blocks.size(), migrationWorkerPoolSize);
  }
  return id;
}

DownloadKVResult RemoteKVManagerImpl::getDownloadResult(
    uint64_t transferId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(transferId);
  if (it == downloads.end()) {
    return DownloadKVResult{MigrationStatus::UNKNOWN, {}};
  }
  return DownloadKVResult{it->second.status, it->second.downloadedBlockHashes};
}

uint64_t RemoteKVManagerImpl::offloadToStore(const OffloadKVRequest& request) {
  const uint64_t id = tt::utils::MigrationIDGenerator::generate();
  const auto now = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(mtx);
    auto [it, inserted] =
        offloads.emplace(id, OffloadState{MigrationStatus::IN_PROGRESS, now});
    if (!inserted) {
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] id collision on offload transfer_id={}; "
          "returning existing record (status={})",
          id, static_cast<int>(it->second.status));
      return id;
    }
  }

  const tt::messaging::OffloadRequestMessage msg{
      .id = id,
      .src_slot = request.srcSlot,
      .blocks = request.blocks,
  };
  const std::string payload = tt::messaging::serialize(msg);

  std::string err;
  const bool sent =
      offloadRequestProducer && offloadRequestProducer->send(payload, &err);

  if (!sent) {
    TT_LOG_ERROR(
        "[RemoteKVManagerImpl] offloadRequestProducer.send failed for "
        "transfer_id={}: {}",
        id,
        offloadRequestProducer ? err
                               : std::string("no offloadRequestProducer"));
  } else {
    TT_LOG_INFO(
        "[RemoteKVManagerImpl] offloadToStore published transfer_id={} "
        "src_slot={} blocks={}",
        id, request.srcSlot, request.blocks.size());
  }
  return id;
}

MigrationStatus RemoteKVManagerImpl::getOffloadStatus(
    uint64_t transferId) const {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = offloads.find(transferId);
  if (it == offloads.end()) {
    return MigrationStatus::UNKNOWN;
  }
  return it->second.status;
}

void RemoteKVManagerImpl::drainLoop() {
  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop entered");

  while (running.load(std::memory_order_relaxed)) {
    bool didWork = false;

    if (ackConsumer) {
      auto msg = ackConsumer->receive(drainPollMs);
      if (msg.has_value()) {
        didWork = true;
        auto parsed = tt::messaging::parseMigrationResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerImpl] dropping unparseable ack payload: {}",
              *msg);
        } else {
          std::lock_guard<std::mutex> lock(mtx);
          auto it = migrations.find(parsed->migration_id);
          if (it == migrations.end()) {
            TT_LOG_WARN(
                "[RemoteKVManagerImpl] ack for unknown migration_id={}; "
                "ignoring",
                parsed->migration_id);
          } else if (it->second.status != MigrationStatus::IN_PROGRESS) {
            TT_LOG_DEBUG(
                "[RemoteKVManagerImpl] ack for already-terminal migration_id="
                "{}, status={}; ignoring",
                parsed->migration_id, static_cast<int>(it->second.status));
          } else {
            it->second.status = parsed->status;
          }
        }
      }
    }

    if (downloadAckConsumer) {
      auto msg = downloadAckConsumer->receive(drainPollMs);
      if (msg.has_value()) {
        didWork = true;
        auto parsed = tt::messaging::parseDownloadResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerImpl] dropping unparseable download-ack "
              "payload: {}",
              *msg);
        } else {
          applyDownloadAck(*parsed);
        }
      }
    }

    if (offloadAckConsumer) {
      auto msg = offloadAckConsumer->receive(drainPollMs);
      if (msg.has_value()) {
        didWork = true;
        auto parsed = tt::messaging::parseOffloadResponse(*msg);
        if (!parsed.has_value()) {
          TT_LOG_WARN(
              "[RemoteKVManagerImpl] dropping unparseable offload-ack payload: "
              "{}",
              *msg);
        } else {
          std::lock_guard<std::mutex> lock(mtx);
          auto it = offloads.find(parsed->id);
          if (it == offloads.end()) {
            TT_LOG_WARN(
                "[RemoteKVManagerImpl] ack for unknown offload transfer_id={}; "
                "ignoring",
                parsed->id);
          } else if (it->second.status != MigrationStatus::IN_PROGRESS) {
            TT_LOG_DEBUG(
                "[RemoteKVManagerImpl] ack for already-terminal offload "
                "transfer_id={}, status={}; ignoring",
                parsed->id, static_cast<int>(it->second.status));
          } else {
            it->second.status = parsed->status;
          }
        }
      }
    }

    if (!ackConsumer && !downloadAckConsumer && !offloadAckConsumer) {
      // Nothing to poll on -- respect the cadence so the loop doesn't spin.
      std::this_thread::sleep_for(std::chrono::milliseconds(drainPollMs));
    } else if (!didWork) {
      // Nothing arrived; already blocked in receive() for drainPollMs, so no
      // extra sleep needed.
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - lastSweep >= sweepInterval) {
      std::lock_guard<std::mutex> lock(mtx);
      sweepLocked(now);
      lastSweep = now;
    }
  }

  TT_LOG_INFO("[RemoteKVManagerImpl] drain loop exited");
}

void RemoteKVManagerImpl::applyDownloadAck(
    const tt::messaging::DownloadResponseMessage& ack) {
  std::lock_guard<std::mutex> lock(mtx);
  auto it = downloads.find(ack.id);
  if (it == downloads.end()) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] ack for unknown transfer_id={}; ignoring",
        ack.id);
    return;
  }

  auto& state = it->second;
  if (state.status != MigrationStatus::IN_PROGRESS) {
    TT_LOG_DEBUG(
        "[RemoteKVManagerImpl] ack for already-terminal transfer_id={}, "
        "status={}; ignoring",
        ack.id, static_cast<int>(state.status));
    return;
  }

  if (ack.status == MigrationStatus::FAILED) {
    state.status = MigrationStatus::FAILED;
    state.downloadedBlockHashes.clear();
    return;
  }

  if (ack.status != MigrationStatus::SUCCESSFUL) {
    TT_LOG_WARN(
        "[RemoteKVManagerImpl] unexpected download ack status={} for "
        "transfer_id={}; ignoring",
        static_cast<int>(ack.status), ack.id);
    return;
  }

  if (state.successfulAcksReceived == 0) {
    state.downloadedBlockHashes = ack.downloaded_block_hashes;
  } else {
    // Contiguous-prefix intersection with the running set: any first
    // divergence truncates the usable prefix (see DownloadKVResult
    // docstring's worker-hole worked example).
    std::size_t common = 0;
    while (common < state.downloadedBlockHashes.size() &&
           common < ack.downloaded_block_hashes.size() &&
           state.downloadedBlockHashes[common] ==
               ack.downloaded_block_hashes[common]) {
      ++common;
    }
    state.downloadedBlockHashes.resize(common);
  }
  ++state.successfulAcksReceived;

  if (state.successfulAcksReceived >= migrationWorkerPoolSize) {
    state.status = MigrationStatus::SUCCESSFUL;
  }
}

void RemoteKVManagerImpl::sweepLocked(
    std::chrono::steady_clock::time_point now) {
  size_t timedOutMigrations = 0;
  for (auto& [id, state] : migrations) {
    if (state.status == MigrationStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = MigrationStatus::FAILED;
      ++timedOutMigrations;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] migration_id={} timed out after {}ms; marked "
          "FAILED",
          id, timeout.count());
    }
  }
  if (timedOutMigrations > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} migration(s)",
                timedOutMigrations);
  }

  size_t timedOutDownloads = 0;
  for (auto& [id, state] : downloads) {
    if (state.status == MigrationStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = MigrationStatus::FAILED;
      state.downloadedBlockHashes.clear();
      ++timedOutDownloads;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] download transfer_id={} timed out after {}ms; "
          "marked FAILED",
          id, timeout.count());
    }
  }
  if (timedOutDownloads > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} download(s)",
                timedOutDownloads);
  }

  size_t timedOutOffloads = 0;
  for (auto& [id, state] : offloads) {
    if (state.status == MigrationStatus::IN_PROGRESS &&
        now - state.submittedAt >= timeout) {
      state.status = MigrationStatus::FAILED;
      ++timedOutOffloads;
      TT_LOG_WARN(
          "[RemoteKVManagerImpl] offload transfer_id={} timed out after {}ms; "
          "marked FAILED",
          id, timeout.count());
    }
  }
  if (timedOutOffloads > 0) {
    TT_LOG_INFO("[RemoteKVManagerImpl] sweeper timed out {} offload(s)",
                timedOutOffloads);
  }
}

}  // namespace tt::services
