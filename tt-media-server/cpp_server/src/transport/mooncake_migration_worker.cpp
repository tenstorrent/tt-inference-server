// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_worker.hpp"

#include <chrono>
#include <thread>
#include <utility>

#include "transport/i_storage_backend.hpp"
#include "transport/peer_discovery.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {
constexpr int K_HOLD_POLL_MS = 200;
constexpr int K_HEARTBEAT_SEC = 30;  // periodic "still alive" log while holding
}  // namespace

MooncakeMigrationWorker::MooncakeMigrationWorker(
    MigrationWorkerConfig config, std::shared_ptr<ITransferEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {}

MooncakeMigrationWorker::~MooncakeMigrationWorker() { teardown(); }

// Ordered bring-up. Each phase only proceeds if the previous one
// succeeded.
bool MooncakeMigrationWorker::bringUp() {
  if (!engine_) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] bringUp: no engine");
    return false;
  }
  if (config_.host_dram_bytes == 0) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] bringUp: host_dram_bytes is 0");
    return false;
  }

  // Phase 2: allocate the host-DRAM pool peers will write into.
  hostDramPool_.assign(config_.host_dram_bytes, 0);

  // Phase 3: init the engine against the metadata service.
  EngineConfig ecfg;
  ecfg.metadata_uri = config_.metadata_uri;
  ecfg.local_server_name = config_.segment_name;
  ecfg.protocol = config_.protocol;
  if (!engine_->init(ecfg)) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] bringUp: engine init failed");
    return false;
  }

  // Phase 4: register memory — this publishes our segment to the cluster.
  if (!engine_->registerLocalMemory(hostDramPool_.data(),
                                    hostDramPool_.size())) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] bringUp: registerLocalMemory failed");
    return false;
  }
  memoryRegistered_ = true;
  TT_LOG_INFO("[MooncakeMigrationWorker] '{}' published {} bytes",
              config_.segment_name, hostDramPool_.size());

  // Phase 5: discover peers — the readiness gate.
  if (!connect()) {
    teardown();
    return false;
  }
  return true;
}

// Phase 6: hold until the caller's stop source fires, then tear down.
void MooncakeMigrationWorker::run(const std::atomic<bool>& stopRequested) {
  TT_LOG_INFO(
      "[MooncakeMigrationWorker] '{}' READY; holding until stop ({} peers)",
      config_.segment_name, peers_.size());
  const auto start = std::chrono::steady_clock::now();
  auto nextHeartbeat = start + std::chrono::seconds(K_HEARTBEAT_SEC);
  while (!stopRequested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_HOLD_POLL_MS));
    const auto now = std::chrono::steady_clock::now();
    if (now < nextHeartbeat) continue;
    const auto upSec =
        std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    TT_LOG_INFO("[MooncakeMigrationWorker] '{}' alive — {} peers, up {}s",
                config_.segment_name, peers_.size(), upSec);
    nextHeartbeat = now + std::chrono::seconds(K_HEARTBEAT_SEC);
  }
  TT_LOG_INFO("[MooncakeMigrationWorker] '{}' stopping", config_.segment_name);
  teardown();
}

// Reverse-order teardown: stop being discoverable before the engine drops, so
// no in-flight peer write lands on memory we've freed. Idempotent.
void MooncakeMigrationWorker::teardown() {
  // exchange() makes this idempotent even if two threads race here: only the
  // caller that flips true->false performs the single unregister.
  if (memoryRegistered_.exchange(false) && engine_) {
    engine_->unregisterLocalMemory(hostDramPool_.data());
  }
}

// Phase 5 detail: resolve every configured peer through the metadata service
// and cache the handles. Blocks until all peers are found or the discovery
// timeout elapses.
bool MooncakeMigrationWorker::connect() {
  if (!engine_) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] connect: no engine");
    return false;
  }
  if (config_.peer_segment_names.empty()) {
    TT_LOG_WARN("[MooncakeMigrationWorker] connect: no peers configured");
    peers_.clear();
    return true;
  }

  PeerDiscovery discovery(
      PeerDiscoveryConfig{/*poll_interval_ms=*/1000,
                          /*timeout_sec=*/config_.discovery_timeout_sec});
  auto resolved = discovery.resolveAll(*engine_, config_.peer_segment_names);
  if (!resolved) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] connect: discovery timed out before all "
        "peers were reachable");
    return false;
  }

  peers_ = std::move(*resolved);
  TT_LOG_INFO("[MooncakeMigrationWorker] CONNECTED to {} peers", peers_.size());
  return true;
}

// Step 1 (sender): write a known tensor into this galaxy's device DRAM, so the
// data plane has something to migrate. Goes straight through the storage
// backend (UMD for device DRAM).
bool MooncakeMigrationWorker::writeTensorOnSender(
    const std::vector<uint8_t>& tensor) {
  if (config_.role != MigrationRole::Sender) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] writeTensorOnSender called on a non-sender "
        "worker");
    return false;
  }
  if (!engine_ || !engine_->storage()) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] writeTensorOnSender: no engine/storage");
    return false;
  }
  TT_LOG_INFO(
      "[MooncakeMigrationWorker] writeTensorOnSender(bytes={}, device_addr="
      "{:#x})",
      tensor.size(), config_.device_addr);
  return engine_->storage()->writeFrom(config_.device_addr, tensor.data(),
                                       tensor.size());
}

// Step 2 (sender): stage the tensor from device DRAM into a registered host
// buffer, then push it to the receiver's segment over the transport — the
// bounce-buffer flow from mooncake/poc-transfer-engine/adr-mooncake-backend.md.
bool MooncakeMigrationWorker::transferToReceiver() {
  if (config_.role != MigrationRole::Sender) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] transferToReceiver called on a non-sender "
        "worker");
    return false;
  }
  if (!engine_ || !engine_->storage()) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] transferToReceiver: no engine/storage");
    return false;
  }

  // Stage device DRAM -> registered host buffer.
  staging_.assign(config_.tensor_bytes, 0);
  if (!engine_->registerLocalMemory(staging_.data(), staging_.size())) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] transferToReceiver: registerLocalMemory "
        "failed");
    return false;
  }

  bool ok = engine_->storage()->readInto(config_.device_addr, staging_.size(),
                                         staging_.data());
  if (ok) {
    const SegmentHandle peer = engine_->openSegment(config_.peer_segment_name);
    if (peer == kInvalidSegment) {
      TT_LOG_ERROR(
          "[MooncakeMigrationWorker] transferToReceiver: openSegment({}) "
          "failed",
          config_.peer_segment_name);
      ok = false;
    } else {
      TransferRequest request;
      request.op = TransferOp::Write;
      request.local_addr = staging_.data();
      request.target = peer;
      request.target_offset = 0;
      request.length = staging_.size();
      const TransferStatus status = engine_->submitAndWait(request);
      ok = status.state == TransferState::Completed;
    }
  }

  // Best-effort unregister; the transfer result stands regardless.
  engine_->unregisterLocalMemory(staging_.data());
  return ok;
}

// Step 3 (receiver): read this galaxy's device DRAM (where the transfer landed)
// back into a host buffer and byte-compare against the expected tensor.
bool MooncakeMigrationWorker::verifyTensorOnReceiver(
    const std::vector<uint8_t>& expected) {
  if (config_.role != MigrationRole::Receiver) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] verifyTensorOnReceiver called on a "
        "non-receiver worker");
    return false;
  }
  if (!engine_ || !engine_->storage()) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] verifyTensorOnReceiver: no engine/storage");
    return false;
  }

  std::vector<uint8_t> readback(expected.size(), 0);
  if (!engine_->storage()->readInto(config_.device_addr, readback.size(),
                                    readback.data())) {
    TT_LOG_ERROR(
        "[MooncakeMigrationWorker] verifyTensorOnReceiver: readInto failed");
    return false;
  }

  const bool match = readback == expected;
  TT_LOG_INFO(
      "[MooncakeMigrationWorker] verifyTensorOnReceiver(bytes={}, device_addr="
      "{:#x}) -> {}",
      expected.size(), config_.device_addr, match ? "MATCH" : "MISMATCH");
  return match;
}

}  // namespace tt::transport
