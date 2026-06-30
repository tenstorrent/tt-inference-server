// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_worker.hpp"

#include <utility>

#include "transport/i_storage_backend.hpp"
#include "transport/peer_discovery_service.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

MooncakeMigrationWorker::MooncakeMigrationWorker(
    MigrationWorkerConfig config, std::shared_ptr<ITransferEngine> engine,
    std::shared_ptr<PeerDiscoveryService> discovery)
    : config_(std::move(config)),
      engine_(std::move(engine)),
      discovery_(std::move(discovery)) {}

MooncakeMigrationWorker::~MooncakeMigrationWorker() { teardown(); }

// Convenience overload: bring up with a cancel token that never fires.
bool MooncakeMigrationWorker::bringUp() {
  static const std::atomic<bool> never{false};
  return bringUp(never);
}

// Ordered bring-up. Each phase only proceeds if the previous one
// succeeded.
bool MooncakeMigrationWorker::bringUp(const std::atomic<bool>& cancelToken) {
  if (!engine_) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] bringUp: no engine");
    return false;
  }
  if (!discovery_) {
    TT_LOG_ERROR("[MooncakeMigrationWorker] bringUp: no discovery service");
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

  // Phase 5: discover peers — the readiness gate. The worker owns *when* this
  // happens (after publish, so peers can resolve us back) and forwards the
  // cancel token so a stop request aborts discovery promptly; the injected
  // PeerDiscoveryService owns *how*.
  auto resolved =
      discovery_->discover(*engine_, config_.peer_segment_names, &cancelToken);
  if (!resolved) {
    teardown();
    return false;
  }
  peers_ = std::move(*resolved);
  return true;
}

// Reverse-order teardown: stop being discoverable before the engine drops, so
// no in-flight peer write lands on memory we've freed. Idempotent.
void MooncakeMigrationWorker::teardown() {
  // exchange() makes this idempotent even if two threads race here: only the
  // caller that flips true->false performs the single unregister.
  if (memoryRegistered_.exchange(false) && engine_) {
    if (!engine_->unregisterLocalMemory(hostDramPool_.data())) {
      TT_LOG_WARN(
          "[MooncakeMigrationWorker] '{}' unregisterLocalMemory failed during "
          "teardown; segment may linger in the metadata service",
          config_.segment_name);
    }
  }
}

// Step 1 (sender): write a known tensor into this galaxy's device DRAM, so the
// data plane has something to migrate. Goes straight through the storage
// backend (UMD for device DRAM).
bool MooncakeMigrationWorker::writeTensorOnSender(
    const std::vector<uint8_t>& tensor) {
  if (config_.role != MigrationRole::SENDER) {
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
  if (config_.role != MigrationRole::SENDER) {
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
    if (peer == K_INVALID_SEGMENT) {
      TT_LOG_ERROR(
          "[MooncakeMigrationWorker] transferToReceiver: openSegment({}) "
          "failed",
          config_.peer_segment_name);
      ok = false;
    } else {
      TransferRequest request;
      request.op = TransferOp::WRITE;
      request.local_addr = staging_.data();
      request.target = peer;
      request.target_offset = 0;
      request.length = staging_.size();
      const TransferStatus status = engine_->submitAndWait(request);
      ok = status.state == TransferState::COMPLETED;
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
  if (config_.role != MigrationRole::RECEIVER) {
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
