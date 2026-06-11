// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_worker.hpp"

#include <utility>

#include "transport/i_storage_backend.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

MooncakeMigrationWorker::MooncakeMigrationWorker(
    MigrationWorkerConfig config, std::shared_ptr<ITransferEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {}

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
