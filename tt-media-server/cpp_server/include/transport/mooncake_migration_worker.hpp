// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "transport/i_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/// Which side of the galaxy-to-galaxy transfer this worker plays.
enum class MigrationRole : uint8_t {
  Sender,    ///< Writes the tensor and pushes it through the transfer engine.
  Receiver,  ///< Receives the tensor and verifies it.
};

/**
 * @brief Configuration for the migration-worker spike.
 *
 * Models the issue #3890 setup: an independent process per galaxy that owns a
 * transfer engine (Mooncake transport + a device-DRAM custom backend) and a
 * region of device DRAM identified by NocAddr.
 */
struct MigrationWorkerConfig {
  MigrationRole role = MigrationRole::Sender;
  std::string peer_segment_name;  ///< Receiver's advertised segment.
  NocAddr device_addr = 0;        ///< Device-DRAM location of the tensor.
  std::size_t tensor_bytes = 0;   ///< Tensor size to move/verify.
};

/**
 * @brief Migration worker for the #3890 spike — the independent C++ process
 *        that proves a tensor can move from one galaxy's device DRAM to
 *        another's through the Transfer Engine with the custom UMD backend.
 *
 * Drives the three scope items from issue #3890 directly:
 *   1. writeTensorOnSender()    — "Write tensor on sender galaxy"
 *   2. transferToReceiver()     — "Use transfer engine with custom backend to
 *                                  transfer it to receiver galaxy"
 *   3. verifyTensorOnReceiver() — "Verify tensor is correct on receiver galaxy"
 *
 * Holds an ITransferEngine (expected to be a MooncakeTransferEngine wrapping a
 * DeviceDramStorageBackend) plus the registered host staging buffer. Staging
 * and verification go through the engine's storage backend; the host->host hop
 * goes through the engine's transport. The full two-galaxy run requires a live
 * peer (the engine must be init()'d and the peer segment advertised); without
 * one, transferToReceiver() reports failure.
 */
class MooncakeMigrationWorker {
 public:
  MooncakeMigrationWorker(MigrationWorkerConfig config,
                          std::shared_ptr<ITransferEngine> engine);

  /// Step 1 (sender): write a known tensor into this galaxy's device DRAM.
  bool writeTensorOnSender(const std::vector<uint8_t>& tensor);

  /// Step 2 (sender): stage from device DRAM and transfer to the receiver.
  bool transferToReceiver();

  /// Step 3 (receiver): read this galaxy's device DRAM and byte-compare.
  bool verifyTensorOnReceiver(const std::vector<uint8_t>& expected);

 private:
  MigrationWorkerConfig config_;
  std::shared_ptr<ITransferEngine> engine_;
  std::vector<uint8_t> staging_;  ///< Registered host staging buffer.
};

}  // namespace tt::transport
