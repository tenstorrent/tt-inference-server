// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
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

  // --- #4294 production bring-up (driven by bringUp()) ---
  std::string metadata_uri;  ///< Discovery service the engine connects to.
  std::string segment_name;  ///< This worker's own advertised logical name.
  TransportProtocol protocol = TransportProtocol::Tcp;
  std::size_t host_dram_bytes = 0;  ///< Pool the worker registers/publishes.

  /// Peers this worker discovers on bring-up. Each entry is a logical segment
  /// name resolved through the metadata service; ports are dynamic.
  std::vector<std::string> peer_segment_names;
  int discovery_timeout_sec = 30;  ///< Discovery gives up after this long.
};

/**
 * @brief A migration worker that owns its full lifecycle on a host.
 *
 * The worker takes an (uninitialised) ITransferEngine by injection and then
 * owns the ordered bring-up that makes it a live, discoverable participant
 * (#4294): allocate its host-DRAM pool, init the engine against the metadata
 * service, register/publish that pool, and discover its peers. The phase
 * ordering — register *before* connect, so peers can resolve us in return — is
 * a correctness invariant enforced here, not by the caller. Teardown
 * (unregister) happens in reverse order, including on destruction.
 *
 * It also still drives the original #3890 spike scope (writeTensorOnSender /
 * transferToReceiver / verifyTensorOnReceiver), which operate once the engine
 * is up.
 */
class MooncakeMigrationWorker {
 public:
  MooncakeMigrationWorker(MigrationWorkerConfig config,
                          std::shared_ptr<ITransferEngine> engine);
  ~MooncakeMigrationWorker();

  MooncakeMigrationWorker(const MooncakeMigrationWorker&) = delete;
  MooncakeMigrationWorker& operator=(const MooncakeMigrationWorker&) = delete;

  /**
   * @brief Ordered bring-up: allocate pool → init engine → register/publish →
   *        discover peers. Fail-fast; on failure unwinds whatever earlier
   *        phases set up. Returns true once the worker is READY (engine up,
   *        segment published, all peers connected).
   */
  bool bringUp();

  /**
   * @brief Block until @p stopRequested is set (the hold-until-SIGTERM phase),
   *        then tear down. The stop source (e.g. a signal handler) is owned by
   *        the caller; the worker only observes it.
   */
  void run(const std::atomic<bool>& stopRequested);

  /// Segment handles for the peers resolved during bringUp() (name -> handle).
  const std::map<std::string, SegmentHandle>& peers() const { return peers_; }

  /// Step 1 (sender): write a known tensor into this galaxy's device DRAM.
  bool writeTensorOnSender(const std::vector<uint8_t>& tensor);

  /// Step 2 (sender): stage from device DRAM and transfer to the receiver.
  bool transferToReceiver();

  /// Step 3 (receiver): read this galaxy's device DRAM and byte-compare.
  bool verifyTensorOnReceiver(const std::vector<uint8_t>& expected);

 private:
  /// Phase 5: discover every configured peer, caching handles in peers_.
  bool connect();
  /// Reverse-order teardown; idempotent so run() and ~dtor can both call it.
  void teardown();

  MigrationWorkerConfig config_;
  std::shared_ptr<ITransferEngine> engine_;
  std::vector<uint8_t> hostDramPool_;  ///< Registered/published by bringUp().
  std::vector<uint8_t> staging_;       ///< Spike host staging buffer (#3890).
  std::map<std::string, SegmentHandle> peers_;  ///< Resolved by bringUp().
  bool memoryRegistered_ = false;
};

}  // namespace tt::transport
