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

class PeerDiscoveryService;

/// Which side of the galaxy-to-galaxy transfer this worker plays.
enum class MigrationRole : uint8_t {
  SENDER,    ///< Writes the tensor and pushes it through the transfer engine.
  RECEIVER,  ///< Receives the tensor and verifies it.
};

/**
 * @brief Configuration for the migration-worker spike.
 *
 * Models the issue #3890 setup: an independent process per galaxy that owns a
 * transfer engine (Mooncake transport + a device-DRAM custom backend) and a
 * region of device DRAM identified by NocAddr.
 */
struct MigrationWorkerConfig {
  MigrationRole role = MigrationRole::SENDER;
  std::string peer_segment_name;  ///< Receiver's advertised segment.
  NocAddr device_addr = 0;        ///< Device-DRAM location of the tensor.
  std::size_t tensor_bytes = 0;   ///< Tensor size to move/verify.

  // --- #4294 production bring-up (driven by bringUp()) ---
  std::string metadata_uri;  ///< Discovery service the engine connects to.
  std::string segment_name;  ///< This worker's own advertised logical name.
  TransportProtocol protocol = TransportProtocol::TCP;
  std::size_t host_dram_bytes = 0;  ///< Pool the worker registers/publishes.

  /// Peers this worker discovers on bring-up. Each entry is a logical segment
  /// name resolved through the metadata service; ports are dynamic. The *how*
  /// of discovery (timeout, poll interval) lives in PeerDiscoveryService, not
  /// here — this config only names the peers (the *what*).
  std::vector<std::string> peer_segment_names;

  /// Half-open span of KV cache layers this worker is responsible for:
  /// [layer_start, layer_end). layer_end == 0 means "unset" — the worker owns
  /// every layer (no sharding). Used by ownsLayer() to route requests when a
  /// single request is broadcast to every worker in the role.
  uint32_t layer_start = 0;
  uint32_t layer_end = 0;
};

/**
 * @brief A migration worker that owns its Mooncake lifecycle on a host.
 *
 * Takes an (uninitialised) ITransferEngine and a PeerDiscoveryService by
 * injection, then owns the ordered bring-up that makes it a live, discoverable
 * participant: allocate its host-DRAM pool, init the engine against the
 * metadata service, register/publish that pool, and — by *delegating* to the
 * discovery service — resolve its peers. The phase ordering (register *before*
 * discover, so peers can resolve us in return) is a correctness invariant the
 * worker owns; the discovery mechanism itself is not its concern. Teardown
 * (unregister) happens in reverse order, automatically on destruction.
 *
 * Scope is deliberately *Mooncake only* — the worker has no main loop and no
 * notion of how migration requests arrive. Process orchestration (signal
 * handling, request transport, lifetime) is the binary's job; the worker just
 * provides the migration primitives the binary calls into.
 *
 * It also still drives the original #3890 spike scope (writeTensorOnSender /
 * transferToReceiver / verifyTensorOnReceiver), which operate once the engine
 * is up.
 */
class MooncakeMigrationWorker {
 public:
  MooncakeMigrationWorker(MigrationWorkerConfig config,
                          std::shared_ptr<ITransferEngine> engine,
                          std::shared_ptr<PeerDiscoveryService> discovery);
  ~MooncakeMigrationWorker();

  MooncakeMigrationWorker(const MooncakeMigrationWorker&) = delete;
  MooncakeMigrationWorker& operator=(const MooncakeMigrationWorker&) = delete;

  /**
   * @brief Ordered bring-up: allocate pool → init engine → register/publish →
   *        discover peers. Fail-fast; on failure unwinds whatever earlier
   *        phases set up. Returns true once the worker is READY (engine up,
   *        segment published, all peers connected).
   *
   * The @p cancelToken overload lets discovery be aborted promptly (e.g. on
   * SIGTERM) instead of blocking until the discovery timeout; the parameterless
   * overload is equivalent to a token that never fires.
   */
  bool bringUp();
  bool bringUp(const std::atomic<bool>& cancelToken);

  /// Segment handles for the peers resolved during bringUp() (name -> handle).
  const std::map<std::string, SegmentHandle>& peers() const { return peers_; }

  /// Whether this worker is responsible for @p layerId, i.e. layerId falls in
  /// its configured [layer_start, layer_end) span. An unset span (layer_end ==
  /// 0) means the worker owns every layer, so this returns true for any id —
  /// preserving the non-sharded behaviour when no layer range was given.
  bool ownsLayer(uint32_t layerId) const;

  /// Step 1 (sender): write a known tensor into this galaxy's device DRAM.
  bool writeTensorOnSender(const std::vector<uint8_t>& tensor);

  /// Step 2 (sender): stage from device DRAM and transfer to the receiver.
  bool transferToReceiver();

  /// Step 3 (receiver): read this galaxy's device DRAM and byte-compare.
  bool verifyTensorOnReceiver(const std::vector<uint8_t>& expected);

 private:
  /// Reverse-order teardown; idempotent so bringUp() failure paths and ~dtor
  /// can both call it without double-unregistering.
  void teardown();

  MigrationWorkerConfig config_;
  std::shared_ptr<ITransferEngine> engine_;
  std::shared_ptr<PeerDiscoveryService>
      discovery_;                      ///< How peers are resolved.
  std::vector<uint8_t> hostDramPool_;  ///< Registered/published by bringUp().
  std::vector<uint8_t> staging_;       ///< Spike host staging buffer.
  std::map<std::string, SegmentHandle> peers_;  ///< Resolved by bringUp().
  /// Atomic so concurrent teardown paths (failed bringUp + ~dtor) can't
  /// double-unregister; teardown() flips it with exchange() to stay idempotent.
  std::atomic<bool> memoryRegistered_{false};
};

}  // namespace tt::transport
