// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_control_message.hpp"
#include "transport/kv_staging_pool.hpp"  // KvStagingPool, defaultStagingBytes
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

class WorkerHealth;

/**
 * @brief Emits one filled window to the receiver and blocks until it is
 * drained.
 *
 * The data-plane sender is channel-free (like MooncakeKvSender::transferSlot);
 * the orchestrator supplies this sink to send a WindowReady over the control
 * channel and await the WindowAck. It returns true iff the receiver drained the
 * window and its slots (credits) are free to reuse — false aborts the
 * migration.
 */
using WindowSink = std::function<bool(
    uint64_t uuid, const std::vector<BounceSectionDescriptor>&)>;

/**
 * @brief Prefill-host (sender) side of an RDMA-over-host KV migration.
 *
 * It plans the slot — pair src/dst chunks by (layer, ordinal), fan out to each
 * decode replica, merge source-contiguous runs — with each chunk's destination
 * a bounce section. It streams the slot through the bounce buffer a window at a
 * time: fill up to `section_count` free sections (device read -> staging ->
 * one-sided WRITE into the section), emit the window's descriptors via the
 * sink, and wait for the credits back before reusing slots. Each merged run is
 * capped at one bounce section; its destination device coordinates travel in
 * the descriptor.
 */
class MooncakeKvSender {
 public:
  /// @param staging optional pre-registered staging pool shared across senders;
  ///        lazily self-created if null.
  /// @param health optional observability hook (transfer-failure counters).
  MooncakeKvSender(std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
                   std::shared_ptr<const IKvTable> prefillTable,
                   std::shared_ptr<const IKvTable> decodeTable,
                   std::string prefillHost, std::string decodeHost,
                   std::shared_ptr<KvStagingPool> staging = nullptr,
                   WorkerHealth* health = nullptr);

  /**
   * @brief Transfer one slot's chunks through the receiver's bounce buffer.
   * @param uuid         migration id (threaded to the sink for the wire).
   * @param request      what to migrate (slot, layer/position ranges).
   * @param segmentName  the receiver's bounce-buffer segment (from
   * BounceReady).
   * @param geometry     the bounce geometry the receiver advertised.
   * @param sink         sends a filled window and returns when it is drained.
   * @return true iff every chunk transferred and every window was acked.
   */
  bool transferSlot(uint64_t uuid, const MigrationRequest& request,
                    const std::string& segmentName,
                    const BounceGeometry& geometry, const WindowSink& sink);

 private:
  void refreshPeerSegment(const std::string& segmentName);

  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  WorkerHealth* health_ = nullptr;
  std::shared_ptr<const IKvTable> prefill_table_;
  std::shared_ptr<const IKvTable> decode_table_;
  std::string prefill_host_;
  std::string decode_host_;
  std::shared_ptr<KvStagingPool> staging_;
};

}  // namespace tt::transport
