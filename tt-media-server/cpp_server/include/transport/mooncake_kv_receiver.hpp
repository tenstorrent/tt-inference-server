// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "transport/double_pinned_buffer.hpp"
#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_control_message.hpp"

namespace tt::transport {

/**
 * @brief Decode-host (receiver) side of an RDMA-over-host KV migration.
 *
 * At construction it allocates a small bounce buffer (BounceGeometry) and
 * registers it as the single Mooncake
 * segment the sender writes into — tens of MiB, not the whole KV region, so
 * `ibv_reg_mr` pins almost nothing. It holds NO table: every window's
 * descriptors carry the destination device coordinates, so the sender still
 * owns all addressing and the receiver just copies each drained section to the
 * device address it is told (drainWindow), bounds-checking the section against
 * the bounce buffer. Draining happens window by window as WindowReady messages
 * arrive, so the receiver device-write overlaps the sender's next-window
 * network transfer.
 */
class MooncakeKvReceiver {
 public:
  /// @param advertisedSegmentName the name the sender opens (this engine's
  ///        server name under P2PHANDSHAKE); returned in BounceReady.
  /// @param deviceMap when set (DRISC path), the bounce buffer is NOC-mapped
  /// after engine
  ///        registration so the drain DMAs straight from the bounce buffer to
  ///        device DRAM — the SAME buffer the engine ibv_reg_mr's. Null =>
  ///        host/MMIO mode (engine registration only).
  /// Allocates + registers the bounce buffer once; check registered() for
  /// liveness.
  MooncakeKvReceiver(std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
                     std::string advertisedSegmentName,
                     BounceGeometry geometry = defaultBounceGeometry(),
                     DeviceMapFn deviceMap = {});

  /// Unregisters the bounce-buffer segment.
  ~MooncakeKvReceiver();

  MooncakeKvReceiver(const MooncakeKvReceiver&) = delete;
  MooncakeKvReceiver& operator=(const MooncakeKvReceiver&) = delete;

  /// Whether the bounce buffer was registered at construction.
  bool registered() const { return registered_; }

  /// The segment name the sender opens.
  const std::string& segmentName() const { return advertised_segment_name_; }

  /// Geometry advertised to the sender in BounceReady.
  BounceGeometry geometry() const { return buffer_.geometry(); }

  /**
   * @brief Drain one window: copy each descriptor's section bytes to every
   * device target it lists (replica fan-out on this host).
   *
   * Not atomic across targets: on false the device may hold a partial mix, so
   * the caller must not consider the migration complete. A descriptor whose
   * section escapes the bounce buffer is rejected (returns false for that
   * window) — the one essential guard against an out-of-bounds sender
   * descriptor.
   * @return true iff every target of every descriptor wrote successfully.
   */
  bool drainWindow(const std::vector<BounceSectionDescriptor>& window);

 private:
  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  std::string advertised_segment_name_;
  KvBounceBuffer buffer_;
  DeviceMapFn device_map_;
  bool registered_ = false;
};

}  // namespace tt::transport
