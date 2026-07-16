// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_cache_mirror.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief Decode-host (receiver) side of a KV migration.
 *
 * Drives the receiver half of the data plane. At construction it builds
 * one physical mirror over its *full* local table and registers it as the
 * single Mooncake segment the sender writes into — offsets are stable for the
 * receiver's lifetime, so concurrent migrations to disjoint chunks share the
 * one segment safely. For each migration it records which chunks that uuid will
 * touch (prepareMirror) and, once told the writes are done, selectively drains
 * only those chunks back to device DRAM via UMD (drain). It never computes a
 * remote address; the sender owns all addressing.
 *
 * One MooncakeKvReceiver serves one decode host and is shared across every
 * prefill control session on that host — prepareMirror/drain are mutex-guarded
 * for the pending-uuid map. Concurrent RDMA WRITEs into the shared mirror_ are
 * safe only when migrations touch disjoint address ranges (table-driven slot /
 * layer routing must not overlap). Overlapping concurrent migrations are an
 * unstated invariant violation — do not rely on pendingMutex_ to serialize
 * mirror bytes.
 */
class MooncakeKvReceiver {
 public:
  /// @param advertised_segment_name the name the sender opens (this engine's
  ///        server name under P2PHANDSHAKE); returned by prepareMirror.
  /// Builds and registers the full-table mirror once; check registered() to see
  /// whether the segment is live.
  MooncakeKvReceiver(std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
                     std::shared_ptr<const IKvTable> localTable,
                     std::string host, std::string advertisedSegmentName);

  /// Unregisters the mirror segment.
  ~MooncakeKvReceiver();

  MooncakeKvReceiver(const MooncakeKvReceiver&) = delete;
  MooncakeKvReceiver& operator=(const MooncakeKvReceiver&) = delete;

  /// Whether the full-table mirror was registered at construction (false if the
  /// local table held no chunks for this host or registration failed).
  bool registered() const { return registered_; }

  /**
   * @brief Record the chunks `slice` (the destination coordinates) will drain
   *        for `uuid` and advertise the (already-registered) segment. Does not
   *        allocate or register.
   * @return the segment name for the sender to open, or std::nullopt on failure
   *         (mirror not registered, duplicate uuid, or no local chunk in
   * range).
   */
  std::optional<std::string> prepareMirror(const KvSlice& slice, uint64_t uuid);

  /// Selectively drain the prepared migration's chunks mirror -> device.
  ///
  /// Retryable forward recovery: the bytes live in the persistent mirror, so a
  /// failed drain (false) KEEPS the uuid's plan and can be re-driven by a
  /// re-sent DoneMarker with no re-transfer; a successful drain (true) forgets
  /// the uuid. The shared mirror segment stays registered for the receiver's
  /// lifetime. drain() is NOT atomic across chunks: on false the decode device
  /// may hold a partial mix of new and stale KV, so the slot must not be
  /// consumed until a drain succeeds (see README "Contract for a higher-layer
  /// caller").
  /// @return true if every chunk drained; false leaves the migration retryable.
  bool drain(uint64_t uuid);

  /// Number of migrations with a prepared (not yet drained) mirror.
  std::size_t pendingCount() const;

 private:
  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  std::shared_ptr<const IKvTable> local_table_;
  std::string host_;
  std::string advertised_segment_name_;
  KvCacheMirror
      mirror_;  ///< Full-table image, registered once at construction.
  bool registered_ = false;
  mutable std::mutex pendingMutex_;
  std::unordered_map<uint64_t, HostKvPlan>
      pending_;  ///< uuid -> chunks to drain.
};

}  // namespace tt::transport
