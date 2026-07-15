// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_cache_layout.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

class WorkerHealth;

/// Default size of ONE staging buffer in a KvStagingPool. Two are held (for
/// double-buffering), so a pool holds 2 * this. Sized to be a large window that
/// batches many chunks; a slot larger than a window is transferred in several.
inline constexpr uint64_t K_DEFAULT_STAGING_BYTES = 32ull * 1024 * 1024;

/// Default window-count divisor (see stagingWindowDivisor / transferSlot).
inline constexpr uint32_t K_DEFAULT_WINDOW_DIVISOR = 4;

/**
 * @brief Per-buffer staging size in bytes (tunable).
 *
 * Env `TT_MOONCAKE_STAGING_BYTES` overrides; falls back to
 * K_DEFAULT_STAGING_BYTES. This is the hard ceiling on one batch and, x2, the
 * sender's peak host memory. Exposed so a benchmark can sweep it.
 */
uint64_t defaultStagingBytes();

/**
 * @brief How many windows to target per slot (tunable).
 *
 * The sender sizes each window ~= totalBytes / divisor (capped at the staging
 * buffer, floored at one chunk), so there are multiple windows for the
 * double-buffer to overlap. Larger divisor => more, smaller batches (finer
 * overlap); 1 => a single batch of the whole slice (up to the buffer). Env
 * `TT_MOONCAKE_WINDOW_DIVISOR` overrides; falls back to
 * K_DEFAULT_WINDOW_DIVISOR (min 1).
 */
uint32_t stagingWindowDivisor();

/**
 * @brief A set of registered host staging buffers, allocated and registered
 *        with the transfer engine ONCE and reused across every migration.
 *
 * Memory registration is far too costly to repeat per slot — on RDMA it pins
 * pages + builds an ibv_reg_mr, routinely ms-scale for tens of MiB. So the pool
 * registers its buffers at construction and unregisters them at destruction
 * (RAII), and MooncakeKvSender stages through them without touching
 * register/unregister on the hot path (mirroring MooncakeKvReceiver, whose
 * mirror is likewise registered once).
 *
 * Holds two buffers so the sender can double-buffer (stage buffer N+1 while
 * buffer N transfers). NOT thread-safe: a pool must be driven by one
 * transferSlot at a time. That holds today because migrations run serially
 * (MooncakeMigrationExecutor) and the multi-host fan-out is serial, which is
 * why one pool can be SHARED across a prefill host's per-decode-host senders
 * instead of each holding its own (N * 2 * 32 MiB). If the fan-out is ever
 * parallelized, concurrent legs need distinct pools.
 */
class KvStagingPool {
 public:
  static constexpr int kBuffers = 2;

  /// Registers `kBuffers` buffers of `bufferBytes` each with `engine` (defaults
  /// to the tunable defaultStagingBytes()). Check registered() — a false there
  /// means the sender must fail the migration.
  explicit KvStagingPool(std::shared_ptr<ITransferEngine> engine,
                         uint64_t bufferBytes = defaultStagingBytes());
  ~KvStagingPool();

  KvStagingPool(const KvStagingPool&) = delete;
  KvStagingPool& operator=(const KvStagingPool&) = delete;

  bool registered() const { return registered_; }
  uint64_t bufferBytes() const { return buffers_[0].size(); }
  uint8_t* buffer(int i) { return buffers_[i].data(); }

 private:
  std::shared_ptr<ITransferEngine> engine_;
  std::array<std::vector<uint8_t>, kBuffers> buffers_;
  int registered_count_ = 0;
  bool registered_ = false;
};

/**
 * @brief Prefill-host (sender) side of a KV migration.
 *
 * Drives the sender half of the data plane. For each chunk of the slot it:
 *   1. reads the bytes from its local (prefill) device DRAM (one source
 *      replica), and
 *   2. computes the full destination addressing from the decode table it
 *      received at init — the mirror offset for each decode replica — and
 * pushes the bytes there with a one-sided Mooncake WRITE (the fan-out).
 *
 * Because the sender owns all destination addressing (mirror offset today, the
 * decode device NocAddr already known for future RDMA-direct), the receiver
 * only drains its mirror — see MooncakeKvReceiver.
 *
 * Holds both tables: the prefill table (local, for reads) and the decode table
 * (remote, exchanged at init, for destination offsets).
 */
class MooncakeKvSender {
 public:
  /// @param staging optional pre-registered staging pool to reuse across
  ///        senders (the multi-host fan-out shares one). If null, the sender
  ///        lazily creates and owns its own on first transferSlot.
  /// @param health optional; when set, a transfer failure bumps the transfer/
  ///        re-resolve counters (pure observability, never gates readiness).
  MooncakeKvSender(std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
                   std::shared_ptr<const IKvTable> prefillTable,
                   std::shared_ptr<const IKvTable> decodeTable,
                   std::string prefillHost, std::string decodeHost,
                   std::shared_ptr<KvStagingPool> staging = nullptr,
                   WorkerHealth* health = nullptr);

  /**
   * @brief Transfer one slot's chunks into the decode mirror segment.
   * @param request      what to migrate (slot, layer/position ranges).
   * @param segment_name the receiver's advertised segment (from MirrorReady).
   * @return true if every chunk read + wrote successfully.
   */
  bool transferSlot(const MigrationRequest& request,
                    const std::string& segmentName);

 private:
  // Force-refresh the peer's segment descriptor after a failed transfer so the
  // NEXT request re-resolves its current address. A TCP sender caches the
  // descriptor at openSegment() time, so a peer that restarted on a fresh
  // dynamic port would otherwise be targeted at its dead address forever.
  void refreshPeerSegment(const std::string& segmentName);

  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  WorkerHealth* health_ = nullptr;
  std::shared_ptr<const IKvTable> prefill_table_;
  std::shared_ptr<const IKvTable> decode_table_;
  std::string prefill_host_;
  std::string decode_host_;
  // Destination addressing built once from the *full* decode table, so the
  // mirror offsets are byte-identical to the receiver's full-table mirror and
  // stable across migrations (see MooncakeKvReceiver / allHostLocations).
  KvCacheLayout dst_layout_;
  // Registered staging buffers, reused across migrations. Injected (shared with
  // sibling senders) or lazily self-created on first transferSlot.
  std::shared_ptr<KvStagingPool> staging_;
};

}  // namespace tt::transport
