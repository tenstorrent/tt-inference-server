// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "transport/double_pinned_buffer.hpp"
#include "transport/i_transfer_engine.hpp"

namespace tt::transport {

/// Default size of ONE staging buffer in a KvStagingPool. Two are held (for
/// double-buffering), so a pool holds 2 * this. Sized to be a large window that
/// batches many chunks; a slot larger than a window is transferred in several.
inline constexpr uint64_t K_DEFAULT_STAGING_BYTES = 32ull * 1024 * 1024;

/// Default window-count divisor (see stagingWindowDivisor).
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
 * Env `TT_MOONCAKE_WINDOW_DIVISOR` overrides; falls back to
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
 * (RAII), and the sender stages through them without touching register/
 * unregister on the hot path.
 *
 * Each buffer is a DoublePinnedBuffer: engine-registered (ibv_reg_mr) and, when
 * a `deviceMap` is supplied (DRISC), also NOC-mapped so device reads DMA
 * straight into staging.
 *
 * Holds two buffers so the sender can double-buffer (stage buffer N+1 while
 * buffer N transfers). NOT thread-safe: a pool must be driven by one transfer
 * at a time. The multi-host fan-out is serial, so one pool can be SHARED across
 * a prefill host's per-decode-host senders; if the fan-out is ever
 * parallelized, concurrent legs need distinct pools.
 */
class KvStagingPool {
 public:
  static constexpr int kBuffers = 2;

  /// Registers `kBuffers` buffers of `bufferBytes` each with `engine` (defaults
  /// to the tunable defaultStagingBytes()). `deviceMap` (optional, DRISC) also
  /// NOC-maps each buffer. Check registered() — a false there means the sender
  /// must fail the migration.
  explicit KvStagingPool(std::shared_ptr<ITransferEngine> engine,
                         uint64_t bufferBytes = defaultStagingBytes(),
                         DeviceMapFn deviceMap = {});
  ~KvStagingPool();

  KvStagingPool(const KvStagingPool&) = delete;
  KvStagingPool& operator=(const KvStagingPool&) = delete;

  bool registered() const { return registered_; }
  uint64_t bufferBytes() const { return buffers_[0] ? buffers_[0]->size() : 0; }
  uint8_t* buffer(int i) { return buffers_[i] ? buffers_[i]->base() : nullptr; }

 private:
  std::array<std::unique_ptr<DoublePinnedBuffer>, kBuffers> buffers_;
  bool registered_ = false;
};

}  // namespace tt::transport
