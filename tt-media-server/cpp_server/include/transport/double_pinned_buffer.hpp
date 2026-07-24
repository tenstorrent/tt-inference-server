// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "transport/i_transfer_engine.hpp"

namespace tt::transport {

/**
 * @brief NOC-maps a host region for device (DRISC) DMA. Register-only: the
 * DRISC NOC mapping is released with the device links' lifetime, not
 * per-region, so the mapped buffer must outlive the DriscDeviceIo.
 *
 * DriscDeviceIo::registerHostRegion is the production source; a recording
 * lambda stands in for tests. Empty/null => host/MMIO mode: the buffer is
 * engine-only.
 */
using DeviceMapFn = std::function<void(void* va, std::size_t bytes)>;

/**
 * @brief A host buffer pinned for BOTH transports of RDMA-over-host.
 *
 * The keystone integration: ONE contiguous host buffer that is simultaneously
 *   (a) registered with the transfer engine — `ibv_reg_mr` for the RDMA NIC,
 * and (b) NOC-mapped via the device registrar — so the TT chip's DRISC engine
 * can DMA device DRAM into/out of it (zero-copy, no bounce memcpy). The same VA
 * is pinned by both the NIC and the chip — the thing that makes device DRAM ->
 * host bounce buffer -> RDMA -> host bounce buffer -> device DRAM a single
 * buffer on each side.
 *
 * Page-aligned (>= the 64-byte DRISC-DMA payload contract; matches the
 * disaggregation NOC-mapped arenas' 4 KiB alignment). ONE allocation == ONE
 * NOC mapping, honoring the KMD's 16-NOC-mapped-buffers-per-chip cap (never map
 * per slot). RAII unregisters the engine side on destruction; the device NOC
 * map has no explicit unmap (see DeviceMapFn) — this buffer must outlive the
 * device links.
 */
class DoublePinnedBuffer {
 public:
  /// Page alignment (satisfies the 64 B DRISC payload contract and IOMMU/NOC
  /// mapping; the disaggregation host arenas use the same 4 KiB).
  static constexpr std::size_t kAlign = 4096;

  /// Allocate `bytes` (rounded up to kAlign) page-aligned, register the whole
  /// allocation with `engine`, and — if `deviceMap` is set — NOC-map it too.
  /// Check registered(): false means the engine rejected it (fail the caller).
  DoublePinnedBuffer(std::shared_ptr<ITransferEngine> engine, std::size_t bytes,
                     const DeviceMapFn& deviceMap = {});
  ~DoublePinnedBuffer();

  DoublePinnedBuffer(const DoublePinnedBuffer&) = delete;
  DoublePinnedBuffer& operator=(const DoublePinnedBuffer&) = delete;

  uint8_t* base() { return base_; }
  const uint8_t* base() const { return base_; }

  /// Requested (logical) size.
  std::size_t size() const { return size_; }
  /// Actual allocation (size rounded up to kAlign) — what is registered/mapped.
  std::size_t capacity() const { return capacity_; }
  /// Whether the engine registration succeeded.
  bool registered() const { return registered_; }

 private:
  std::shared_ptr<ITransferEngine> engine_;
  uint8_t* base_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
  bool registered_ = false;
};

}  // namespace tt::transport
