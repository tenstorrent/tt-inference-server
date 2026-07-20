// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/double_pinned_buffer.hpp"

#include <cstdlib>
#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

DoublePinnedBuffer::DoublePinnedBuffer(std::shared_ptr<ITransferEngine> engine,
                                       std::size_t bytes,
                                       const DeviceMapFn& deviceMap)
    : engine_(std::move(engine)), size_(bytes) {
  if (bytes == 0) return;
  // aligned_alloc requires the size be a multiple of the alignment.
  capacity_ = (bytes + kAlign - 1) & ~(kAlign - 1);
  base_ = static_cast<uint8_t*>(std::aligned_alloc(kAlign, capacity_));
  if (base_ == nullptr) {
    TT_LOG_ERROR("[DoublePinnedBuffer] aligned_alloc({} bytes) failed",
                 capacity_);
    return;
  }
  if (!engine_ || !engine_->registerLocalMemory(base_, capacity_)) {
    TT_LOG_ERROR("[DoublePinnedBuffer] registerLocalMemory({} bytes) failed",
                 capacity_);
    return;  // registered_ stays false; buffer freed in dtor
  }
  registered_ = true;
  // NOC-map the SAME buffer for device (DRISC) DMA, so a device read/write
  // whose host buffer is this region DMAs directly (no bounce). One allocation
  // = one NOC mapping (KMD caps NOC-mapped buffers at 16/chip — never map per
  // slot).
  if (deviceMap) {
    deviceMap(base_, capacity_);
  }
}

DoublePinnedBuffer::~DoublePinnedBuffer() {
  if (registered_ && engine_) {
    engine_->unregisterLocalMemory(base_);
  }
  // The device NOC mapping has no per-region unmap; it is released when the
  // DriscDeviceIo's links are destroyed (which must happen after this buffer).
  if (base_ != nullptr) std::free(base_);
}

}  // namespace tt::transport
