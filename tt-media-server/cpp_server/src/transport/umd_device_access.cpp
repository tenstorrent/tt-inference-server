// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/umd_device_access.hpp"

#include <utility>

#include "utils/logger.hpp"

#ifdef USE_METAL_CPP_LIB
#include <cstdint>
#include <exception>

// Coexistence device access: the disaggregation UmdDevice opens the chip via
// the raw UMD Cluster WITHOUT start_device(), so it takes no CHIP_IN_USE flock
// and builds no dispatch firmware — letting the migration worker share the chip
// with a live inference engine. (Replaces the old tt_metal::CreateDevice path,
// which took exclusive chip ownership and JIT-built dispatch kernels.)
#include "device_io.hpp"
#endif

namespace tt::transport {

#ifdef USE_METAL_CPP_LIB

namespace {
namespace dis = tt::tt_metal::experimental::disaggregation::detail;
}  // namespace

// Real coexistence UMD backend: owns a disaggregation UmdDevice, which opens
// the chip via the raw UMD Cluster WITHOUT start_device() — no CHIP_IN_USE
// flock, no dispatch firmware — so it coexists with a live inference engine on
// the same chip. A NocAddr's channel selects the Metal dram_view; local_addr is
// the intra-view byte offset (no rebasing here, as in the migration layer's
// UmdDeviceReader/Writer).
struct UmdDeviceAccess::Impl {
  int deviceId = 0;
  std::unique_ptr<dis::UmdDevice> device;
};

UmdDeviceAccess::UmdDeviceAccess(int deviceId)
    : impl_(std::make_unique<Impl>()) {
  impl_->deviceId = deviceId;
  try {
    impl_->device = dis::UmdDevice::open(deviceId);
    TT_LOG_INFO(
        "[UmdDeviceAccess] opened deviceId={} ({} DRAM channels) via "
        "coexistence UMD (no start_device/flock)",
        deviceId, impl_->device->num_dram_channels());
  } catch (const std::exception& e) {
    impl_->device = nullptr;
    TT_LOG_ERROR(
        "[UmdDeviceAccess] failed to open deviceId={}: {}. read/write will "
        "report failure.",
        deviceId, e.what());
  }
}

UmdDeviceAccess::~UmdDeviceAccess() = default;

UmdDeviceAccess::UmdDeviceAccess(UmdDeviceAccess&&) noexcept = default;

UmdDeviceAccess& UmdDeviceAccess::operator=(UmdDeviceAccess&&) noexcept =
    default;

bool UmdDeviceAccess::read(NocAddr addr, std::size_t size, void* hostBuffer) {
  if (impl_->device == nullptr) {
    TT_LOG_WARN("[UmdDeviceAccess] read on closed/unopened deviceId={}",
                impl_->deviceId);
    return false;
  }
  if (size == 0) {
    return true;
  }
  if (hostBuffer == nullptr) {
    TT_LOG_ERROR("[UmdDeviceAccess] read with null hostBuffer (size={})", size);
    return false;
  }
  if (size > UINT32_MAX) {
    TT_LOG_ERROR("[UmdDeviceAccess] read size {} exceeds UMD 32-bit limit",
                 size);
    return false;
  }
  try {
    impl_->device->read_dram(static_cast<uint32_t>(nocChannel(addr)),
                             nocLocalAddr(addr), static_cast<uint32_t>(size),
                             hostBuffer);
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[UmdDeviceAccess] read(channel={}, local_addr={}, size={}) failed: {}",
        nocChannel(addr), nocLocalAddr(addr), size, e.what());
    return false;
  }
}

bool UmdDeviceAccess::write(NocAddr addr, const void* hostBuffer,
                            std::size_t size) {
  if (impl_->device == nullptr) {
    TT_LOG_WARN("[UmdDeviceAccess] write on closed/unopened deviceId={}",
                impl_->deviceId);
    return false;
  }
  if (size == 0) {
    return true;
  }
  if (hostBuffer == nullptr) {
    TT_LOG_ERROR("[UmdDeviceAccess] write with null hostBuffer (size={})",
                 size);
    return false;
  }
  if (size > UINT32_MAX) {
    TT_LOG_ERROR("[UmdDeviceAccess] write size {} exceeds UMD 32-bit limit",
                 size);
    return false;
  }
  try {
    impl_->device->write_dram(static_cast<uint32_t>(nocChannel(addr)),
                              nocLocalAddr(addr), static_cast<uint32_t>(size),
                              hostBuffer);
    // write_dram() issues posted, Relaxed-ordered TLB writes that are not
    // guaranteed visible when it returns. Barrier so the bytes are committed
    // before we report completion (the receiver's drain consumer reads them).
    // TODO(perf): the receiver drains many chunks per slot; a single flush()
    // after all writes (vs a barrier per chunk) is a future optimization.
    impl_->device->dram_barrier();
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[UmdDeviceAccess] write(channel={}, local_addr={}, size={}) failed: "
        "{}",
        nocChannel(addr), nocLocalAddr(addr), size, e.what());
    return false;
  }
}

#else  // !USE_METAL_CPP_LIB

// Fallback when tt-metal is not part of the build: keep the storage half a
// no-op that reports failure, so transport_lib still builds in every
// configuration (see CMakeLists.txt).
struct UmdDeviceAccess::Impl {
  int deviceId = 0;
};

UmdDeviceAccess::UmdDeviceAccess(int deviceId)
    : impl_(std::make_unique<Impl>()) {
  impl_->deviceId = deviceId;
  TT_LOG_WARN(
      "[UmdDeviceAccess] constructed for deviceId={} but tt-metal is not in "
      "this build (USE_METAL_CPP_LIB undefined); read/write report failure",
      deviceId);
}

UmdDeviceAccess::~UmdDeviceAccess() = default;

UmdDeviceAccess::UmdDeviceAccess(UmdDeviceAccess&&) noexcept = default;

UmdDeviceAccess& UmdDeviceAccess::operator=(UmdDeviceAccess&&) noexcept =
    default;

bool UmdDeviceAccess::read(NocAddr addr, std::size_t size,
                           void* /*hostBuffer*/) {
  TT_LOG_WARN(
      "[UmdDeviceAccess] read(channel={}, local_addr={}, size={}) unavailable "
      "(built without tt-metal)",
      nocChannel(addr), nocLocalAddr(addr), size);
  return false;
}

bool UmdDeviceAccess::write(NocAddr addr, const void* /*hostBuffer*/,
                            std::size_t size) {
  TT_LOG_WARN(
      "[UmdDeviceAccess] write(channel={}, local_addr={}, size={}) unavailable "
      "(built without tt-metal)",
      nocChannel(addr), nocLocalAddr(addr), size);
  return false;
}

#endif  // USE_METAL_CPP_LIB

}  // namespace tt::transport
