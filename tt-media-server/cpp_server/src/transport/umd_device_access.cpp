// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/umd_device_access.hpp"

#include <utility>

#include "utils/logger.hpp"

#ifdef USE_METAL_CPP_LIB
#include <cstdint>
#include <exception>
#include <span>

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_metal.hpp"
#endif

namespace tt::transport {

#ifdef USE_METAL_CPP_LIB

// Real UMD backend: owns a tt-metal IDevice and stages bytes between a host
// buffer and a DRAM channel via tt_metal's ReadFromDeviceDRAMChannel /
// WriteToDeviceDRAMChannel. A NocAddr's local_addr is passed through as the
// channel-local byte offset (no rebasing here — keeping a firmware-reserved
// low-DRAM region clear is a higher-level concern, as in the migration layer's
// UmdDeviceReader/Writer).
struct UmdDeviceAccess::Impl {
  int deviceId = 0;
  tt::tt_metal::IDevice* device = nullptr;
};

UmdDeviceAccess::UmdDeviceAccess(int deviceId)
    : impl_(std::make_unique<Impl>()) {
  impl_->deviceId = deviceId;
  try {
    impl_->device = tt::tt_metal::CreateDevice(deviceId);
    const uint64_t dramBase =
        impl_->device->allocator()->get_base_allocator_addr(
            tt::tt_metal::HalMemType::DRAM);
    TT_LOG_INFO(
        "[UmdDeviceAccess] opened deviceId={} ({} DRAM channels, "
        "allocatable_base=0x{:x})",
        deviceId, impl_->device->num_dram_channels(), dramBase);
  } catch (const std::exception& e) {
    impl_->device = nullptr;
    TT_LOG_ERROR(
        "[UmdDeviceAccess] failed to open deviceId={}: {}. read/write will "
        "report failure.",
        deviceId, e.what());
  }
}

UmdDeviceAccess::~UmdDeviceAccess() {
  if (impl_ && impl_->device != nullptr) {
    try {
      tt::tt_metal::CloseDevice(impl_->device);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[UmdDeviceAccess] CloseDevice(deviceId={}) failed: {}",
                   impl_->deviceId, e.what());
    }
    impl_->device = nullptr;
  }
}

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
  try {
    std::span<uint8_t> dst(static_cast<uint8_t*>(hostBuffer), size);
    return tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
        impl_->device, static_cast<int>(nocChannel(addr)), nocLocalAddr(addr),
        dst);
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
  try {
    std::span<const uint8_t> src(static_cast<const uint8_t*>(hostBuffer), size);
    return tt::tt_metal::detail::WriteToDeviceDRAMChannel(
        impl_->device, static_cast<int>(nocChannel(addr)), nocLocalAddr(addr),
        src);
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
