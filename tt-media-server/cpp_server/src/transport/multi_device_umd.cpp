// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/multi_device_umd.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

void MultiDeviceUmd::addDevice(LocalDeviceId device,
                               std::shared_ptr<UmdDeviceAccess> access) {
  devices_[device] = std::move(access);
}

bool MultiDeviceUmd::hasDevice(LocalDeviceId device) const {
  return devices_.find(device) != devices_.end();
}

bool MultiDeviceUmd::read(LocalDeviceId device, NocAddr nocAddr,
                          std::size_t size, void* hostBuffer) {
  const auto it = devices_.find(device);
  if (it == devices_.end() || !it->second) {
    TT_LOG_ERROR("[MultiDeviceUmd] read: no UMD handle for device {}", device);
    return false;
  }
  return it->second->read(nocAddr, size, hostBuffer);
}

bool MultiDeviceUmd::write(LocalDeviceId device, NocAddr nocAddr,
                           const void* hostBuffer, std::size_t size) {
  const auto it = devices_.find(device);
  if (it == devices_.end() || !it->second) {
    TT_LOG_ERROR("[MultiDeviceUmd] write: no UMD handle for device {}", device);
    return false;
  }
  return it->second->write(nocAddr, hostBuffer, size);
}

}  // namespace tt::transport
