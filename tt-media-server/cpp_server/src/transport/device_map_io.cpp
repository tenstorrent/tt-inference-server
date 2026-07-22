// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/device_map_io.hpp"

#include <fstream>

#include "utils/logger.hpp"

namespace tt::transport {

DeviceMap loadDeviceMapStream(std::istream& input) {
  DeviceMap deviceMap;
  uint32_t mesh = 0;
  uint32_t chip = 0;
  uint64_t umd = 0;
  while (input >> mesh >> chip >> umd) {
    deviceMap.set(FabricNode{mesh, chip}, umd);
  }
  return deviceMap;
}

std::optional<DeviceMap> loadDeviceMapFile(const std::string& path) {
  if (path.empty()) {
    return DeviceMap{};
  }
  std::ifstream file(path);
  if (!file.good()) {
    TT_LOG_ERROR("[device_map_io] cannot open device map file: {}", path);
    return std::nullopt;
  }
  DeviceMap deviceMap = loadDeviceMapStream(file);
  TT_LOG_INFO("[device_map_io] loaded {} entries from {}", deviceMap.size(),
              path);
  return deviceMap;
}

}  // namespace tt::transport
