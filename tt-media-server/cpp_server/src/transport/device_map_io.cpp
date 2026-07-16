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

DeviceMap loadDeviceMapFile(const std::string& path) {
  if (path.empty()) {
    return {};
  }
  std::ifstream file(path);
  if (!file.good()) {
    TT_LOG_WARN(
        "[device_map_io] cannot open {}; using empty device map (placeholder "
        "chip ids)",
        path);
    return {};
  }
  DeviceMap deviceMap = loadDeviceMapStream(file);
  TT_LOG_INFO("[device_map_io] loaded {} entries from {}", deviceMap.size(),
              path);
  return deviceMap;
}

}  // namespace tt::transport
