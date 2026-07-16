// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <istream>
#include <string>

#include "transport/device_map.hpp"

namespace tt::transport {

/**
 * @brief Load a FabricNode→UMD map from `mesh chip umd` lines.
 *
 * Same format as print_local_device_map stdout and the worker's historical
 * --device-map files. Shared by the migration worker, engine_handoff_sender,
 * and unit tests.
 */
DeviceMap loadDeviceMapStream(std::istream& input);

/// Empty path => empty map. Unreadable path => empty map + warning log.
DeviceMap loadDeviceMapFile(const std::string& path);

}  // namespace tt::transport
