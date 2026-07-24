// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <istream>
#include <optional>
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

/**
 * @brief Load a device map from a file path.
 *
 * - Empty path => empty DeviceMap (discovery-only / map not configured).
 * - Unreadable path => nullopt (hard error; never silently empty).
 * - Readable path with zero parseable entries => empty DeviceMap; callers that
 *   require a transfer map must reject emptiness themselves.
 */
std::optional<DeviceMap> loadDeviceMapFile(const std::string& path);

}  // namespace tt::transport
