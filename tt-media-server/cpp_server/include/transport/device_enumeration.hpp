// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <unordered_map>

namespace tt::transport {

/**
 * @brief Enumerate every visible device once, returning `unique_id -> 0-based
 *        index`.
 *
 * The index is what the coexistence UMD open path (`dis::UmdDevice::open()`,
 * the same path `DriscDeviceIo::addDevice()` drives) expects. A DeviceMap
 * carries a chip's 64-bit ASIC unique_id (the third device-map column); this
 * map translates it into that index so the caller opens the intended chip
 * rather than casting the unique_id to an index and opening the wrong one.
 *
 * Enumerating opens every visible chip once; the handles are dropped at return
 * (the shared per-process Cluster stays alive, so re-opening the chosen chips
 * is cheap). Returns an empty map when built without tt-metal
 * (USE_METAL_CPP_LIB undefined), so a non-empty DeviceMap resolves to a clean
 * miss.
 */
std::unordered_map<uint64_t, int> enumerateDevicesByUniqueId();

}  // namespace tt::transport
