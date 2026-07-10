// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "transport/kv_cache_layout.hpp"  // LocalDeviceId
#include "transport/kv_table_view.hpp"    // FabricNode, encodeDevice

namespace tt::transport {

/**
 * @brief `FabricNode → physical UMD ASIC unique id`.
 *
 * The KV table addresses devices by FabricNode{mesh,chip} (collapsed to a
 * LocalDeviceId via encodeDevice). To actually read/write a chip the worker
 * needs that node's **UMD chip id** (the 64-bit ASIC unique id), which can only
 * be resolved against a live device by a CreateDevice-capable process — the
 * engine. The engine resolves it and hands this map to the worker alongside the
 * table; the device-IO layer uses it to open the right chip (replacing the
 * `chip == chip_id` placeholder).
 *
 * Keyed by LocalDeviceId (what the address layer and device IO already use).
 * encodeDevice is invertible for our id ranges — `(mesh << 16) | chip` — so the
 * (mesh, chip) coordinates survive a round-trip through the wire form.
 */
class DeviceMap {
 public:
  void set(const FabricNode& node, uint64_t umdChipId) {
    by_device_[encodeDevice(node)] = umdChipId;
  }
  void setByDevice(LocalDeviceId device, uint64_t umdChipId) {
    by_device_[device] = umdChipId;
  }

  std::optional<uint64_t> umdChip(LocalDeviceId device) const {
    const auto it = by_device_.find(device);
    if (it == by_device_.end()) return std::nullopt;
    return it->second;
  }
  std::optional<uint64_t> umdChip(const FabricNode& node) const {
    return umdChip(encodeDevice(node));
  }

  std::size_t size() const { return by_device_.size(); }
  bool empty() const { return by_device_.empty(); }
  const std::unordered_map<LocalDeviceId, uint64_t>& entries() const {
    return by_device_;
  }

 private:
  std::unordered_map<LocalDeviceId, uint64_t> by_device_;
};

}  // namespace tt::transport
