// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/device_map.hpp"

namespace tt::transport {

/**
 * @brief Engine → worker DeviceMap handoff (NOT the disaggregation shmem path).
 *
 * The KV address table (.pb) is read from a file the engine already wrote
 * (typically under /tmp). Only the FabricNode→UMD map needs a live push: the
 * engine (or deploy bridge) resolves ASIC unique ids against open devices and
 * ships them over a co-located socket. The worker loads the table from disk and
 * waits for this map before opening chips.
 *
 * Wire format (little-endian, length-prefixed):
 *   [u32 count]( [u32 mesh][u32 chip][u64 umd_chip_id] )*
 */

struct EngineHandoffPayload {
  DeviceMap device_map;
};

std::vector<uint8_t> serializeEngineHandoff(const DeviceMap& deviceMap);

/// Parse a handoff blob. nullopt on truncation / malformed / trailing bytes.
std::optional<EngineHandoffPayload> parseEngineHandoff(
    std::span<const uint8_t> bytes);

/// Engine (producer) side: send one DeviceMap handoff over the transport.
bool sendEngineHandoff(sockets::ISocketTransport& transport,
                       const DeviceMap& deviceMap);

/// Worker (consumer) side: one tryReceiveMessage, then parse.
/// nullopt on NO_DATA, CLOSED, or malformed bytes. Callers that need
/// wait-until-ready should poll (see awaitEngineHandoffOnPeer).
std::optional<DeviceMap> receiveEngineHandoff(
    sockets::ISocketTransport& transport);

/// Interface: where the worker gets its DeviceMap from the engine / deploy
/// bridge.
class IEngineDeviceMapSource {
 public:
  virtual ~IEngineDeviceMapSource() = default;
  virtual std::optional<DeviceMap> fetch() = 0;
};

/// Pulls the handoff from the engine over an `ISocketTransport` (one attempt).
class SocketEngineDeviceMapSource : public IEngineDeviceMapSource {
 public:
  explicit SocketEngineDeviceMapSource(
      std::shared_ptr<sockets::ISocketTransport> transport);
  std::optional<DeviceMap> fetch() override;

 private:
  std::shared_ptr<sockets::ISocketTransport> transport_;
};

}  // namespace tt::transport
