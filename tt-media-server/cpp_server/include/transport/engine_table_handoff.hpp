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
#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief Our own engine → worker table + device-map handoff (NOT the
 *        disaggregation MigrationLayerClient shmem path).
 *
 * The migration worker is a separate process from the inference engine, but the
 * engine is the only authority for both the KV-cache address table (NoC
 * addresses come from its on-device tensor allocation) and the device map
 * (FabricNode → ASIC unique id, resolvable only against live devices). So at
 * bring-up the worker pulls both from the engine over a co-located link.
 *
 * This defines a self-contained contract — a single framed message carrying the
 * serialized table plus the device map — moved over any `ISocketTransport`
 * (a local socket between the sidecar/daemonset worker and the engine; a
 * loopback fake in tests). The engine implements the producer
 * (`sendEngineHandoff`); the worker consumes via `IEngineTableSource`. Nothing
 * here depends on the disaggregation shmem client, MPI, or a `.pb` on the
 * worker's disk.
 *
 * Wire format (little-endian, length-prefixed; mirrors kv_control_message):
 *   [u32 table_blob_len][table_blob bytes]
 *   [u32 devmap_count]( [u32 mesh][u32 chip][u64 umd_chip_id] )*
 */

/// What the worker receives: the parsed table and the device map.
struct EngineTables {
  std::shared_ptr<const IKvTable> table;
  DeviceMap device_map;
};

/// Just the wire fields, before the (guarded) table deserialization.
struct EngineHandoffPayload {
  std::vector<uint8_t> table_blob;  ///< serialized KvChunkAddressTable bytes.
  DeviceMap device_map;
};

std::vector<uint8_t> serializeEngineHandoff(
    const std::vector<uint8_t>& tableBlob, const DeviceMap& deviceMap);

/// Parse a handoff blob. nullopt on truncation / malformed bytes. Does NOT
/// deserialize the table (callers that want an IKvTable use
/// receiveEngineHandoff or deserializeKvTable), so this stays usable without
/// ENABLE_KV_TABLE.
std::optional<EngineHandoffPayload> parseEngineHandoff(
    std::span<const uint8_t> bytes);

/// Engine (producer) side: send one handoff message over the transport.
bool sendEngineHandoff(sockets::ISocketTransport& transport,
                       const std::vector<uint8_t>& tableBlob,
                       const DeviceMap& deviceMap);

/// Worker (consumer) side: receive one message, parse it, and deserialize the
/// table. nullopt on transport close, malformed bytes, or a table that fails to
/// parse (e.g. ENABLE_KV_TABLE is OFF).
std::optional<EngineTables> receiveEngineHandoff(
    sockets::ISocketTransport& transport);

/// Seam: where the worker gets its table + device map from the engine. A static
/// source can stand in for the engine in tests; the socket source is the real
/// co-located link. Swapping in a shmem source later changes only this.
class IEngineTableSource {
 public:
  virtual ~IEngineTableSource() = default;
  virtual std::optional<EngineTables> fetch() = 0;
};

/// Pulls the handoff from the engine over an `ISocketTransport`.
class SocketEngineTableSource : public IEngineTableSource {
 public:
  explicit SocketEngineTableSource(
      std::shared_ptr<sockets::ISocketTransport> transport);
  std::optional<EngineTables> fetch() override;

 private:
  std::shared_ptr<sockets::ISocketTransport> transport_;
};

}  // namespace tt::transport
