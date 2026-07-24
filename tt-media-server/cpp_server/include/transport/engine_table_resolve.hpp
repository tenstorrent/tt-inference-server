// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/device_map.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief Host-local table (from .pb file) + device map for worker bring-up.
 *
 * The table always comes from disk (engine /tmp export or deploy path). The
 * DeviceMap comes from a localhost socket handoff and/or a --device-map file.
 */
struct ResolvedEngineTables {
  std::shared_ptr<const IKvTable> table;
  std::vector<uint8_t> blob;
  DeviceMap deviceMap;
};

using ListenTransportFactory =
    std::function<std::shared_ptr<sockets::ISocketTransport>(uint16_t port)>;

/// Load .pb + optional device-map file (empty path => empty DeviceMap).
/// Non-empty map path that is missing/unreadable/empty-entries => nullopt.
std::optional<ResolvedEngineTables> resolveEngineTablesFromFiles(
    const std::string& tablePath, const std::string& deviceMapPath);

/**
 * @brief Wait on an already-accepted peer for one DeviceMap handoff.
 *
 * Polls tryReceiveMessage (NO_DATA vs CLOSED). Wait-forever with WARN
 * heartbeats; returns nullopt on stop, CLOSED without data, bad parse, or an
 * empty DeviceMap (socket path never accepts placeholder bring-up).
 */
std::optional<DeviceMap> awaitEngineHandoffOnPeer(
    sockets::ISocketTransport& peer, const std::atomic<bool>& stop);

/**
 * @brief Listen for one DeviceMap handoff via multi-accept (one-shot peer).
 *
 * @p listenFactory must return an ISocketTransport that has already called
 * initializeAsServer(port) but has NOT start()ed yet (so enableMultiAccept can
 * be installed first — same pattern as KvMigrationReceiverServer).
 */
std::optional<DeviceMap> awaitEngineHandoffOnListen(
    uint16_t port, const ListenTransportFactory& listenFactory,
    const std::atomic<bool>& stop);

/**
 * @brief Resolve table from @p tablePath; DeviceMap from socket and/or file.
 *
 * - Table is always loaded from @p tablePath (required).
 * - engineHandoffPort != 0: await DeviceMap on the listen port (preferred);
 *   empty handoff is rejected.
 * - Else: loadDeviceMapFile(deviceMapPath) (empty path => empty map; non-empty
 *   path must be readable and non-empty).
 * - If both handoff and deviceMapPath are set, handoff wins and the file is
 *   ignored (caller should WARN).
 */
std::optional<ResolvedEngineTables> resolveEngineTables(
    uint16_t engineHandoffPort, const ListenTransportFactory& listenFactory,
    const std::string& tablePath, const std::string& deviceMapPath,
    const std::atomic<bool>& stop);

}  // namespace tt::transport
