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
 * @brief Host-local table + device map used to bring up a migration worker.
 *
 * Mirrors LoadedKvTable (table + raw .pb blob for TABLE_EXCHANGE) plus the
 * FabricNode→UMD map needed by MultiDeviceUmd.
 */
struct ResolvedEngineTables {
  std::shared_ptr<const IKvTable> table;
  std::vector<uint8_t> blob;
  DeviceMap deviceMap;
};

using ListenTransportFactory =
    std::function<std::shared_ptr<sockets::ISocketTransport>(uint16_t port)>;

/// File-mode resolve: load .pb + optional device-map file.
std::optional<ResolvedEngineTables> resolveEngineTablesFromFiles(
    const std::string& tablePath, const std::string& deviceMapPath);

/**
 * @brief Wait on an already-accepted peer for one engine handoff.
 *
 * Polls tryReceiveMessage (NO_DATA vs CLOSED). Wait-forever with WARN
 * heartbeats; returns nullopt on stop, CLOSED without data, or bad parse.
 */
std::optional<ResolvedEngineTables> awaitEngineHandoffOnPeer(
    sockets::ISocketTransport& peer, const std::atomic<bool>& stop);

/**
 * @brief Listen for one engine handoff via multi-accept (one-shot peer).
 *
 * @p listenFactory must return an ISocketTransport that has already called
 * initializeAsServer(port) but has NOT start()ed yet (so enableMultiAccept can
 * be installed first — same pattern as KvMigrationReceiverServer).
 *
 * Binds INADDR_ANY like the control port; contract is host-local. Wait-forever
 * + WARN heartbeats; honors @p stop.
 */
std::optional<ResolvedEngineTables> awaitEngineHandoffOnListen(
    uint16_t port, const ListenTransportFactory& listenFactory,
    const std::atomic<bool>& stop);

/**
 * @brief Resolve table + device map for worker bring-up.
 *
 * - engineHandoffPort != 0: awaitEngineHandoffOnListen (file paths ignored).
 * - engineHandoffPort == 0: resolveEngineTablesFromFiles.
 */
std::optional<ResolvedEngineTables> resolveEngineTables(
    uint16_t engineHandoffPort, const ListenTransportFactory& listenFactory,
    const std::string& tablePath, const std::string& deviceMapPath,
    const std::atomic<bool>& stop);

}  // namespace tt::transport
