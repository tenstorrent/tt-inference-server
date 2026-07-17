// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/engine_table_resolve.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "transport/device_map_io.hpp"
#include "transport/engine_table_handoff.hpp"
#include "transport/kv_table_provisioning.hpp"
#include "utils/logger.hpp"

namespace tt::transport {
namespace {

constexpr auto kHandoffPoll = std::chrono::milliseconds(1000);
constexpr auto kHandoffHeartbeat = std::chrono::milliseconds(30000);

}  // namespace

std::optional<ResolvedEngineTables> resolveEngineTablesFromFiles(
    const std::string& tablePath, const std::string& deviceMapPath) {
  auto loaded = loadKvTableFile(tablePath);
  if (!loaded || !loaded->table) {
    TT_LOG_ERROR("[engine_table_resolve] failed to load table from {}",
                 tablePath);
    return std::nullopt;
  }
  return ResolvedEngineTables{std::move(loaded->table), std::move(loaded->blob),
                              loadDeviceMapFile(deviceMapPath)};
}

std::optional<DeviceMap> awaitEngineHandoffOnPeer(
    sockets::ISocketTransport& peer, const std::atomic<bool>& stop) {
  auto lastWarn = std::chrono::steady_clock::now();
  while (!stop.load(std::memory_order_relaxed)) {
    const sockets::ReceiveResult result = peer.tryReceiveMessage();
    switch (result.status) {
      case sockets::ReceiveStatus::DATA: {
        auto payload = parseEngineHandoff(result.data);
        if (!payload) {
          TT_LOG_ERROR(
              "[engine_table_resolve] peer delivered malformed DeviceMap "
              "handoff");
          return std::nullopt;
        }
        TT_LOG_INFO(
            "[engine_table_resolve] DeviceMap handoff received: "
            "device_map_entries={}",
            payload->device_map.size());
        return std::move(payload->device_map);
      }
      case sockets::ReceiveStatus::CLOSED:
        TT_LOG_ERROR(
            "[engine_table_resolve] peer closed before a complete DeviceMap "
            "handoff arrived");
        return std::nullopt;
      case sockets::ReceiveStatus::NO_DATA:
        break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - lastWarn >= kHandoffHeartbeat) {
      TT_LOG_WARN(
          "[engine_table_resolve] still waiting for DeviceMap handoff bytes on "
          "accepted peer (readyz stays not-ready)");
      lastWarn = now;
    }
    std::this_thread::sleep_for(kHandoffPoll);
  }
  TT_LOG_WARN(
      "[engine_table_resolve] cancelled while waiting for DeviceMap handoff on "
      "peer");
  return std::nullopt;
}

std::optional<DeviceMap> awaitEngineHandoffOnListen(
    uint16_t port, const ListenTransportFactory& listenFactory,
    const std::atomic<bool>& stop) {
  if (!listenFactory) {
    TT_LOG_ERROR("[engine_table_resolve] missing listen transport factory");
    return std::nullopt;
  }
  auto listenTransport = listenFactory(port);
  if (!listenTransport) {
    TT_LOG_ERROR(
        "[engine_table_resolve] failed to create listen transport on port {}",
        port);
    return std::nullopt;
  }

  std::mutex peerMutex;
  std::condition_variable peerCv;
  std::shared_ptr<sockets::ISocketTransport> acceptedPeer;

  const bool multiAccept = listenTransport->enableMultiAccept(
      [&](std::shared_ptr<sockets::ISocketTransport> peer) {
        std::lock_guard<std::mutex> lock(peerMutex);
        if (acceptedPeer) {
          return;  // one-shot: ignore further peers
        }
        acceptedPeer = std::move(peer);
        peerCv.notify_all();
      });
  if (!multiAccept) {
    TT_LOG_ERROR(
        "[engine_table_resolve] listen transport does not support multi-accept "
        "(need TcpSocketTransport one-shot peer capture)");
    return std::nullopt;
  }

  listenTransport->start();
  TT_LOG_INFO(
      "[engine_table_resolve] listening for DeviceMap handoff on port {} "
      "(readyz stays not-ready until received)",
      port);

  auto lastWarn = std::chrono::steady_clock::now();
  std::shared_ptr<sockets::ISocketTransport> peer;
  while (!stop.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lock(peerMutex);
      peerCv.wait_for(lock, kHandoffPoll, [&] {
        return static_cast<bool>(acceptedPeer) ||
               stop.load(std::memory_order_relaxed);
      });
      if (acceptedPeer) {
        peer = acceptedPeer;
        break;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - lastWarn >= kHandoffHeartbeat) {
      TT_LOG_WARN(
          "[engine_table_resolve] still waiting for DeviceMap sender on port "
          "{} (readyz stays not-ready)",
          port);
      lastWarn = now;
    }
  }

  if (!peer) {
    listenTransport->stop();
    TT_LOG_WARN(
        "[engine_table_resolve] cancelled before a DeviceMap sender connected "
        "on port {}",
        port);
    return std::nullopt;
  }

  auto deviceMap = awaitEngineHandoffOnPeer(*peer, stop);
  listenTransport->stop();
  return deviceMap;
}

std::optional<ResolvedEngineTables> resolveEngineTables(
    uint16_t engineHandoffPort, const ListenTransportFactory& listenFactory,
    const std::string& tablePath, const std::string& deviceMapPath,
    const std::atomic<bool>& stop) {
  auto loaded = loadKvTableFile(tablePath);
  if (!loaded || !loaded->table) {
    TT_LOG_ERROR("[engine_table_resolve] failed to load table from {}",
                 tablePath);
    return std::nullopt;
  }

  DeviceMap deviceMap;
  if (engineHandoffPort != 0) {
    auto handedOff =
        awaitEngineHandoffOnListen(engineHandoffPort, listenFactory, stop);
    if (!handedOff) return std::nullopt;
    deviceMap = std::move(*handedOff);
  } else {
    deviceMap = loadDeviceMapFile(deviceMapPath);
  }

  return ResolvedEngineTables{std::move(loaded->table), std::move(loaded->blob),
                              std::move(deviceMap)};
}

}  // namespace tt::transport
