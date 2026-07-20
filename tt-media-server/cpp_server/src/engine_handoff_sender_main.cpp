// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// engine_handoff_sender: host-local bridge that pushes a FabricNode→UMD
// DeviceMap to a waiting mooncake_kv_migration_worker via sendEngineHandoff.
// The worker already has its KV table from a .pb file (often under /tmp); this
// sender only supplies the live (or synthesized) chip map.
//
// Links transport_lib only (no Metal). For live maps, pipe print_local_device_map:
//
//   print_local_device_map | engine_handoff_sender --host 127.0.0.1 --port N \
//       --device-map-stdin
//
// Or push a pre-built file (deploy / #4571 synth):
//
//   engine_handoff_sender --host 127.0.0.1 --port N --device-map tag.devmap

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map_io.hpp"
#include "transport/engine_table_handoff.hpp"
#include "utils/logger.hpp"

namespace {

struct SenderConfig {
  std::string host = "127.0.0.1";
  uint16_t port = 0;
  std::string deviceMapPath;
  bool deviceMapFromStdin = false;
};

void usage() {
  std::cerr
      << "usage: engine_handoff_sender --host HOST --port N\n"
         "  (--device-map PATH | --device-map-stdin)\n"
         "  pushes DeviceMap only; worker loads .pb from --table /tmp path\n"
         "  refuses empty/unreadable maps (exit non-zero)\n";
}

bool parseArgs(int argc, char** argv, SenderConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--host" && next(cfg.host)) continue;
    if (a == "--port" && next(v)) {
      try {
        const int port = std::stoi(v);
        if (port <= 0 || port > 65535) throw std::out_of_range("range");
        cfg.port = static_cast<uint16_t>(port);
      } catch (...) {
        std::cerr << "--port must be 1..65535, got: " << v << "\n";
        return false;
      }
      continue;
    }
    if (a == "--device-map" && next(cfg.deviceMapPath)) continue;
    if (a == "--device-map-stdin") {
      cfg.deviceMapFromStdin = true;
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (cfg.host.empty() || cfg.port == 0) {
    std::cerr << "--host and --port are required\n";
    return false;
  }
  // Exactly one source: stdin XOR file path.
  if (cfg.deviceMapFromStdin != cfg.deviceMapPath.empty()) {
    std::cerr << "use exactly one of --device-map / --device-map-stdin\n";
    return false;
  }
  return true;
}

std::optional<tt::transport::DeviceMap> loadDeviceMap(const SenderConfig& cfg) {
  if (cfg.deviceMapFromStdin) {
    auto deviceMap = tt::transport::loadDeviceMapStream(std::cin);
    if (deviceMap.empty()) {
      TT_LOG_ERROR(
          "[engine_handoff_sender] stdin DeviceMap is empty — refusing to "
          "push a placeholder map");
      return std::nullopt;
    }
    TT_LOG_INFO("[engine_handoff_sender] device-map: {} entries from stdin",
                deviceMap.size());
    return deviceMap;
  }
  auto deviceMap = tt::transport::loadDeviceMapFile(cfg.deviceMapPath);
  if (!deviceMap) {
    TT_LOG_ERROR(
        "[engine_handoff_sender] failed to load DeviceMap from '{}' — refusing "
        "to push",
        cfg.deviceMapPath);
    return std::nullopt;
  }
  if (deviceMap->empty()) {
    TT_LOG_ERROR(
        "[engine_handoff_sender] DeviceMap file '{}' has no entries — refusing "
        "to push an empty map",
        cfg.deviceMapPath);
    return std::nullopt;
  }
  return deviceMap;
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("engine-handoff-sender");

  SenderConfig cfg;
  if (!parseArgs(argc, argv, cfg)) {
    usage();
    return 2;
  }

  const auto deviceMap = loadDeviceMap(cfg);
  if (!deviceMap) {
    return 1;
  }

  auto transport = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!transport->initializeAsClient(cfg.host, cfg.port)) {
    TT_LOG_ERROR("[engine_handoff_sender] connect {}:{} failed", cfg.host,
                 cfg.port);
    return 1;
  }
  transport->start();

  if (!tt::transport::sendEngineHandoff(*transport, *deviceMap)) {
    TT_LOG_ERROR("[engine_handoff_sender] sendEngineHandoff failed");
    transport->stop();
    return 1;
  }

  TT_LOG_INFO(
      "[engine_handoff_sender] sent DeviceMap handoff to {}:{} "
      "device_map_entries={}",
      cfg.host, cfg.port, deviceMap->size());
  transport->stop();
  return 0;
}
