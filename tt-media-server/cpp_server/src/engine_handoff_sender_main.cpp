// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// engine_handoff_sender: host-local bridge that pushes a KV table .pb +
// FabricNode→UMD device map to a waiting mooncake_kv_migration_worker via
// sendEngineHandoff. Stand-in for the model runner until it calls the same API.
//
// Links transport_lib only (no Metal). For live maps, pipe print_local_device_map:
//
//   print_local_device_map | engine_handoff_sender --host 127.0.0.1 --port N \
//       --table local.pb --device-map-stdin

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map_io.hpp"
#include "transport/engine_table_handoff.hpp"
#include "utils/logger.hpp"

namespace {

struct SenderConfig {
  std::string host = "127.0.0.1";
  uint16_t port = 0;
  std::string tablePath;
  std::string deviceMapPath;
  bool deviceMapFromStdin = false;
};

void usage() {
  std::cerr
      << "usage: engine_handoff_sender --host HOST --port N --table PATH.pb\n"
         "  [--device-map PATH | --device-map-stdin]\n"
         "  empty device map is allowed (synthetic / single-mesh placeholder)\n";
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
    if (a == "--table" && next(cfg.tablePath)) continue;
    if (a == "--device-map" && next(cfg.deviceMapPath)) continue;
    if (a == "--device-map-stdin") {
      cfg.deviceMapFromStdin = true;
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (cfg.host.empty() || cfg.port == 0 || cfg.tablePath.empty()) {
    std::cerr << "--host, --port and --table are required\n";
    return false;
  }
  if (cfg.deviceMapFromStdin && !cfg.deviceMapPath.empty()) {
    std::cerr << "use only one of --device-map / --device-map-stdin\n";
    return false;
  }
  return true;
}

std::vector<uint8_t> readFileBytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    return {};
  }
  const auto size = file.tellg();
  if (size < 0) {
    return {};
  }
  std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!file) {
    return {};
  }
  return bytes;
}

tt::transport::DeviceMap loadDeviceMap(const SenderConfig& cfg) {
  if (cfg.deviceMapFromStdin) {
    auto deviceMap = tt::transport::loadDeviceMapStream(std::cin);
    TT_LOG_INFO("[engine_handoff_sender] device-map: {} entries from stdin",
                deviceMap.size());
    return deviceMap;
  }
  return tt::transport::loadDeviceMapFile(cfg.deviceMapPath);
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("engine-handoff-sender");

  SenderConfig cfg;
  if (!parseArgs(argc, argv, cfg)) {
    usage();
    return 2;
  }

  const std::vector<uint8_t> tableBlob = readFileBytes(cfg.tablePath);
  if (tableBlob.empty()) {
    TT_LOG_ERROR("[engine_handoff_sender] failed to read --table {}",
                 cfg.tablePath);
    return 1;
  }

  const tt::transport::DeviceMap deviceMap = loadDeviceMap(cfg);

  auto transport = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!transport->initializeAsClient(cfg.host, cfg.port)) {
    TT_LOG_ERROR("[engine_handoff_sender] connect {}:{} failed", cfg.host,
                 cfg.port);
    return 1;
  }
  transport->start();

  if (!tt::transport::sendEngineHandoff(*transport, tableBlob, deviceMap)) {
    TT_LOG_ERROR("[engine_handoff_sender] sendEngineHandoff failed");
    transport->stop();
    return 1;
  }

  TT_LOG_INFO(
      "[engine_handoff_sender] sent handoff to {}:{} table_bytes={} "
      "device_map_entries={}",
      cfg.host, cfg.port, tableBlob.size(), deviceMap.size());
  // Close promptly — worker must still be able to read buffered bytes after FIN
  // (multi-accept peer path, no background POLLHUP teardown).
  transport->stop();
  return 0;
}
