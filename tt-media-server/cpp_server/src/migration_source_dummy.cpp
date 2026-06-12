// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Dummy migration source: creates a MooncakeTransferEngine, registers a buffer
// filled with a known pattern, and stays alive so that a MigrationWorker
// consumer can perform an RDMA pull against it.
//
// Prints its advertised segment name to stdout (format: "SEGMENT=host:port")
// so test scripts can parse it and feed it into Kafka messages.
//
// Usage:
//   ./build/migration_source_dummy [options]
//
// CLI flags:
//   --local-server-name HOST:PORT   Engine address hint (P2P rewrites port)
//   --buffer-size BYTES             Size of registered buffer (default: 65536)
//   --fill-byte VALUE               Fill byte pattern 0-255 (default: 0xAB)
//
// The process will block until it receives SIGINT or SIGTERM.

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/logger.hpp"

#ifdef TT_TRANSPORT_WITH_MOONCAKE
#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#endif

namespace {

volatile std::sig_atomic_t gRunning = 1;

void signalHandler(int /*signal*/) { gRunning = 0; }

}  // namespace

int main(int argc, char* argv[]) {
  std::string localServerName = "0.0.0.0:0";
  std::size_t bufferSize = 65536;
  uint8_t fillByte = 0xAB;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--local-server-name" && i + 1 < argc) {
      localServerName = argv[++i];
    } else if (arg == "--buffer-size" && i + 1 < argc) {
      bufferSize = std::stoull(argv[++i]);
    } else if (arg == "--fill-byte" && i + 1 < argc) {
      fillByte = static_cast<uint8_t>(std::stoi(argv[++i]));
    } else if (arg == "--help") {
      std::cout
          << "Migration Source Dummy — Mooncake segment source for testing\n"
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  --local-server-name HOST:PORT  Engine hint (default: "
             "0.0.0.0:0)\n"
          << "  --buffer-size BYTES            Buffer size (default: 65536)\n"
          << "  --fill-byte VALUE              Fill pattern 0-255 (default: "
             "171/0xAB)\n"
          << "  --help                         Show this help\n\n"
          << "Prints SEGMENT=<host:port> to stdout once ready.\n"
          << "Send SIGINT/SIGTERM to stop.\n";
      return 0;
    }
  }

  tt::utils::ZeroOverheadLogger::initialize("source");

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

#ifdef TT_TRANSPORT_WITH_MOONCAKE
  // Create engine.
  auto storage = std::make_shared<tt::transport::HostDramStorageBackend>();
  auto engine =
      std::make_shared<tt::transport::MooncakeTransferEngine>(storage);

  tt::transport::EngineConfig cfg;
  cfg.local_server_name = localServerName;

  if (!engine->init(cfg)) {
    std::cerr << "ERROR: MooncakeTransferEngine init failed\n";
    return 1;
  }

  std::string segmentName = engine->localServerName();
  TT_LOG_INFO("[Source] Engine initialised, segment = {}", segmentName);

  // Allocate buffer and fill with known pattern.
  std::vector<uint8_t> buffer(bufferSize, fillByte);
  if (!engine->registerLocalMemory(buffer.data(), buffer.size())) {
    std::cerr << "ERROR: registerLocalMemory failed\n";
    return 1;
  }

  TT_LOG_INFO("[Source] Registered {} bytes at addr {:#x} (fill=0x{:02X})",
              bufferSize, reinterpret_cast<uintptr_t>(buffer.data()), fillByte);

  // Print machine-parseable output for test scripts.
  // Flush immediately so pipe readers see it right away.
  std::cout << "SEGMENT=" << segmentName << std::endl;
  std::cout << "BUFFER_ADDR=" << reinterpret_cast<uintptr_t>(buffer.data())
            << std::endl;
  std::cout << "BUFFER_SIZE=" << bufferSize << std::endl;
  std::cout << "READY" << std::endl;

  TT_LOG_INFO("[Source] Ready — waiting for transfers (Ctrl+C to stop)...");

  // Block until signal.
  while (gRunning) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  TT_LOG_INFO("[Source] Shutting down...");
  engine->unregisterLocalMemory(buffer.data());
  TT_LOG_INFO("[Source] Done.");

#else
  (void)localServerName;
  (void)bufferSize;
  (void)fillByte;
  std::cerr
      << "ERROR: Built without TT_TRANSPORT_WITH_MOONCAKE — cannot act as "
         "source.\n"
      << "Rebuild with: ./build.sh --blaze --kafka --mooncake\n";
  return 1;
#endif

  return 0;
}
