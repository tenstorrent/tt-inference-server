// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// bringup_mooncake_worker: production entry point for a migration worker
// (#4294). Brings a worker up through a strict phased startup — config →
// allocate host-DRAM pool → init engine → register memory (publishes our
// segment) → discover peers (readiness gate) → hold until SIGTERM — then tears
// down in reverse order. The discovery itself lives in MooncakeMigrationWorker
// / PeerDiscovery; this file is the thin orchestration around it.

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"
#include "utils/logger.hpp"

namespace {

using namespace tt::transport;

constexpr std::size_t K_DEFAULT_HOST_DRAM_BYTES = 4ULL << 30;  // 4 GiB
constexpr std::size_t K_RAM_HEADROOM_BYTES = 1ULL << 30;       // 1 GiB
constexpr int K_DEFAULT_DISCOVERY_TIMEOUT_SEC = 30;

struct WorkerConfig {
  std::string metadata_uri;        ///< Discovery service (REQUIRED).
  std::string name;                ///< This worker's logical segment name.
  std::vector<std::string> peers;  ///< Peers to discover on bring-up.
  std::size_t host_dram_bytes = K_DEFAULT_HOST_DRAM_BYTES;
  int discovery_timeout_sec = K_DEFAULT_DISCOVERY_TIMEOUT_SEC;
  TransportProtocol protocol = TransportProtocol::Tcp;
};

std::atomic<bool> g_stopRequested{false};

void onSignal(int /*signum*/) { g_stopRequested.store(true); }

void usage() {
  std::cerr
      << "usage: bringup_mooncake_worker\n"
         "  --metadata URI         discovery service (REQUIRED), e.g.\n"
         "                         http://META_HOST:8080/metadata\n"
         "  --name NAME            this worker's logical segment name "
         "(REQUIRED)\n"
         "  --peer NAME            peer to discover; repeatable (REQUIRED)\n"
         "  [--host-dram-bytes N]  pool size, page-aligned (default 4 GiB)\n"
         "  [--protocol tcp|rdma]  transport (default tcp)\n"
         "  [--discovery-timeout-sec S] (default 30)\n"
         "\n"
         "Multi-NIC hosts: set MC_TCP_BIND_ADDRESS to the IP peers connect "
         "to.\n";
}

// Page-alignment and RAM-headroom checks keep a misconfigured pool from
// silently over-committing the host (#4294 review note).
bool validateHostDramBytes(std::size_t bytes, std::string& err) {
  if (bytes == 0) {
    err = "must be > 0";
    return false;
  }
  const long pageSize = ::sysconf(_SC_PAGESIZE);
  if (pageSize > 0 && bytes % static_cast<std::size_t>(pageSize) != 0) {
    err = "must be page-aligned to " + std::to_string(pageSize);
    return false;
  }
  const long physPages = ::sysconf(_SC_PHYS_PAGES);
  if (pageSize > 0 && physPages > 0) {
    const auto totalRam = static_cast<std::size_t>(physPages) *
                          static_cast<std::size_t>(pageSize);
    if (bytes + K_RAM_HEADROOM_BYTES > totalRam) {
      err = "exceeds physical RAM minus 1 GiB headroom";
      return false;
    }
  }
  return true;
}

bool parseProtocol(const std::string& value, TransportProtocol& out) {
  if (value == "tcp") {
    out = TransportProtocol::Tcp;
    return true;
  }
  if (value == "rdma") {
    out = TransportProtocol::Rdma;
    return true;
  }
  return false;
}

// Strict unsigned parse: the WHOLE token must be a plain non-negative integer.
// strtoull alone would silently accept "4Gib" as 4 (it stops at the first
// non-digit); we reject any leftover so a typo fails loud instead of allocating
// 4 bytes (#4294 bug).
bool parseSizeBytes(const std::string& value, std::size_t& out,
                    std::string& err) {
  if (value.empty() || value.front() == '-') {
    err = "must be a non-negative integer";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
  if (errno == ERANGE) {
    err = "out of range";
    return false;
  }
  if (end == value.c_str() || *end != '\0') {
    err = "must be a plain integer (no suffixes like 'GiB')";
    return false;
  }
  out = static_cast<std::size_t>(parsed);
  return true;
}

// Strict positive-int parse: the WHOLE token must parse and be > 0. atoi would
// turn "abc" (and an explicit "0") into 0, which means "give up discovery
// immediately" — so an invalid value silently broke bring-up (#4294 bug).
bool parsePositiveInt(const std::string& value, int& out, std::string& err) {
  if (value.empty()) {
    err = "must be a positive integer";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno == ERANGE || parsed > std::numeric_limits<int>::max()) {
    err = "out of range";
    return false;
  }
  if (end == value.c_str() || *end != '\0') {
    err = "must be a plain integer";
    return false;
  }
  if (parsed <= 0) {
    err = "must be > 0";
    return false;
  }
  out = static_cast<int>(parsed);
  return true;
}

// Phase 1: parse and validate everything before any resource is touched.
bool parseConfig(int argc, char** argv, WorkerConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--metadata" && next(cfg.metadata_uri)) continue;
    if (a == "--name" && next(cfg.name)) continue;
    if (a == "--peer" && next(v)) {
      cfg.peers.push_back(v);
      continue;
    }
    if (a == "--host-dram-bytes" && next(v)) {
      std::string perr;
      if (!parseSizeBytes(v, cfg.host_dram_bytes, perr)) {
        std::cerr << "--host-dram-bytes invalid ('" << v << "'): " << perr
                  << "\n";
        return false;
      }
      continue;
    }
    if (a == "--protocol" && next(v) && parseProtocol(v, cfg.protocol))
      continue;
    if (a == "--discovery-timeout-sec" && next(v)) {
      std::string perr;
      if (!parsePositiveInt(v, cfg.discovery_timeout_sec, perr)) {
        std::cerr << "--discovery-timeout-sec invalid ('" << v << "'): " << perr
                  << "\n";
        return false;
      }
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (cfg.metadata_uri.empty() || cfg.name.empty() || cfg.peers.empty()) {
    std::cerr << "--metadata, --name and at least one --peer are required\n";
    return false;
  }
  std::string err;
  if (!validateHostDramBytes(cfg.host_dram_bytes, err)) {
    std::cerr << "--host-dram-bytes invalid: " << err << "\n";
    return false;
  }
  return true;
}

// Translate parsed CLI flags into the worker's domain config. The worker owns
// the lifecycle (allocate/init/register/connect/run/teardown); main only wires
// dependencies and maps config — it does not sequence the phases.
MigrationWorkerConfig toWorkerConfig(const WorkerConfig& cli) {
  MigrationWorkerConfig cfg;
  cfg.metadata_uri = cli.metadata_uri;
  cfg.segment_name = cli.name;
  cfg.protocol = cli.protocol;
  cfg.host_dram_bytes = cli.host_dram_bytes;
  cfg.peer_segment_names = cli.peers;
  cfg.discovery_timeout_sec = cli.discovery_timeout_sec;
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("bringup-worker");

  WorkerConfig cli;
  if (!parseConfig(argc, argv, cli)) {
    usage();
    return 2;
  }

  std::signal(SIGTERM, onSignal);
  std::signal(SIGINT, onSignal);

  // Composition root: build the engine (storage + transport) and hand it to a
  // worker that owns its own bring-up and teardown.
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  MooncakeMigrationWorker worker(toWorkerConfig(cli), std::move(engine));

  if (!worker.bringUp()) {
    TT_LOG_ERROR("[bringup] '{}' bring-up failed", cli.name);
    return 1;
  }
  worker.run(g_stopRequested);
  return 0;
}
