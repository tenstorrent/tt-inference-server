// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Migration-worker discovery harness: the receiver advertises a
// predefined logical name in the Mooncake Metadata Service; the sender resolves
// that name to the receiver's OS-assigned dynamic port and transfers one
// tensor. Host RAM only; built only with Mooncake (--mooncake).

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

namespace {

using namespace tt::transport;

// Written last, after the tensor transfer completes. TCP ordering means the
// receiver seeing this flag guarantees the tensor bytes already landed.
constexpr std::uint8_t K_DONE_FLAG = 0xAB;

constexpr int K_DISCOVERY_POLL_MS = 100;

struct Options {
  std::string role;      // "sender" | "receiver"
  std::string metadata;  // discovery service URI
  std::string name;      // this engine's advertised logical name
  std::string peer;      // sender only: receiver's name
  std::size_t bytes = 65536;
  int timeout_sec = 30;
  // sender only: additionally resolve the peer's routable HOST via
  // resolveServerName (the rpc_meta lookup the real prefill worker uses to
  // discover a decode host), and fail if it can't. openSegment already proves
  // handle-level discovery; this proves the host-level discovery our worker
  // integration depends on.
  bool check_resolve = false;
};

void usage() {
  std::cerr
      << "usage: migration_worker_discovery --role sender|receiver\n"
         "  --metadata URI    discovery service (REQUIRED), e.g.\n"
         "                    http://META_HOST:8080/metadata | etcd://... | "
         "redis://...\n"
         "  --name NAME       this engine's predefined logical segment name\n"
         "                    (receiver: the name it advertises; sender: its "
         "own name)\n"
         "  --peer NAME       sender only: the receiver's predefined --name\n"
         "  [--bytes N]       tensor size (default 65536)\n"
         "  [--timeout-sec S] (default 30)\n"
         "  [--check-resolve] sender only: also resolve the peer's HOST via\n"
         "                    resolveServerName (rpc_meta lookup) and fail if "
         "empty\n"
         "\n"
         "Discovery: the receiver registers --name in the metadata service "
         "under an\n"
         "OS-assigned dynamic port; the sender opens that predefined --peer "
         "name and\n"
         "the service resolves the address. No rendezvous file / predefined "
         "port.\n"
         "\n"
         "Multi-NIC hosts: set MC_TCP_BIND_ADDRESS to the IP the peer should\n"
         "connect to if auto-detection picks the wrong interface.\n";
}

bool parseArgs(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    if (a == "--role" && next(o.role)) continue;
    if (a == "--metadata" && next(o.metadata)) continue;
    if (a == "--name" && next(o.name)) continue;
    if (a == "--peer" && next(o.peer)) continue;
    if (a == "--check-resolve") {
      o.check_resolve = true;
      continue;
    }
    std::string v;
    if (a == "--bytes" && next(v)) {
      o.bytes = std::strtoull(v.c_str(), nullptr, 0);
      continue;
    }
    if (a == "--timeout-sec" && next(v)) {
      o.timeout_sec = std::atoi(v.c_str());
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (o.role != "sender" && o.role != "receiver") return false;
  if (o.metadata.empty() || o.name.empty()) return false;
  if (o.metadata == "P2PHANDSHAKE") {
    std::cerr << "--metadata P2PHANDSHAKE has no discovery service; use a real "
                 "metadata service (see transport_migration_e2e for the P2P "
                 "rendezvous path)\n";
    return false;
  }
  if (o.role == "sender" && o.peer.empty()) {
    std::cerr << "--role sender needs --peer (the receiver's --name)\n";
    return false;
  }
  return true;
}

// Both sides compute this independently, so the receiver can verify content
// without the payload being communicated out-of-band.
std::vector<std::uint8_t> makePattern(std::size_t n) {
  std::vector<std::uint8_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<std::uint8_t>((i * 131u + 7u) & 0xFFu);
  }
  return v;
}

std::shared_ptr<MooncakeTransferEngine> makeEngine(const Options& o) {
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.metadata_uri = o.metadata;
  cfg.local_server_name = o.name;
  cfg.protocol = TransportProtocol::TCP;
  if (!engine->init(cfg)) return nullptr;
  return engine;
}

int runSender(const Options& o) {
  auto engine = makeEngine(o);
  if (!engine) {
    std::cerr << "[sender] engine init failed\n";
    return 1;
  }
  std::cout << "[sender] '" << o.name << "' up at " << engine->localServerName()
            << " (metadata=" << o.metadata << ")\n";

  std::vector<std::uint8_t> srcRegion(o.bytes, 0);
  const auto srcAddr = reinterpret_cast<NocAddr>(srcRegion.data());
  auto storage = engine->storage();

  const std::vector<std::uint8_t> pattern = makePattern(o.bytes);
  if (!storage->writeFrom(srcAddr, pattern.data(), pattern.size())) {
    std::cerr << "[sender] writeFrom (seed source) failed\n";
    return 1;
  }

  // Staging buffer is tensor + 1 done-flag byte.
  std::vector<std::uint8_t> staging(o.bytes + 1, 0);
  if (!engine->registerLocalMemory(staging.data(), staging.size())) {
    std::cerr << "[sender] registerLocalMemory failed\n";
    return 1;
  }
  if (!storage->readInto(srcAddr, o.bytes, staging.data())) {
    std::cerr << "[sender] readInto (source->host) failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  // Retry until the receiver has registered; the service resolves its dynamic
  // port, which we never learn or pass.
  std::cout << "[sender] discovering peer '" << o.peer
            << "' via metadata service...\n";
  SegmentHandle peer = K_INVALID_SEGMENT;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(o.timeout_sec);
  while (peer == K_INVALID_SEGMENT &&
         std::chrono::steady_clock::now() < deadline) {
    peer = engine->openSegment(o.peer);
    if (peer == K_INVALID_SEGMENT) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
    }
  }
  if (peer == K_INVALID_SEGMENT) {
    std::cerr << "[sender] openSegment(" << o.peer
              << ") failed (receiver not discoverable within timeout)\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }
  std::cout << "[sender] discovered peer '" << o.peer << "'\n";

  // Host-level discovery: this is the exact call the real prefill worker uses
  // to turn a decode's logical --name into its routable host (rpc_meta lookup)
  // so it can open a control channel. openSegment above proved handle
  // discovery; this proves the worker-integration path resolves a host too.
  if (o.check_resolve) {
    const std::string host = engine->resolveServerName(o.peer);
    if (host.empty()) {
      std::cerr << "[sender] resolveServerName(" << o.peer
                << ") returned empty (rpc_meta lookup failed)\n";
      engine->unregisterLocalMemory(staging.data());
      return 1;
    }
    std::cout << "[sender] resolved peer '" << o.peer << "' -> host " << host
              << "\n";
  }

  TransferRequest tensorReq;
  tensorReq.op = TransferOp::WRITE;
  tensorReq.local_addr = staging.data();
  tensorReq.target = peer;
  tensorReq.target_offset = 0;
  tensorReq.length = o.bytes;
  if (engine->submitAndWait(tensorReq).state != TransferState::COMPLETED) {
    std::cerr << "[sender] tensor transfer failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  staging[o.bytes] = K_DONE_FLAG;
  TransferRequest flagReq;
  flagReq.op = TransferOp::WRITE;
  flagReq.local_addr = staging.data() + o.bytes;
  flagReq.target = peer;
  flagReq.target_offset = o.bytes;
  flagReq.length = 1;
  if (engine->submitAndWait(flagReq).state != TransferState::COMPLETED) {
    std::cerr << "[sender] flag transfer failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  engine->unregisterLocalMemory(staging.data());
  std::cout << "[sender] done: transferred " << o.bytes << " bytes to '"
            << o.peer << "'\n";
  return 0;
}

int runReceiver(const Options& o) {
  auto engine = makeEngine(o);
  if (!engine) {
    std::cerr << "[receiver] engine init failed\n";
    return 1;
  }

  // Single registered buffer (tensor + flag byte), so it resolves to offset 0.
  std::vector<std::uint8_t> staging(o.bytes + 1, 0);
  if (!engine->registerLocalMemory(staging.data(), staging.size())) {
    std::cerr << "[receiver] registerLocalMemory failed\n";
    return 1;
  }

  std::cout << "[receiver] '" << o.name << "' registered, reachable at "
            << engine->localServerName() << " (metadata=" << o.metadata
            << ")\n";
  std::cout << "[receiver] waiting for " << o.bytes
            << " bytes; sender opens us by name '" << o.name << "'\n";

  auto* flag = reinterpret_cast<volatile std::uint8_t*>(&staging[o.bytes]);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(o.timeout_sec);
  while (*flag != K_DONE_FLAG) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "[receiver] timed out waiting for transfer\n";
      engine->unregisterLocalMemory(staging.data());
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::vector<std::uint8_t> dstRegion(o.bytes, 0);
  const auto dstAddr = reinterpret_cast<NocAddr>(dstRegion.data());
  if (!engine->storage()->writeFrom(dstAddr, staging.data(), o.bytes)) {
    std::cerr << "[receiver] writeFrom (host->dst) failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  MigrationWorkerConfig wcfg;
  wcfg.role = MigrationRole::RECEIVER;
  wcfg.device_addr = dstAddr;
  wcfg.tensor_bytes = o.bytes;
  // Discovery is unused here — this PoC drives the data-plane spike directly.
  MooncakeMigrationWorker worker{wcfg, engine, /*discovery=*/nullptr};

  const std::vector<std::uint8_t> expected = makePattern(o.bytes);
  const bool ok = worker.verifyTensorOnReceiver(expected);

  engine->unregisterLocalMemory(staging.data());
  std::cout << "[receiver] verification: " << (ok ? "PASS" : "FAIL") << "\n";
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parseArgs(argc, argv, o)) {
    usage();
    return 2;
  }
  return o.role == "sender" ? runSender(o) : runReceiver(o);
}
