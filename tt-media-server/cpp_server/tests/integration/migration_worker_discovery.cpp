// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Migration-worker discovery harness for issue #4209 — validates that the
// Mooncake Metadata Service can serve as the discovery mechanism between two
// transfer engines running on two separate hosts.
//
// Why this is a distinct PoC from transport_migration_e2e (#3890):
//
//   The #3890 harness uses metadata_uri = "P2PHANDSHAKE": each engine binds a
//   *random OS-assigned port* and there is no shared registry, so the sender
//   cannot know the receiver's address ahead of time. That harness smuggles the
//   receiver's actual host:randomPort name through a rendezvous FILE on a shared
//   path — which does not work across two independent hosts.
//
//   #4209 removes that hack. With a real metadata service (mooncake_master's
//   HTTP metadata server, etcd, or redis), the receiver advertises under a
//   PREDEFINED LOGICAL NAME; Mooncake's "new RPC mapping" path registers
//   <name> -> {auto-detected IP, OS-assigned dynamic port} in that service. The
//   sender opens the predefined name and the service resolves the dynamic
//   address for it. No rendezvous file, no predefined ports — pure name-based
//   discovery across hosts.
//
//   sender host RAM --readInto--> registered host staging
//       --submitTransfer(TCP)--> receiver registered host staging
//       --writeFrom--> receiver host RAM --readInto--> verify
//
// Storage is HOST RAM only (#4209 scope: "Use only host RAM, no device
// memory"). Only built when Mooncake is in the build
// (TT_TRANSPORT_WITH_MOONCAKE); otherwise the engine is a no-op and init()
// reports failure.

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

// A "done" sentinel byte is appended after the tensor in the receiver's staging
// buffer. The sender writes the tensor, waits for that transfer to complete,
// then writes the flag; the receiver polls the flag. Because submitAndWait
// blocks until the tensor transfer completes (and TCP delivery is ordered),
// seeing flag == K_DONE_FLAG guarantees the tensor bytes have already landed.
constexpr std::uint8_t K_DONE_FLAG = 0xAB;

// Poll interval while the sender waits for the receiver to register its segment
// in the metadata service.
constexpr int K_DISCOVERY_POLL_MS = 100;

struct Options {
  std::string role;                  // "sender" | "receiver"
  std::string metadata;              // discovery service, e.g.
                                     // "http://HOST:8080/metadata"
  std::string name;                  // this engine's advertised logical name
  std::string peer;                  // sender: receiver's predefined name
  std::size_t bytes = 65536;         // tensor size
  int timeout_sec = 30;
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

// Deterministic, position-dependent pattern both sides agree on, so the
// receiver verifies without the payload being communicated out-of-band.
std::vector<std::uint8_t> makePattern(std::size_t n) {
  std::vector<std::uint8_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<std::uint8_t>((i * 131u + 7u) & 0xFFu);
  }
  return v;
}

std::shared_ptr<MooncakeTransferEngine> makeEngine(const Options& o) {
  auto engine =
      std::make_shared<MooncakeTransferEngine>(
          std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.metadata_uri = o.metadata;
  cfg.local_server_name = o.name;
  cfg.protocol = TransportProtocol::Tcp;
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

  // Host RAM standing in as the source "device" memory.
  std::vector<std::uint8_t> srcRegion(o.bytes, 0);
  const auto srcAddr = reinterpret_cast<NocAddr>(srcRegion.data());
  auto storage = engine->storage();

  const std::vector<std::uint8_t> pattern = makePattern(o.bytes);
  if (!storage->writeFrom(srcAddr, pattern.data(), pattern.size())) {
    std::cerr << "[sender] writeFrom (seed source) failed\n";
    return 1;
  }

  // Stage source -> registered host buffer (tensor + 1 done-flag byte).
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

  // Discovery: open the receiver by its PREDEFINED logical name. Retry until
  // the receiver has registered in the metadata service. The dynamic port is
  // resolved by the service — we never learn or pass it.
  std::cout << "[sender] discovering peer '" << o.peer
            << "' via metadata service...\n";
  SegmentHandle peer = kInvalidSegment;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(o.timeout_sec);
  while (peer == kInvalidSegment &&
         std::chrono::steady_clock::now() < deadline) {
    peer = engine->openSegment(o.peer);
    if (peer == kInvalidSegment) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
    }
  }
  if (peer == kInvalidSegment) {
    std::cerr << "[sender] openSegment(" << o.peer
              << ") failed (receiver not discoverable within timeout)\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }
  std::cout << "[sender] discovered peer '" << o.peer << "'\n";

  // Transfer the tensor into the receiver's segment at offset 0.
  TransferRequest tensorReq;
  tensorReq.op = TransferOp::Write;
  tensorReq.local_addr = staging.data();
  tensorReq.target = peer;
  tensorReq.target_offset = 0;
  tensorReq.length = o.bytes;
  if (engine->submitAndWait(tensorReq).state != TransferState::Completed) {
    std::cerr << "[sender] tensor transfer failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  // Then the done-flag byte at offset == bytes.
  staging[o.bytes] = K_DONE_FLAG;
  TransferRequest flagReq;
  flagReq.op = TransferOp::Write;
  flagReq.local_addr = staging.data() + o.bytes;
  flagReq.target = peer;
  flagReq.target_offset = o.bytes;
  flagReq.length = 1;
  if (engine->submitAndWait(flagReq).state != TransferState::Completed) {
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

  // Register the staging buffer the sender writes into (tensor + flag byte).
  // It is this segment's only buffer, so it resolves to target_offset 0.
  std::vector<std::uint8_t> staging(o.bytes + 1, 0);
  if (!engine->registerLocalMemory(staging.data(), staging.size())) {
    std::cerr << "[receiver] registerLocalMemory failed\n";
    return 1;
  }

  // After registration the segment is discoverable in the metadata service
  // under our predefined --name; the sender needs only that name, not our
  // dynamic port (logged here for visibility only).
  std::cout << "[receiver] '" << o.name << "' registered, reachable at "
            << engine->localServerName() << " (metadata=" << o.metadata
            << ")\n";
  std::cout << "[receiver] waiting for " << o.bytes
            << " bytes; sender opens us by name '" << o.name << "'\n";

  // Wait for the sender's one-sided writes to land (flag flips last).
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

  // Stage host staging -> receiver host RAM (the "device" stand-in), then read
  // it back and byte-compare via the migration worker.
  std::vector<std::uint8_t> dstRegion(o.bytes, 0);
  const auto dstAddr = reinterpret_cast<NocAddr>(dstRegion.data());
  if (!engine->storage()->writeFrom(dstAddr, staging.data(), o.bytes)) {
    std::cerr << "[receiver] writeFrom (host->dst) failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  MigrationWorkerConfig wcfg;
  wcfg.role = MigrationRole::Receiver;
  wcfg.device_addr = dstAddr;
  wcfg.tensor_bytes = o.bytes;
  MooncakeMigrationWorker worker(wcfg, engine);

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
