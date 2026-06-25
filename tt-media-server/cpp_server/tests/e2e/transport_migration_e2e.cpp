// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Two-process acceptance harness for the #3890 migration path — the
// "Two galaxies (acceptance)" item in
// mooncake/poc-transfer-engine/adr-mooncake-backend.md.
//
// It exercises the full bounce-buffer round trip:
//
//   sender device DRAM --readInto(UMD)--> host staging
//       --submitAndWait(TCP/RDMA)--> receiver host staging
//       --writeFrom(UMD)--> receiver device DRAM --readInto--> verify
//
// Run it as two processes (see tests/integration/run_transport_migration_e2e.sh
// for a single-host loopback launch). Both sides derive the same deterministic
// byte pattern, so the receiver verifies without the payload being communicated
// out-of-band.
//
// Storage modes:
//   --storage device : real TT device DRAM via UMD (needs a build with
//                       TT_METAL_HOME set and real hardware) — the true
//                       acceptance test.
//   --storage host   : a host buffer stands in for device DRAM, so the full
//                       transport + staging logic can be validated over real
//                       Mooncake TCP on any machine with a --mooncake build.
//
// Only built when Mooncake is in the build (TT_TRANSPORT_WITH_MOONCAKE); the
// MooncakeTransferEngine is otherwise a no-op and init() reports failure.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transport/device_dram_storage_backend.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"

namespace {

using namespace tt::transport;

// A "done" sentinel byte is appended after the tensor in the receiver's staging
// buffer. The sender writes the tensor, waits for that transfer to complete,
// then writes the flag; the receiver polls the flag. Because submitAndWait
// blocks until the tensor transfer completes (and TCP delivery is ordered),
// seeing flag == kDoneFlag guarantees the tensor bytes have already landed.
constexpr std::uint8_t K_DONE_FLAG = 0xAB;

struct Options {
  std::string role;                // "sender" | "receiver"
  std::string local_name;          // this process's host:port (bind hint)
  std::string peer_name;           // sender: receiver's *actual* segment name
  std::string rendezvous;          // file used to exchange the receiver's name
  std::string storage = "device";  // "device" | "host"
  std::string metadata = "P2PHANDSHAKE";
  std::uint64_t device_addr = 0;  // device mode: NocAddr of the tensor
  std::size_t bytes = 4096;       // tensor size
  int device_id = 0;              // device mode: UMD chip id
  int timeout_sec = 30;
};

void usage() {
  std::cerr
      << "usage: transport_migration_e2e --role sender|receiver --local "
         "HOST:PORT\n"
         "  Peer discovery (sender needs one of):\n"
         "    --peer HOST:PORT     receiver's ACTUAL advertised name (it "
         "prints it)\n"
         "    --rendezvous PATH    file the receiver writes its name to "
         "(shared path)\n"
         "  --storage device|host  [--device-addr 0xNNN] [--bytes N] "
         "[--device-id N]\n"
         "  [--metadata URI] [--timeout-sec S]\n"
         "\n"
         "Note: under P2PHANDSHAKE the port in --local is ignored; the "
         "transport\n"
         "binds a random port, so peers must exchange the real name "
         "(--rendezvous,\n"
         "or copy the receiver's printed name into the sender's --peer).\n";
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
    if (a == "--local" && next(o.local_name)) continue;
    if (a == "--peer" && next(o.peer_name)) continue;
    if (a == "--rendezvous" && next(o.rendezvous)) continue;
    if (a == "--storage" && next(o.storage)) continue;
    if (a == "--metadata" && next(o.metadata)) continue;
    std::string v;
    if (a == "--device-addr" && next(v)) {
      o.device_addr = std::strtoull(v.c_str(), nullptr, 0);
      continue;
    }
    if (a == "--bytes" && next(v)) {
      o.bytes = std::strtoull(v.c_str(), nullptr, 0);
      continue;
    }
    if (a == "--device-id" && next(v)) {
      o.device_id = std::atoi(v.c_str());
      continue;
    }
    if (a == "--timeout-sec" && next(v)) {
      o.timeout_sec = std::atoi(v.c_str());
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if ((o.role != "sender" && o.role != "receiver") || o.local_name.empty()) {
    return false;
  }
  if (o.role == "sender" && o.peer_name.empty() && o.rendezvous.empty()) {
    std::cerr << "--role sender needs --peer or --rendezvous\n";
    return false;
  }
  return true;
}

// Deterministic, position-dependent pattern both sides agree on.
std::vector<std::uint8_t> makePattern(std::size_t n) {
  std::vector<std::uint8_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<std::uint8_t>((i * 131u + 7u) & 0xFFu);
  }
  return v;
}

// Publish the receiver's actual segment name atomically (write temp + rename).
bool writeRendezvous(const std::string& path, const std::string& name) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream os(tmp, std::ios::trunc);
    if (!os) return false;
    os << name << "\n";
  }
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

// Wait for the receiver to publish its name, returning it (or empty on
// timeout).
std::string readRendezvous(const std::string& path, int timeoutSec) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  for (;;) {
    std::ifstream is(path);
    std::string line;
    if (is && std::getline(is, line) && !line.empty()) return line;
    if (std::chrono::steady_clock::now() > deadline) return {};
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

// Holds the storage backend and (for host mode) the host region standing in for
// device DRAM. Returns the address the worker should use as device_addr.
struct Storage {
  std::shared_ptr<IStorageBackend> backend;
  std::vector<std::uint8_t> host_region;  // host mode only
  std::uint64_t device_addr = 0;
};

Storage makeStorage(const Options& o) {
  Storage s;
  if (o.storage == "host") {
    s.backend = std::make_shared<HostDramStorageBackend>();
    s.host_region.assign(o.bytes, 0);
    s.device_addr = reinterpret_cast<std::uint64_t>(s.host_region.data());
  } else {
    s.backend = std::make_shared<DeviceDramStorageBackend>(
        std::make_shared<UmdDeviceAccess>(o.device_id));
    s.device_addr = o.device_addr;
  }
  return s;
}

int runSender(const Options& o) {
  Storage storage = makeStorage(o);
  auto engine = std::make_shared<MooncakeTransferEngine>(storage.backend);

  EngineConfig cfg;
  cfg.metadata_uri = o.metadata;
  cfg.local_server_name = o.local_name;
  cfg.protocol = TransportProtocol::Tcp;
  if (!engine->init(cfg)) {
    std::cerr << "[sender] engine init failed\n";
    return 1;
  }

  // Resolve the receiver's actual segment name: explicit --peer wins, else wait
  // for it on the rendezvous file (the receiver publishes it after init).
  std::string peerName = o.peer_name;
  if (peerName.empty()) {
    std::cout << "[sender] waiting for receiver name on " << o.rendezvous
              << "...\n";
    peerName = readRendezvous(o.rendezvous, o.timeout_sec);
    if (peerName.empty()) {
      std::cerr << "[sender] no receiver name from rendezvous " << o.rendezvous
                << "\n";
      return 1;
    }
  }
  std::cout << "[sender] peer segment: " << peerName << "\n";

  // Step 1: write the known tensor into "sender device DRAM".
  MigrationWorkerConfig wcfg;
  wcfg.role = MigrationRole::Sender;
  wcfg.peer_segment_name = peerName;
  wcfg.device_addr = storage.device_addr;
  wcfg.tensor_bytes = o.bytes;
  // Discovery is unused here — this PoC drives the data-plane spike directly.
  MooncakeMigrationWorker worker(wcfg, engine, /*discovery=*/nullptr);

  const std::vector<std::uint8_t> pattern = makePattern(o.bytes);
  if (!worker.writeTensorOnSender(pattern)) {
    std::cerr << "[sender] writeTensorOnSender failed\n";
    return 1;
  }

  // Step 2: stage device -> registered host buffer (tensor + 1 flag byte), then
  // push tensor and flag to the receiver's segment. We drive the transport
  // directly here (rather than worker.transferToReceiver) so we own the staging
  // buffer and can append the completion flag.
  std::vector<std::uint8_t> staging(o.bytes + 1, 0);
  if (!engine->registerLocalMemory(staging.data(), staging.size())) {
    std::cerr << "[sender] registerLocalMemory failed\n";
    return 1;
  }
  if (!storage.backend->readInto(storage.device_addr, o.bytes,
                                 staging.data())) {
    std::cerr << "[sender] readInto (device->host) failed\n";
    return 1;
  }

  // The receiver must be up and have registered its segment first; retry the
  // P2P handshake a few times.
  SegmentHandle peer = kInvalidSegment;
  for (int attempt = 0; attempt < o.timeout_sec * 10 && peer == kInvalidSegment;
       ++attempt) {
    peer = engine->openSegment(peerName);
    if (peer == kInvalidSegment) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  if (peer == kInvalidSegment) {
    std::cerr << "[sender] openSegment(" << peerName << ") failed\n";
    return 1;
  }

  // Transfer the tensor into the receiver's segment at offset 0.
  TransferRequest tensorReq;
  tensorReq.op = TransferOp::Write;
  tensorReq.local_addr = staging.data();
  tensorReq.target = peer;
  tensorReq.target_offset = 0;
  tensorReq.length = o.bytes;
  if (engine->submitAndWait(tensorReq).state != TransferState::Completed) {
    std::cerr << "[sender] tensor transfer failed\n";
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
    return 1;
  }

  engine->unregisterLocalMemory(staging.data());
  std::cout << "[sender] done: wrote and transferred " << o.bytes
            << " bytes to " << peerName << "\n";
  return 0;
}

int runReceiver(const Options& o) {
  Storage storage = makeStorage(o);
  auto engine = std::make_shared<MooncakeTransferEngine>(storage.backend);

  EngineConfig cfg;
  cfg.metadata_uri = o.metadata;
  cfg.local_server_name = o.local_name;
  cfg.protocol = TransportProtocol::Tcp;
  if (!engine->init(cfg)) {
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

  // Publish our ACTUAL segment name so the sender can find us. Under
  // P2PHANDSHAKE this is host:<random_port>, not what was passed in --local.
  // Done after registerLocalMemory so the segment is ready before the sender
  // can read the name and open it.
  const std::string myName = engine->localServerName();
  std::cout << "[receiver] actual segment name: " << myName << "\n";
  if (!o.rendezvous.empty()) {
    if (!writeRendezvous(o.rendezvous, myName)) {
      std::cerr << "[receiver] failed to write rendezvous " << o.rendezvous
                << "\n";
      engine->unregisterLocalMemory(staging.data());
      return 1;
    }
    std::cout << "[receiver] published name to " << o.rendezvous << "\n";
  } else {
    std::cout << "[receiver] pass this to the sender: --peer " << myName
              << "\n";
  }
  std::cout << "[receiver] waiting for " << o.bytes << " bytes...\n";

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

  // Stage host staging -> receiver device DRAM.
  if (!storage.backend->writeFrom(storage.device_addr, staging.data(),
                                  o.bytes)) {
    std::cerr << "[receiver] writeFrom (host->device) failed\n";
    engine->unregisterLocalMemory(staging.data());
    return 1;
  }

  // Verify: read device DRAM back and byte-compare against the agreed pattern.
  MigrationWorkerConfig wcfg;
  wcfg.role = MigrationRole::Receiver;
  wcfg.device_addr = storage.device_addr;
  wcfg.tensor_bytes = o.bytes;
  // Discovery is unused here — this PoC drives the data-plane spike directly.
  MooncakeMigrationWorker worker(wcfg, engine, /*discovery=*/nullptr);

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
