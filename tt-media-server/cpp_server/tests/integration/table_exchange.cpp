// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Prefill<->decode KV-table exchange PoC for #4268. Each node advertises a
// logical name in the Mooncake Metadata Service, then PUSHES its own
// KvChunkAddressTable into the peer's pre-registered receive buffer with
// one-sided Writes over the Transfer Engine; both sides end up holding both
// tables (the Mooncake stand-in for the MPI Connect() exchange). Control plane
// = name discovery; data plane = one-sided Write. We use Write (push), the same
// path the migration_worker PoC uses, because the TCP transport's read-serving
// path is unreliable here. Raw bytes (no compression); host RAM only; built
// only with Mooncake. Switching TransportProtocol to Rdma later does not change
// this flow.

#include <glog/logging.h>  // FLAGS_minloglevel: silence Mooncake's glog polling

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

namespace {

using namespace tt::transport;

// Written last into the peer's receive buffer, after its table body+header have
// landed. TCP ordering + sequential Writes => the peer seeing this flag means
// all preceding bytes already arrived.
constexpr std::uint8_t K_DONE_FLAG = 0xAB;
constexpr int K_DISCOVERY_POLL_MS = 1000;

// Receive buffer reserves room for the largest table a peer may push. The peer
// writes the real size into the header; the flag lives at a fixed offset past
// the reserved area so the reader knows where to poll without knowing the size.
constexpr std::size_t K_DEFAULT_MAX_BYTES = 128ull * 1024 * 1024;

// Self-describing header the publisher writes ahead of its table, so the reader
// learns the table size + checksum from a fixed offset. Both nodes are the same
// architecture, so the raw struct goes on the wire.
struct TableHeader {
  std::uint64_t table_bytes = 0;
  std::uint64_t checksum = 0;  // FNV-1a over the table bytes
};
constexpr std::size_t K_HEADER_BYTES = sizeof(TableHeader);

struct Options {
  std::string metadata;  // discovery service URI
  std::string name;      // this node's advertised logical name
  std::string peer;      // the peer node's name
  std::string table;     // this node's table file to publish
  std::string out;       // where to write the peer's pushed table
  int timeout_sec = 30;
  std::size_t max_bytes = K_DEFAULT_MAX_BYTES;  // receive-buffer table capacity
};

void usage() {
  std::cerr
      << "usage: table_exchange --metadata URI --name NAME --peer NAME "
         "--table PATH --out PATH\n"
         "  --metadata URI    discovery service (REQUIRED), e.g.\n"
         "                    http://META_HOST:8080/metadata\n"
         "  --name NAME       this node's logical segment name (e.g. "
         "kvtable-prefill-0)\n"
         "  --peer NAME       the peer node's --name (e.g. kvtable-decode-0)\n"
         "  --table PATH      this node's .pb table to publish\n"
         "  --out PATH        where to write the peer's pushed .pb table\n"
         "  [--timeout-sec S] (default 30)\n"
         "  [--max-bytes N]   receive-buffer table capacity (default 128 MiB)\n"
         "\n"
         "Symmetric: every node pushes its own table into the peer's receive "
         "buffer, so\n"
         "both sides hold both tables (the Mooncake stand-in for the MPI "
         "Connect() exchange).\n"
         "Switch TransportProtocol to Rdma later without touching this flow.\n";
}

bool parseArgs(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    if (a == "--metadata" && next(o.metadata)) continue;
    if (a == "--name" && next(o.name)) continue;
    if (a == "--peer" && next(o.peer)) continue;
    if (a == "--table" && next(o.table)) continue;
    if (a == "--out" && next(o.out)) continue;
    std::string v;
    if (a == "--timeout-sec" && next(v)) {
      o.timeout_sec = std::atoi(v.c_str());
      continue;
    }
    if (a == "--max-bytes" && next(v)) {
      o.max_bytes = std::strtoull(v.c_str(), nullptr, 0);
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (o.metadata.empty() || o.name.empty() || o.peer.empty() ||
      o.table.empty() || o.out.empty()) {
    return false;
  }
  if (o.metadata == "P2PHANDSHAKE") {
    std::cerr << "--metadata needs a real discovery service (not "
                 "P2PHANDSHAKE)\n";
    return false;
  }
  return true;
}

std::shared_ptr<MooncakeTransferEngine> makeEngine(const Options& o) {
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.metadata_uri = o.metadata;
  cfg.local_server_name = o.name;
  cfg.protocol = TransportProtocol::Tcp;
  if (!engine->init(cfg)) return nullptr;
  return engine;
}

// Cheap end-to-end integrity check that needs no copy of the original on the
// reading side (the publisher ships the expected value in the header).
std::uint64_t fnv1a(const std::uint8_t* data, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

bool readFile(const std::string& path, std::vector<std::uint8_t>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const auto size = static_cast<std::streamsize>(f.tellg());
  out.resize(static_cast<std::size_t>(size));
  f.seekg(0);
  return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), size));
}

bool writeFile(const std::string& path, const std::uint8_t* data,
               std::size_t n) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
  return static_cast<bool>(f);
}

// One-sided push into the peer's already-registered receive buffer. local must
// lie inside a region we registered up front; peer_offset is relative to the
// peer's receive buffer (its first registered region, buffers[0]).
bool writeToPeer(MooncakeTransferEngine& engine, SegmentHandle peer,
                 const std::uint8_t* local, std::size_t len,
                 std::uint64_t peer_offset) {
  TransferRequest req;
  req.op = TransferOp::Write;
  req.local_addr = const_cast<std::uint8_t*>(local);
  req.target = peer;
  req.target_offset = peer_offset;
  req.length = len;
  return engine.submitAndWait(req).state == TransferState::Completed;
}

SegmentHandle openWithRetry(MooncakeTransferEngine& engine,
                            const std::string& peer, int timeout_sec) {
  SegmentHandle handle = kInvalidSegment;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  while (handle == kInvalidSegment &&
         std::chrono::steady_clock::now() < deadline) {
    handle = engine.openSegment(peer);
    if (handle == kInvalidSegment) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
    }
  }
  return handle;
}

bool waitForFlag(volatile std::uint8_t* flag, int timeout_sec) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  while (*flag != K_DONE_FLAG) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// Push our table into the peer's receive buffer straight from each value's own
// memory (no staging copy): body from the table vector, header + flag from
// small locals. Order is the protocol — body, then header, then the flag last,
// each a blocking one-sided Write so the peer seeing the flag means the body
// and header already landed.
bool pushTableToPeer(MooncakeTransferEngine& engine, SegmentHandle peer,
                     const std::vector<std::uint8_t>& table,
                     const TableHeader& header, const std::uint8_t& flag,
                     std::uint64_t recvFlagOffset) {
  return writeToPeer(engine, peer, table.data(), table.size(),
                     K_HEADER_BYTES) &&
         writeToPeer(engine, peer,
                     reinterpret_cast<const std::uint8_t*>(&header),
                     K_HEADER_BYTES, 0) &&
         writeToPeer(engine, peer, &flag, 1, recvFlagOffset);
}

int runExchange(const Options& o) {
  std::vector<std::uint8_t> myTable;
  if (!readFile(o.table, myTable)) {
    std::cerr << "[node] cannot read table file: " << o.table << "\n";
    return 1;
  }
  if (myTable.size() > o.max_bytes) {
    std::cerr << "[node] table (" << myTable.size()
              << " B) exceeds --max-bytes (" << o.max_bytes
              << "); raise --max-bytes\n";
    return 1;
  }

  // Outbound framing: the table is pushed straight from its own memory, so we
  // only need the tiny header (size + checksum) and the one-byte done flag as
  // separate sources alongside it — no staging copy of the table.
  TableHeader sendHeader{myTable.size(), fnv1a(myTable.data(), myTable.size())};
  std::uint8_t sendFlag = K_DONE_FLAG;

  // Receive buffer: [header][reserved table area][flag]. Registered FIRST so it
  // is segment buffers[0] — the region the peer resolves its Writes against.
  const std::size_t recvFlagOffset = K_HEADER_BYTES + o.max_bytes;
  std::vector<std::uint8_t> recvBuf(
      recvFlagOffset + 1, 0);  // memory where the peer table will land

  auto engine = makeEngine(o);  // classic engine setup
  if (!engine) {
    std::cerr << "[node] engine init failed\n";
    return 1;
  }
  // recvBuf FIRST (buffers[0]); the send-side regions are only local Write
  // sources, registered so the same code works under RDMA (TCP ignores it).
  if (!engine->registerLocalMemory(recvBuf.data(), recvBuf.size()) ||
      !engine->registerLocalMemory(myTable.data(), myTable.size()) ||
      !engine->registerLocalMemory(&sendHeader, K_HEADER_BYTES) ||
      !engine->registerLocalMemory(&sendFlag, sizeof(sendFlag))) {
    std::cerr << "[node] registerLocalMemory failed\n";
    return 1;
  }

  auto unregisterAll = [&]() {
    engine->unregisterLocalMemory(&sendFlag);
    engine->unregisterLocalMemory(&sendHeader);
    engine->unregisterLocalMemory(myTable.data());
    engine->unregisterLocalMemory(recvBuf.data());
  };
  std::cout << "[node] '" << o.name << "' serving " << myTable.size()
            << " B at " << engine->localServerName() << "; discovering peer '"
            << o.peer << "'\n";

  // Logging silencing
  const int prevMinLogLevel = FLAGS_minloglevel;
  FLAGS_minloglevel = 3;  // FATAL only: no per-poll ERROR/WARNING spam

  SegmentHandle peer =
      openWithRetry(*engine, o.peer, o.timeout_sec);  // DYNAMIC DISCOVERY

  FLAGS_minloglevel = prevMinLogLevel;
  if (peer == kInvalidSegment) {
    std::cerr << "[node] openSegment(" << o.peer
              << ") failed (peer not discoverable within timeout)\n";
    unregisterAll();
    return 1;
  }

  const bool pushed = pushTableToPeer(*engine, peer, myTable, sendHeader,
                                      sendFlag, recvFlagOffset);
  if (!pushed) {
    std::cerr << "[node] push to peer failed\n";
  }

  // Wait for the peer to push its table into our receive buffer.
  auto* myFlag =
      reinterpret_cast<volatile std::uint8_t*>(&recvBuf[recvFlagOffset]);
  const bool peerDone = waitForFlag(myFlag, o.timeout_sec);

  bool intact = false;
  std::size_t got = 0;
  if (peerDone) {
    TableHeader rh{};
    std::memcpy(&rh, recvBuf.data(), K_HEADER_BYTES);
    if (rh.table_bytes <= o.max_bytes) {
      got = rh.table_bytes;
      intact = fnv1a(recvBuf.data() + K_HEADER_BYTES, got) == rh.checksum;
      if (!writeFile(o.out, recvBuf.data() + K_HEADER_BYTES, got)) {
        std::cerr << "[node] cannot write peer table: " << o.out << "\n";
      }
    }
  }

  unregisterAll();

  std::cout << "[node] received " << got << " B -> " << o.out
            << "; integrity: " << (intact ? "PASS" : "FAIL")
            << "; pushed ours: " << (pushed ? "yes" : "FAIL")
            << "; peer pushed: " << (peerDone ? "yes" : "TIMEOUT") << "\n";
  return (intact && peerDone && pushed) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parseArgs(argc, argv, o)) {
    usage();
    return 2;
  }
  return runExchange(o);
}
