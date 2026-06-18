// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Prefill<->decode KV-table exchange PoC for #4268. Each node advertises a
// logical name in the Mooncake Metadata Service and serves its own
// KvChunkAddressTable from a single registered host buffer; it then resolves
// the peer's name and pulls the peer's table with one-sided Reads over the
// Transfer Engine, so both sides end up holding both tables (the Mooncake
// stand-in for the MPI Connect() exchange). Control plane = name discovery;
// data plane = one-sided Read. Switching TransportProtocol to Rdma later does
// not change this flow. Raw bytes (no compression): the table moves once at
// startup. Host RAM only; built only with Mooncake.

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

// Written into a node's served buffer by the peer once it has read that node's
// table. A node waits for its own flag before tearing down, so it never
// unregisters a table the peer is still reading. TCP ordering => seeing the
// flag means the table bytes already landed.
constexpr std::uint8_t K_DONE_FLAG = 0xAB;
constexpr int K_DISCOVERY_POLL_MS = 100;

// Self-describing header prepended to a served table, so the peer learns the
// table size from a fixed-size first Read and needs no out-of-band length.
// Both nodes are the same architecture, so the raw struct goes on the wire.
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
  std::string out;       // where to write the peer's pulled table
  int timeout_sec = 30;
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
         "  --out PATH        where to write the peer's pulled .pb table\n"
         "  [--timeout-sec S] (default 30)\n"
         "\n"
         "Symmetric: every node publishes its own table and pulls the peer's, "
         "so both\n"
         "sides hold both tables (the Mooncake stand-in for the MPI Connect() "
         "exchange).\n"
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

// One-sided transfer through a freshly registered local region. Mooncake
// requires the local side of a transfer to be registered too, so register →
// submit → unregister keeps each step self-contained.
bool transferRegion(MooncakeTransferEngine& engine, SegmentHandle peer,
                    TransferOp op, std::uint8_t* local, std::size_t len,
                    std::uint64_t remote_offset) {
  if (!engine.registerLocalMemory(local, len)) return false;
  TransferRequest req;
  req.op = op;
  req.local_addr = local;
  req.target = peer;
  req.target_offset = remote_offset;
  req.length = len;
  const bool ok = engine.submitAndWait(req).state == TransferState::Completed;
  engine.unregisterLocalMemory(local);
  return ok;
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

// One contiguous buffer = [header][table][done flag] so it all resolves against
// this node's single registered region (offset 0).
std::vector<std::uint8_t> buildServedBuffer(
    const std::vector<std::uint8_t>& table) {
  std::vector<std::uint8_t> buf(K_HEADER_BYTES + table.size() + 1, 0);
  const TableHeader hdr{table.size(), fnv1a(table.data(), table.size())};
  std::memcpy(buf.data(), &hdr, K_HEADER_BYTES);
  std::memcpy(buf.data() + K_HEADER_BYTES, table.data(), table.size());
  return buf;
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

// Pull the peer's table: fixed-size header Read first (to learn its size and
// checksum), then the table body. Verifies integrity against the header.
bool fetchPeerTable(MooncakeTransferEngine& engine, SegmentHandle peer,
                    std::vector<std::uint8_t>& out) {
  TableHeader hdr{};
  if (!transferRegion(engine, peer, TransferOp::Read,
                      reinterpret_cast<std::uint8_t*>(&hdr), K_HEADER_BYTES,
                      0)) {
    return false;
  }
  out.resize(hdr.table_bytes);
  if (!transferRegion(engine, peer, TransferOp::Read, out.data(), out.size(),
                      K_HEADER_BYTES)) {
    return false;
  }
  return fnv1a(out.data(), out.size()) == hdr.checksum;
}

int runExchange(const Options& o) {
  std::vector<std::uint8_t> myTable;
  if (!readFile(o.table, myTable)) {
    std::cerr << "[node] cannot read table file: " << o.table << "\n";
    return 1;
  }
  std::vector<std::uint8_t> served = buildServedBuffer(myTable);

  auto engine = makeEngine(o);
  if (!engine) {
    std::cerr << "[node] engine init failed\n";
    return 1;
  }
  if (!engine->registerLocalMemory(served.data(), served.size())) {
    std::cerr << "[node] registerLocalMemory failed\n";
    return 1;
  }
  std::cout << "[node] '" << o.name << "' serving " << myTable.size()
            << " B at " << engine->localServerName()
            << "; discovering peer '" << o.peer << "'\n";

  SegmentHandle peer = openWithRetry(*engine, o.peer, o.timeout_sec);
  if (peer == kInvalidSegment) {
    std::cerr << "[node] openSegment(" << o.peer
              << ") failed (peer not discoverable within timeout)\n";
    engine->unregisterLocalMemory(served.data());
    return 1;
  }

  std::vector<std::uint8_t> peerTable;
  const bool intact = fetchPeerTable(*engine, peer, peerTable);
  if (!writeFile(o.out, peerTable.data(), peerTable.size())) {
    std::cerr << "[node] cannot write peer table: " << o.out << "\n";
    engine->unregisterLocalMemory(served.data());
    return 1;
  }

  // Tell the peer we finished reading its table (flag sits past its body).
  std::uint8_t done = K_DONE_FLAG;
  transferRegion(*engine, peer, TransferOp::Write, &done, 1,
                 K_HEADER_BYTES + peerTable.size());

  // Wait until the peer has finished reading ours before unregistering.
  auto* myFlag =
      reinterpret_cast<volatile std::uint8_t*>(&served[K_HEADER_BYTES +
                                                       myTable.size()]);
  const bool peerDone = waitForFlag(myFlag, o.timeout_sec);
  engine->unregisterLocalMemory(served.data());

  std::cout << "[node] pulled " << peerTable.size() << " B -> " << o.out
            << "; integrity: " << (intact ? "PASS" : "FAIL")
            << "; peer read ours: " << (peerDone ? "yes" : "TIMEOUT") << "\n";
  return (intact && peerDone) ? 0 : 1;
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
