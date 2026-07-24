// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Acceptance harness for the real KV-cache migration path.
//
// Two processes (a prefill "sender" and a decode "receiver") drive a full
// table-addressed migration over a real TCP control channel + real Mooncake:
//
//   prefill device DRAM --read(DRISC)--> staging
//       --Mooncake one-sided WRITE(free bounce section)--> decode bounce buffer
//       --drain(DRISC)--> decode device DRAM
//
//   control channel (TCP):  BeginMigration -> BounceReady
//                           -> per window: WindowReady -> WindowAck
//                           -> DoneMarker -> Ack
//
// This exercises the real address layer: per-window bounce-section descriptors
// built from a KV table, device-group fan-out, per-(device,channel) physical
// addresses, and the KvMigration{Sender,Receiver} orchestrators.
//
// Modes:
//   --mode host    : a host-backed device store stands in for device DRAM.
//                    Runs on any --mooncake build, no hardware. The byte
//                    content is a deterministic pattern keyed by (layer,pos),
//                    so the receiver verifies without an out-of-band payload.
//   --mode device  : real TT device DRAM via UMD (needs --blaze --mooncake +
//                    hardware). The sender first writes the pattern into its
//                    source devices, then migrates; the receiver verifies.
//
//   --table builtin   : a small hand-built reduced table (2 layers, 2-replica
//                       groups, multi-channel) — both sides build it locally.
//   --table <pfx>     : real tables loaded from protobuf (needs
//   -DENABLE_KV_TABLE
//                       =ON). Sender loads "<pfx>" as the prefill table; the
//                       receiver loads it as the decode table and ships its
//                       bytes to the sender over the control channel.
//
// Only built when Mooncake is in the build (TARGET transfer_engine).
//
// Driven two ways, sharing one binary:
//   * gtest (ctest `TransportKvMigrationE2E`): the default `--mode host
//     --table builtin` round trip. The test process runs the receiver in-line
//     (so byte mismatches surface as gtest EXPECT_EQ diffs) and fork()/execv()s
//     this same binary as the sender worker (--role sender) — two real Mooncake
//     processes, no hardware. This is the uniform-with-the-suite entry point.
//   * worker / shell harness: invoked with `--role sender|receiver`, each
//     process runs one side.
//     tests/e2e/scripts/run_transport_kv_migration_e2e.sh uses this for the
//     `--mode device` (hardware) and real-table
//     (-DENABLE_KV_TABLE=ON) paths that can't run in one gtest process.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <limits.h>
#include <netinet/in.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/double_pinned_buffer.hpp"
#include "transport/drisc_device_io.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_device_io.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_endpoints.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/kv_migration_orchestrator.hpp"
#include "transport/kv_staging_pool.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

// Kafka-trigger mode: drive the migration through the same components as the
// unified worker (KvMigrationWorker + MooncakeMigrationExecutor). Compiled only
// in a KAFKA_ENABLED build; the default cli trigger needs none of this.
#ifdef TT_E2E_WITH_KAFKA
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "runtime/worker/kv_migration_worker.hpp"
#include "transport/mooncake_migration_executor.hpp"
#endif

namespace {

using namespace tt::transport;

// --- Minimal length-framed TCP control transport ---------------------------
// Self-contained so the harness links only transport_lib. One sendRawData ==
// one receiveRawData (4-byte length prefix + payload).
//
// This is a loopback-only prototype: readAll() blocks until a full message and
// returns empty only on close. It does not override tryReceiveMessage(), so it
// rides the ISocketTransport default (DATA on a message, CLOSED otherwise via
// isConnected()) — blocking reads mean it never needs to report NO_DATA. The
// production TcpSocketTransport IS now a drop-in: it is non-blocking on no-data
// but reports that as NO_DATA through tryReceiveMessage(), which the channel
// waits/retries on instead of treating as a close.
//
// NOTE (productionization, not for now): the length prefix here is a raw
// host-order uint32, whereas TcpSocketTransport frames with htonl/ntohl
// (network order). Harmless today (both ends are TcpControl on one host), but
// when this path is wired for real, drop this harness transport in favor of
// TcpSocketTransport on both ends — that removes this duplication and aligns
// the wire format in a single step.
class TcpControl : public tt::sockets::ISocketTransport {
 public:
  /// @param timeout_sec per-operation socket timeout (0 = block indefinitely).
  ///        Bounds accept() and every read/write so a peer that hangs without
  ///        closing the connection can't wedge the harness forever.
  explicit TcpControl(int timeoutSec = 0) : timeout_sec(timeoutSec) {}

  ~TcpControl() override {
    if (conn >= 0) ::close(conn);
    if (listen >= 0) ::close(listen);
  }

  bool initializeAsServer(uint16_t port) override {
    listen = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen < 0) return false;
    int yes = 1;
    ::setsockopt(listen, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      return false;
    }
    if (::listen(listen, 1) != 0) return false;
    applyTimeout(listen);  // bound the wait for the sender to connect
    conn = ::accept(listen, nullptr, nullptr);
    if (conn < 0) return false;
    applyTimeout(conn);  // bound every subsequent read/write
    return true;
  }

  bool initializeAsClient(const std::string& host, uint16_t port) override {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    for (int attempt = 0; attempt < 600; ++attempt) {  // ~60s of retries
      conn = ::socket(AF_INET, SOCK_STREAM, 0);
      if (conn >= 0 && ::connect(conn, reinterpret_cast<sockaddr*>(&addr),
                                 sizeof(addr)) == 0) {
        applyTimeout(conn);  // bound every subsequent read/write
        return true;
      }
      if (conn >= 0) {
        ::close(conn);
        conn = -1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  void start() override {}
  void stop() override {}
  // Flips false once a read sees the peer close (::read == 0), so the
  // ISocketTransport default tryReceiveMessage reports CLOSED and a run() loop
  // (used by the bounce receiver, whose message count is variable) returns
  // promptly instead of spinning to the socket timeout. A read *timeout*
  // (::read < 0) does NOT mark closed — the peer may just be slow.
  bool isConnected() const override { return conn >= 0 && !peer_closed; }
  std::string getStatus() const override { return "tcp-control"; }

  bool sendRawData(std::span<const uint8_t> data) override {
    const uint32_t len = static_cast<uint32_t>(data.size());
    return writeAll(&len, 4) && writeAll(data.data(), len);
  }
  std::vector<uint8_t> receiveRawData() override {
    uint32_t len = 0;
    if (!readAll(&len, 4)) return {};
    std::vector<uint8_t> buf(len);
    if (len && !readAll(buf.data(), len)) return {};
    return buf;
  }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}

 private:
  // SO_RCVTIMEO/SO_SNDTIMEO so a blocked recv/send (peer hung without closing)
  // returns EAGAIN after timeout_sec_ instead of waiting forever. A read/write
  // that times out is treated as failure → receiveRawData() returns {} →
  // receive() yields nullopt → serveOne()/migrate() abort cleanly.
  void applyTimeout(int fd) const {
    if (timeout_sec <= 0 || fd < 0) return;
    timeval tv{};
    tv.tv_sec = timeout_sec;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }

  bool writeAll(const void* p, std::size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    std::size_t off = 0;
    while (off < n) {
      const ssize_t w = ::write(conn, b + off, n - off);
      if (w <= 0) return false;
      off += static_cast<std::size_t>(w);
    }
    return true;
  }
  bool readAll(void* p, std::size_t n) {
    auto* b = static_cast<uint8_t*>(p);
    std::size_t off = 0;
    while (off < n) {
      const ssize_t r = ::read(conn, b + off, n - off);
      if (r == 0) {  // peer closed the connection
        peer_closed = true;
        return false;
      }
      if (r < 0) return false;  // error / timeout (EAGAIN)
      off += static_cast<std::size_t>(r);
    }
    return true;
  }

  int timeout_sec = 0;
  int listen = -1;
  int conn = -1;
  bool peer_closed = false;
};

// --- Host-backed device store (host mode) ----------------------------------
// Byte-addressed, modelling a contiguous DRAM address space per device: a read
// of [n, n+size) assembles consecutive bytes regardless of the granularity they
// were written at. This matters because the bounce buffer sender merges
// source-contiguous chunks into one large device read, and the
// bounce receiver drains one merged write per slot — a store keyed by exact
// (device, noc) would miss those (a 256 B read finding only a 64 B chunk).
class HostDeviceIo : public IDeviceIo {
 public:
  bool read(LocalDeviceId d, NocAddr n, std::size_t size, void* host) override {
    const auto di = store.find(d);
    if (di == store.end()) return false;
    auto* out = static_cast<uint8_t*>(host);
    for (std::size_t i = 0; i < size; ++i) {
      const auto bi = di->second.find(n + i);
      if (bi == di->second.end()) return false;  // gap -> fault, like real DRAM
      out[i] = bi->second;
    }
    return true;
  }
  bool write(LocalDeviceId d, NocAddr n, const void* host,
             std::size_t size) override {
    auto& m = store[d];
    const auto* p = static_cast<const uint8_t*>(host);
    for (std::size_t i = 0; i < size; ++i) m[n + i] = p[i];
    return true;
  }

 private:
  std::map<LocalDeviceId, std::map<NocAddr, uint8_t>> store;
};

// --- Options ---------------------------------------------------------------
// One decode host the sender fans out to (multi-host mode).
struct Peer {
  std::string host;  // logical host tag in the decode table.
  std::string ip;    // control-channel address.
  uint16_t port = 0;
};

struct Options {
  std::string role;           // sender | receiver
  std::string mode = "host";  // host | device
  std::string transport =
      "bounce";                  // RDMA-over-host bounce buffer (the only path)
  std::string protocol = "tcp";  // Mooncake transport: tcp | rdma (needs a NIC)
  std::string table = "builtin";  // builtin | builtin2 | <protobuf path>
  std::string decode_table;       // sender, real-table mode: decode .pb path
  std::string control_host = "127.0.0.1";
  uint16_t control_port = 18650;
  std::string mooncake_name;  // this proc's Mooncake host:port
  // Mooncake segment metadata: "P2PHANDSHAKE" (default) means the sender opens
  // the receiver's segment by directly connecting to the ip:port it advertised
  // in BounceReady; a real metadata-service URI (e.g.
  // http://HOST:PORT/metadata, etcd://…) registers/resolves segments through
  // that service instead. The TCP control channel is still a direct
  // --control-host/--peer-control dial either way — this only changes Mooncake
  // data-plane segment discovery.
  std::string metadata = "P2PHANDSHAKE";
  std::string prefill_host = "prefill";
  std::string decode_host = "decode";
  uint32_t slot = 5;  // symmetric shorthand: sets both src_slot and dst_slot
  uint32_t layer_begin = 0, layer_end = 2;  // shared (layers map 1:1)
  uint32_t pos_begin = 0, pos_end = 128;  // symmetric shorthand for src+dst pos
  // Asymmetric overrides (UINT32_MAX = unset -> fall back to the shorthand
  // above). Drive a position shift and/or a cross-slot migration.
  uint32_t src_slot = UINT32_MAX, dst_slot = UINT32_MAX;
  uint32_t src_pos_begin = UINT32_MAX, src_pos_end = UINT32_MAX;
  uint32_t dst_pos_begin = UINT32_MAX, dst_pos_end = UINT32_MAX;
  uint64_t uuid = 0x5151;
  int timeout_sec = 60;
  // Bounce transport geometry (receiver side). 0 => defaultBounceGeometry(). A
  // tiny bounce buffer (e.g. 1 slot) forces the multi-window credit handshake
  // over the wire.
  uint32_t bounce_section_count = 0;
  uint64_t bounce_section_size = 0;
  // Seed a deterministic dummy blob at the source and byte-verify the
  // destination even in real-table mode (a mechanism test with NO model loaded:
  // the table's addresses are used as scratch). Always on for builtin.
  bool seed_verify = false;
  // Sender multi-host fan-out: one decode peer per --peer-control HOST=ip:port.
  // Empty => single-host path (--control-host/--control-port/--decode-host).
  std::vector<Peer> peers;
  // Optional FabricNode -> UMD chip map file for device mode. Each line
  // "mesh chip umd_chip_id"; absent => placeholder chip = device & 0xFFFF.
  std::string device_map;
  // How the sender's migration is triggered: "cli" = call migrate() directly
  // (default); "kafka" = drive it through KvMigrationWorker +
  // MooncakeMigrationExecutor (the unified worker's data path), self-producing
  // the request and waiting for the ack. Kafka mode needs a KAFKA_ENABLED
  // build.
  std::string trigger = "cli";
  std::string kafka_brokers = "localhost:9092";
  std::string kafka_request_topic = "kv-migration-requests";
  std::string kafka_ack_topic = "kv-migration-acks";
  std::string kafka_group = "migration-workers";
};

void usage() {
  std::cerr
      << "usage: transport_kv_migration_e2e --role sender|receiver "
         "--mooncake-name HOST:PORT\n"
         "  --control-host H (sender; default 127.0.0.1)  --control-port P "
         "(default 18650)\n"
         "  --metadata URI       (Mooncake segment metadata; default "
         "P2PHANDSHAKE = direct peer connect. A service URI e.g. "
         "http://HOST:PORT/metadata resolves segments via that service; the "
         "control channel stays a direct dial)\n"
         "  --mode host|device   --table builtin|builtin2|<protobuf-path>\n"
         "  --transport bounce   (RDMA-over-host bounce buffer; the only "
         "path)\n"
         "  --bounce-sections N --bounce-section-size B   (bounce receiver "
         "geometry; 0 = "
         "default)\n"
         "  (device mode uses DRISC NOC-DMA; needs MIGRATION_DRISC_SERVICE_ELF "
         "+ HW)\n"
         "  --protocol tcp|rdma   (Mooncake transport; rdma needs an RDMA NIC "
         "+ "
         "an RDMA-enabled Mooncake build)\n"
         "  --prefill-host NAME  --decode-host NAME\n"
         "  --peer-control HOST=ip:port  (sender, repeatable) fan out to N "
         "decode hosts\n"
         "  --device-map FILE   (device mode) 'mesh chip umd_chip_id' per "
         "line\n"
         "  --slot N  --layer-begin N --layer-end N  --pos-begin N --pos-end "
         "N\n"
         "  (asymmetric overrides) --src-slot N --dst-slot N "
         "--src-pos-begin N --src-pos-end N --dst-pos-begin N --dst-pos-end N\n"
         "  --uuid N  --timeout-sec S\n"
         "  --seed-verify   seed a dummy blob at the source + byte-verify the\n"
         "                  destination even for a real table (no model "
         "needed;\n"
         "                  uses the real addresses as scratch)\n";
}

bool parseArgs(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nextStr = [&](std::string& d) {
      if (i + 1 < argc) {
        d = argv[++i];
        return true;
      }
      return false;
    };
    auto nextU = [&](auto& d) {
      if (i + 1 < argc) {
        d = static_cast<std::decay_t<decltype(d)>>(
            std::strtoull(argv[++i], nullptr, 0));
        return true;
      }
      return false;
    };
    if (a == "--role" && nextStr(o.role)) continue;
    if (a == "--mode" && nextStr(o.mode)) continue;
    if (a == "--transport" && nextStr(o.transport)) continue;
    if (a == "--protocol" && nextStr(o.protocol)) continue;
    if (a == "--bounce-sections" && nextU(o.bounce_section_count)) continue;
    if (a == "--bounce-section-size" && nextU(o.bounce_section_size)) continue;
    if (a == "--table" && nextStr(o.table)) continue;
    if (a == "--decode-table" && nextStr(o.decode_table)) continue;
    if (a == "--control-host" && nextStr(o.control_host)) continue;
    if (a == "--mooncake-name" && nextStr(o.mooncake_name)) continue;
    if (a == "--metadata" && nextStr(o.metadata)) continue;
    if (a == "--prefill-host" && nextStr(o.prefill_host)) continue;
    if (a == "--decode-host" && nextStr(o.decode_host)) continue;
    if (a == "--control-port" && nextU(o.control_port)) continue;
    if (a == "--slot" && nextU(o.slot)) continue;
    if (a == "--layer-begin" && nextU(o.layer_begin)) continue;
    if (a == "--layer-end" && nextU(o.layer_end)) continue;
    if (a == "--pos-begin" && nextU(o.pos_begin)) continue;
    if (a == "--pos-end" && nextU(o.pos_end)) continue;
    // Asymmetric overrides (position shift / cross-slot).
    if (a == "--src-slot" && nextU(o.src_slot)) continue;
    if (a == "--dst-slot" && nextU(o.dst_slot)) continue;
    if (a == "--src-pos-begin" && nextU(o.src_pos_begin)) continue;
    if (a == "--src-pos-end" && nextU(o.src_pos_end)) continue;
    if (a == "--dst-pos-begin" && nextU(o.dst_pos_begin)) continue;
    if (a == "--dst-pos-end" && nextU(o.dst_pos_end)) continue;
    if (a == "--uuid" && nextU(o.uuid)) continue;
    if (a == "--timeout-sec" && nextU(o.timeout_sec)) continue;
    if (a == "--seed-verify") {
      o.seed_verify = true;
      continue;
    }
    if (a == "--device-map" && nextStr(o.device_map)) continue;
    if (a == "--trigger" && nextStr(o.trigger)) continue;
    if (a == "--kafka-brokers" && nextStr(o.kafka_brokers)) continue;
    if (a == "--kafka-request-topic" && nextStr(o.kafka_request_topic))
      continue;
    if (a == "--kafka-ack-topic" && nextStr(o.kafka_ack_topic)) continue;
    if (a == "--kafka-group" && nextStr(o.kafka_group)) continue;
    if (a == "--peer-control") {
      std::string spec;
      if (!nextStr(spec)) return false;
      const auto eq = spec.find('=');
      const auto colon = spec.rfind(':');
      if (eq == std::string::npos || colon == std::string::npos || colon < eq) {
        std::cerr << "--peer-control must be HOST=ip:port, got: " << spec
                  << "\n";
        return false;
      }
      Peer p;
      p.host = spec.substr(0, eq);
      p.ip = spec.substr(eq + 1, colon - eq - 1);
      p.port = static_cast<uint16_t>(
          std::strtoul(spec.substr(colon + 1).c_str(), nullptr, 10));
      o.peers.push_back(p);
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if ((o.role != "sender" && o.role != "receiver") || o.mooncake_name.empty()) {
    return false;
  }
  return true;
}

// Bounce geometry from the options, falling back to the env/default for unset
// (0) fields. Only the receiver uses it; the sender learns it over the wire.
BounceGeometry bounceGeoOf(const Options& o) {
  const BounceGeometry def = defaultBounceGeometry();
  return BounceGeometry{
      o.bounce_section_count ? o.bounce_section_count : def.section_count,
      o.bounce_section_size ? o.bounce_section_size : def.section_size};
}

MigrationRequest requestOf(const Options& o) {
  auto pick = [](uint32_t v, uint32_t d) { return v == UINT32_MAX ? d : v; };
  return MigrationRequest{pick(o.src_slot, o.slot),
                          pick(o.dst_slot, o.slot),
                          o.layer_begin,
                          o.layer_end,
                          pick(o.src_pos_begin, o.pos_begin),
                          pick(o.src_pos_end, o.pos_end),
                          pick(o.dst_pos_begin, o.pos_begin),
                          pick(o.dst_pos_end, o.pos_end)};
}

// --- Built-in reduced tables (no protobuf needed) --------------------------
constexpr uint32_t K_CHUNK = 64;

// Deterministic content of `size` bytes for a (layer, position) chunk. `size`
// is the chunk's real size (K_CHUNK for builtin, chunk_size_bytes for the real
// table), so this seeds/verifies whatever chunk size the table declares.
std::vector<uint8_t> patternN(uint32_t layer, uint32_t pos, uint64_t size) {
  std::vector<uint8_t> v(size);
  for (uint64_t i = 0; i < size; ++i) {
    v[i] = static_cast<uint8_t>(layer * 40 + pos + i);
  }
  return v;
}

// The bytes drained to a dst (layer, position) were seeded at the matching
// *src* position — chunks pair by ordinal within their ranges. Map dst -> src
// so verification matches under a position shift (symmetric => srcPos==dstPos).
std::vector<uint8_t> expectedForDst(const MigrationRequest& req, uint32_t step,
                                    uint32_t layer, uint32_t dstPos,
                                    uint64_t size) {
  const uint32_t srcStart =
      req.src_position_begin - (req.src_position_begin % step);
  const uint32_t dstStart =
      req.dst_position_begin - (req.dst_position_begin % step);
  return patternN(layer, srcStart + (dstPos - dstStart), size);
}

// GB/s for `bytes` moved in `ns` (1 byte/ns == 1 GB/s), 0 if ns==0.
double gbps(uint64_t bytes, uint64_t ns) {
  return ns == 0 ? 0.0 : static_cast<double>(bytes) / static_cast<double>(ns);
}
double ms(uint64_t ns) { return static_cast<double>(ns) / 1.0e6; }

uint64_t nsSince(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

// Logical KV volume of a plan: one copy per chunk (ignores replica fan-out) —
// the "useful" bytes migrated, derived from the plan (no data-plane counters).
uint64_t planLogicalBytes(const HostKvPlan& plan) {
  uint64_t b = 0;
  for (const auto& c : plan.chunks) {
    if (!c.targets.empty()) b += c.targets.front().size_bytes;
  }
  return b;
}

// Total wire/device bytes (fan-out included): every replica target.
uint64_t planWireBytes(const HostKvPlan& plan) {
  uint64_t b = 0;
  for (const auto& c : plan.chunks) {
    for (const auto& t : c.targets) b += t.size_bytes;
  }
  return b;
}

// FabricNode -> UMD chip map, keyed by encodeDevice. Empty on no path /
// unreadable, so callers fall back to the placeholder (device & 0xFFFF).
std::unordered_map<LocalDeviceId, int> loadDeviceMap(const std::string& path) {
  std::unordered_map<LocalDeviceId, int> m;
  if (path.empty()) return m;
  std::ifstream f(path);
  if (!f.good()) {
    std::cerr << "[device-map] cannot open " << path
              << "; falling back to placeholder chip ids\n";
    return m;
  }
  uint32_t mesh = 0, chip = 0;
  uint64_t umd = 0;
  while (f >> mesh >> chip >> umd) {
    m[encodeDevice(FabricNode{mesh, chip})] = static_cast<int>(umd);
  }
  std::cerr << "[device-map] loaded " << m.size() << " entries from " << path
            << "\n";
  return m;
}

KvTableConfig builtinConfig() {
  KvTableConfig c;
  c.num_layers = 2;
  c.num_slots = 8;
  c.max_sequence_length = 128;  // positions 0,32,64,96
  c.chunk_n_tokens = 32;
  c.chunk_size_bytes = K_CHUNK;
  return c;
}

// 2 layers, each on a 2-replica device group, 4 contiguous chunks per channel.
std::shared_ptr<InMemoryKvTable> builtinTable(const std::string& host,
                                              FabricNode l0a, FabricNode l0b,
                                              FabricNode l1a, FabricNode l1b,
                                              uint64_t base0, uint32_t ch0,
                                              uint64_t base1, uint32_t ch1) {
  auto t = std::make_shared<InMemoryKvTable>(builtinConfig());
  const uint32_t g0 = t->addDeviceGroup({l0a, l0b});
  const uint32_t g1 = t->addDeviceGroup({l1a, l1b});
  for (const auto& n : {l0a, l0b, l1a, l1b}) t->setHost(n, host);
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    t->setChunk(5, 0, p,
                {makeNocAddr(ch0, base0 + idx * K_CHUNK), K_CHUNK, g0});
    t->setChunk(5, 1, p,
                {makeNocAddr(ch1, base1 + idx * K_CHUNK), K_CHUNK, g1});
  }
  return t;
}

constexpr uint64_t K_DRAM_BASE = 0x200000;  // 2 MiB, comfortably past reserved

std::shared_ptr<InMemoryKvTable> builtinPrefill() {
  return builtinTable("prefill", {1, 0}, {1, 1}, {1, 2}, {1, 3},
                      K_DRAM_BASE + 0x0000, 0, K_DRAM_BASE + 0x1000, 0);
}
std::shared_ptr<InMemoryKvTable> builtinDecode() {
  return builtinTable("decode", {2, 0}, {2, 1}, {2, 2}, {2, 3},
                      K_DRAM_BASE + 0x2000, 0, K_DRAM_BASE + 0x3000, 1);
}

// Two-host split decode table for the multi-host path: layer 0 lives on
// "decode-0" (mesh 2), layer 1 on "decode-1" (mesh 3). A 2-layer request fans
// out to both hosts. Fixed tags so sender and both receivers build the same
// table from their own --decode-host.
constexpr const char* K_DECODE0 = "decode-0";
constexpr const char* K_DECODE1 = "decode-1";
std::shared_ptr<InMemoryKvTable> builtinDecodeSplit() {
  auto t = std::make_shared<InMemoryKvTable>(builtinConfig());
  const FabricNode l0a{2, 0}, l0b{2, 1}, l1a{3, 0}, l1b{3, 1};
  const uint32_t g0 = t->addDeviceGroup({l0a, l0b});
  const uint32_t g1 = t->addDeviceGroup({l1a, l1b});
  t->setHost(l0a, K_DECODE0);
  t->setHost(l0b, K_DECODE0);
  t->setHost(l1a, K_DECODE1);
  t->setHost(l1b, K_DECODE1);
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    t->setChunk(
        5, 0, p,
        {makeNocAddr(0, K_DRAM_BASE + 0x2000 + idx * K_CHUNK), K_CHUNK, g0});
    t->setChunk(
        5, 1, p,
        {makeNocAddr(1, K_DRAM_BASE + 0x3000 + idx * K_CHUNK), K_CHUNK, g1});
  }
  return t;
}

// Build a DRISC NOC-DMA device I/O over the devices touched by `plan` (the
// IDeviceIo). Each addDevice opens the coexistence UMD handle and launches a
// per-chip DRISC service kernel (needs MIGRATION_DRISC_SERVICE_ELF + HW); on
// failure it logs and read/write report failure for that device. Chip id from
// the device map, else the placeholder (FabricNode chip in the low 16 bits).
std::unique_ptr<DriscDeviceIo> makeDrisc(
    const HostKvPlan& plan,
    const std::unordered_map<LocalDeviceId, int>& devmap) {
  auto io = std::make_unique<DriscDeviceIo>();
  for (const auto& chunk : plan.chunks) {
    for (const auto& t : chunk.targets) {
      if (!io->hasDevice(t.device)) {
        const auto it = devmap.find(t.device);
        const int chip = (it != devmap.end())
                             ? it->second
                             : static_cast<int>(t.device & 0xFFFFu);
        io->addDevice(t.device, chip);
      }
    }
  }
  return io;
}

#ifdef TT_E2E_WITH_KAFKA
// Trigger the migration through the unified worker's data path: a
// KvMigrationWorker consumes the request and runs MooncakeMigrationExecutor
// over the multi-host sender, then publishes the ack. Self-produces the request
// (an external producer like migration_cli.py works the same — real topic) and
// waits for the SUCCESSFUL ack. Returns migration success.
bool driveViaKafka(KvMigrationMultiHostSender& multiHost,
                   const MigrationRequest& req, const Options& o) {
  using namespace tt::messaging;
  // Ack reader first (fresh group, read from the start) so it can't miss an ack
  // produced before its group assignment settles.
  KafkaConsumer ackConsumer(KafkaConsumerConfig{
      o.kafka_brokers, o.kafka_ack_topic, o.kafka_group + "-e2e-ackwait"});
  (void)ackConsumer.receive(200);  // prime subscription/assignment.

  auto executor =
      std::make_unique<tt::transport::MooncakeMigrationExecutor>(multiHost);
  auto consumer = std::make_unique<KafkaConsumer>(KafkaConsumerConfig{
      o.kafka_brokers, o.kafka_request_topic, o.kafka_group});
  auto ackProducer = std::make_unique<KafkaProducer>(
      KafkaProducerConfig{o.kafka_brokers, o.kafka_ack_topic});
  tt::worker::KvMigrationWorker worker(
      std::move(consumer), std::move(ackProducer), std::move(executor));
  worker.start();

  KafkaProducer reqProducer(
      KafkaProducerConfig{o.kafka_brokers, o.kafka_request_topic});
  const MigrationRequestMessage m{o.uuid,
                                  req.src_slot,
                                  req.dst_slot,
                                  req.layer_begin,
                                  req.layer_end,
                                  req.src_position_begin,
                                  req.src_position_end,
                                  req.dst_position_begin,
                                  req.dst_position_end};
  std::string err;
  if (!reqProducer.send(serialize(m), &err)) {
    std::cerr << "[sender] kafka produce failed: " << err << "\n";
    worker.stop();
    return false;
  }
  std::cout << "[sender] produced migration request uuid=" << o.uuid << " -> "
            << o.kafka_request_topic << "\n";

  bool ok = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(o.timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    auto raw = ackConsumer.receive(200);
    if (!raw) continue;
    auto ack = parseMigrationResponse(*raw);
    if (ack && ack->migration_id == o.uuid) {
      ok = ack->status == tt::services::MigrationStatus::SUCCESSFUL;
      std::cout << "[sender] ack uuid=" << o.uuid
                << " status=" << static_cast<int>(ack->status) << "\n";
      break;
    }
  }
  if (!ok) std::cerr << "[sender] no SUCCESSFUL ack before timeout\n";
  worker.stop();
  return ok;
}
#endif  // TT_E2E_WITH_KAFKA

// --- Sender ----------------------------------------------------------------
int runSender(const Options& o) {
  // Local (prefill) table + the decode table (whole cluster) + the request.
  std::shared_ptr<const IKvTable> prefill;
  std::shared_ptr<const IKvTable> decode;
  std::string prefillHost = o.prefill_host;
  std::string decodeHost = o.decode_host;

  if (o.table == "builtin") {
    prefill = builtinPrefill();
    decode = builtinDecode();
    prefillHost = "prefill";
    decodeHost = "decode";
  } else if (o.table == "builtin2") {
    prefill = builtinPrefill();
    decode = builtinDecodeSplit();
    prefillHost = "prefill";
  } else {
    auto pre = KvChunkAddressTableAdapter::fromProtobufFile(o.table);
    auto dec =
        o.decode_table.empty()
            ? nullptr
            : KvChunkAddressTableAdapter::fromProtobufFile(o.decode_table);
    if (!pre || !dec) {
      std::cerr
          << "[sender] real-table mode needs --table <prefill.pb> and "
             "--decode-table <decode.pb>, and a -DENABLE_KV_TABLE=ON build\n";
      return 1;
    }
    prefill = std::shared_ptr<const IKvTable>(std::move(pre));
    decode = std::shared_ptr<const IKvTable>(std::move(dec));
  }
  const MigrationRequest req = requestOf(o);

  if (o.trigger == "kafka" && o.peers.empty()) {
    std::cerr << "[sender] --trigger kafka requires at least one "
                 "--peer-control (it drives the multi-host sender)\n";
    return 1;
  }

  // Device or host I/O for the prefill side.
  HostDeviceIo hostIo;
  const HostKvPlan srcPlan =
      buildHostPlan(*prefill, prefillHost, req.srcSlice());
  if (srcPlan.empty()) {
    std::cerr << "[sender] no prefill chunks for request on host '"
              << prefillHost << "'\n";
    return 1;
  }
  // Device backend: DRISC NOC-DMA. addDevice launches the per-chip service
  // kernel here — the composition-root DRISC init. host mode uses the store.
  std::unique_ptr<DriscDeviceIo> drisc;
  const bool useDrisc = (o.mode == "device");
  if (useDrisc) {
    drisc = makeDrisc(srcPlan, loadDeviceMap(o.device_map));
  }
  IDeviceIo& dev = useDrisc ? static_cast<IDeviceIo&>(*drisc)
                            : static_cast<IDeviceIo&>(hostIo);

  // On DRISC, staging is double-pinned: the same buffer the engine ibv_reg_mr's
  // is also NOC-mapped, so device reads DMA straight into staging.
  // deviceMap is the DRISC registrar (null otherwise); the pool is built after
  // the engine is up, below.
  DeviceMapFn deviceMap;
  if (useDrisc) {
    DriscDeviceIo* d = drisc.get();
    deviceMap = [d](void* va, std::size_t bytes) {
      d->registerHostRegion(va, bytes);
    };
  }

  // Seed the source with the deterministic pattern so the receiver can verify.
  // Always for builtin(2); for a real table only under --seed-verify (a
  // no-model mechanism test that uses the table's real addresses as scratch).
  const bool seed =
      o.table == "builtin" || o.table == "builtin2" || o.seed_verify;
  if (seed) {
    for (const auto& chunk : srcPlan.chunks) {
      for (const auto& t : chunk.targets) {
        const auto bytes = patternN(chunk.layer, chunk.position, t.size_bytes);
        dev.write(t.device, t.noc_addr, bytes.data(), bytes.size());
      }
    }
  }

  // Mooncake engine (storage backend unused by the migrator; pass a host one).
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.local_server_name = o.mooncake_name;
  cfg.metadata_uri = o.metadata;
  // C1: TCP by default; RDMA when requested (needs an RDMA NIC + an
  // RDMA-enabled Mooncake build). The bounce data plane is transport-agnostic —
  // under RDMA the double-pinned staging/bounce buffer become the ibv_reg_mr'd
  // MRs, and the per-window WRITE batch is one-sided RDMA instead of TCP.
  cfg.protocol =
      (o.protocol == "rdma") ? TransportProtocol::RDMA : TransportProtocol::TCP;
  if (!engine->init(cfg)) {
    std::cerr << "[sender] Mooncake engine init failed\n";
    return 1;
  }

  // Double-pinned staging when on DRISC (else null -> the sender self-builds a
  // plain engine-only pool). Shared across the sender(s) below.
  std::shared_ptr<KvStagingPool> staging =
      deviceMap ? std::make_shared<KvStagingPool>(engine, defaultStagingBytes(),
                                                  deviceMap)
                : nullptr;

  const uint64_t bytes = planLogicalBytes(srcPlan);  // logical KV volume moved.
  bool ok = false;
  uint64_t migNs = 0;

  if (!o.peers.empty()) {
    // Multi-host fan-out: one control channel per decode host, opened by the
    // connector (production resolution is injected the same way).
    std::unordered_map<std::string, KvControlChannelConnector::Endpoint> eps;
    for (const Peer& p : o.peers) eps[p.host] = {p.ip, p.port};
    const int timeout = o.timeout_sec;
    KvControlChannelConnector connector(
        eps,
        [timeout](const KvControlChannelConnector::Endpoint& e)
            -> std::shared_ptr<tt::sockets::ISocketTransport> {
          auto t = std::make_shared<TcpControl>(timeout);
          if (!t->initializeAsClient(e.host, e.port)) return nullptr;
          return t;
        });
    if (!connector.openChannels()) {
      std::cerr << "[sender] warning: not all decode peers got a transport\n";
    }
    KvMigrationMultiHostSender multiHost(engine, dev, prefill, decode,
                                         prefillHost, connector.channels(),
                                         deviceMap);
    std::cout << "[sender] fan-out to " << multiHost.hostCount()
              << " decode host(s), trigger=" << o.trigger << "\n";
    const auto t0 = std::chrono::steady_clock::now();
    if (o.trigger == "kafka") {
#ifdef TT_E2E_WITH_KAFKA
      ok = driveViaKafka(multiHost, req, o);
#else
      std::cerr << "[sender] --trigger kafka needs a KAFKA_ENABLED build "
                   "(./build.sh --kafka)\n";
      return 1;
#endif
    } else {
      ok = multiHost.migrate(o.uuid, req);
    }
    migNs = nsSince(t0);
  } else {
    // Single decode host over one control channel.
    auto control = std::make_shared<TcpControl>(o.timeout_sec);
    std::cout << "[sender] connecting control -> " << o.control_host << ":"
              << o.control_port << "\n";
    if (!control->initializeAsClient(o.control_host, o.control_port)) {
      std::cerr << "[sender] control connect failed\n";
      return 1;
    }
    KvControlChannel channel(control);
    const auto t0 = std::chrono::steady_clock::now();
    MooncakeKvSender sender(engine, dev, prefill, decode, prefillHost,
                            decodeHost, staging);
    KvMigrationSender orch(channel, sender);
    ok = orch.migrate(o.uuid, req);
    migNs = nsSince(t0);
  }

  // Metal coexistence: release the low-IOVA reservations now that the
  // DRISC transfer is done. In production (unified worker + live engine) this
  // happens before signaling READY so a later model takes pcie_base.
  if (useDrisc && drisc) {
    std::cout << "[sender] releasing " << drisc->numIovaReservations()
              << " IOVA reservation(s)\n";
    drisc->releaseIovaReservations();
  }

  std::cout << "[sender] migrate -> " << (ok ? "OK" : "FAIL") << "\n";
  // Boundary timing + plan-derived volume (no data-plane instrumentation).
  std::printf(
      "[migration] sender: chunks=%zu bytes=%llu | migrate_total=%.3f ms "
      "(%.2f GB/s)\n",
      srcPlan.chunks.size(), (unsigned long long)bytes, ms(migNs),
      gbps(bytes, migNs));
  return ok ? 0 : 1;
}

// --- Receiver --------------------------------------------------------------
int runReceiver(const Options& o) {
  std::shared_ptr<const IKvTable> decode;
  std::string decodeHost = o.decode_host;
  if (o.table == "builtin") {
    decode = builtinDecode();
    decodeHost = "decode";
  } else if (o.table == "builtin2") {
    decode = builtinDecodeSplit();  // this proc's host = --decode-host tag.
  } else {
    auto adapter = KvChunkAddressTableAdapter::fromProtobufFile(o.table);
    if (!adapter) {
      std::cerr << "[receiver] failed to load decode table " << o.table
                << " (need -DENABLE_KV_TABLE=ON)\n";
      return 1;
    }
    decode = std::shared_ptr<const IKvTable>(std::move(adapter));
  }
  const MigrationRequest req = requestOf(o);
  const HostKvPlan plan = buildHostPlan(*decode, decodeHost, req.dstSlice());
  if (plan.empty()) {
    std::cerr << "[receiver] no decode chunks for request on host '"
              << decodeHost << "'\n";
    return 1;
  }

  // Device or host I/O for the decode side. DRISC (NOC-DMA) launches its
  // per-chip service kernel in makeDrisc; host mode uses the store.
  HostDeviceIo hostIo;
  std::unique_ptr<DriscDeviceIo> drisc;
  const bool useDrisc = (o.mode == "device");
  if (useDrisc) {
    drisc = makeDrisc(plan, loadDeviceMap(o.device_map));
  }
  IDeviceIo& dev = useDrisc ? static_cast<IDeviceIo&>(*drisc)
                            : static_cast<IDeviceIo&>(hostIo);
  // DRISC device registrar for the double-pinned bounce buffer (null
  // otherwise).
  DeviceMapFn deviceMap;
  if (useDrisc) {
    DriscDeviceIo* d = drisc.get();
    deviceMap = [d](void* va, std::size_t bytes) {
      d->registerHostRegion(va, bytes);
    };
  }

  // Mooncake engine; its advertised segment name is the sender's WRITE target.
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.local_server_name = o.mooncake_name;
  cfg.metadata_uri = o.metadata;
  cfg.protocol =
      (o.protocol == "rdma") ? TransportProtocol::RDMA : TransportProtocol::TCP;
  if (!engine->init(cfg)) {
    std::cerr << "[receiver] Mooncake engine init failed\n";
    return 1;
  }
  const std::string segmentName = engine->localServerName();
  std::cout << "[receiver] Mooncake segment: " << segmentName << "\n";

  // Control channel (bind + accept the sender). Blocks until the sender dials.
  auto control = std::make_shared<TcpControl>(o.timeout_sec);
  std::cout << "[receiver] control listening on :" << o.control_port << "\n";
  if (!control->initializeAsServer(o.control_port)) {
    std::cerr << "[receiver] control bind/accept failed\n";
    return 1;
  }
  KvControlChannel channel(control);

  const auto serveStart = std::chrono::steady_clock::now();
  {
    // The bounce receiver holds no table; it drains slots straight from the
    // window descriptors. Message count is variable (one WindowReady per
    // window), so serve until the sender finishes and closes (run() returns on
    // the peer-close the DoneMarker/Ack is followed by).
    MooncakeKvReceiver bounceReceiver(engine, dev, segmentName, bounceGeoOf(o),
                                      deviceMap);
    if (!bounceReceiver.registered()) {
      std::cerr << "[receiver] bounce buffer registration failed\n";
      return 1;
    }
    KvMigrationReceiver orch(channel, bounceReceiver);
    std::cout << "[receiver] bounce transport (" << bounceGeoOf(o).section_count
              << " sections x " << bounceGeoOf(o).section_size << " B)\n";
    orch.run();
  }
  const uint64_t serveNs = nsSince(serveStart);

  // Metal coexistence: release the low-IOVA reservations after the
  // drain (see the sender side / production timing note).
  if (useDrisc && drisc) {
    std::cout << "[receiver] releasing " << drisc->numIovaReservations()
              << " IOVA reservation(s)\n";
    drisc->releaseIovaReservations();
  }

  // Boundary timing + plan-derived volume (no data-plane instrumentation).
  const uint64_t bytes = planWireBytes(plan);  // includes replica fan-out.
  std::printf(
      "[migration] receiver[%s]: chunks=%zu bytes=%llu | serve_total=%.3f ms "
      "(%.2f GB/s)\n",
      decodeHost.c_str(), plan.chunks.size(), (unsigned long long)bytes,
      ms(serveNs), gbps(bytes, serveNs));

  // Verify the decode devices hold the agreed pattern (builtin(2) always; real
  // table only under --seed-verify — see the sender's seeding note). Under a
  // position shift the bytes now at a dst position were seeded at the matching
  // *src* position, so map dst -> src by ordinal before computing the expected
  // pattern. Symmetric => srcPos == dst_pos.
  const bool verify =
      o.table == "builtin" || o.table == "builtin2" || o.seed_verify;
  if (!verify) {
    std::cout
        << "[receiver] migration drained (real-table mode; no byte verify)\n";
    return 0;
  }
  bool ok = true;
  const uint32_t step = decode->config().chunk_n_tokens;
  for (const auto& chunk : plan.chunks) {
    for (const auto& t : chunk.targets) {
      const auto expected =
          expectedForDst(req, step, chunk.layer, chunk.position, t.size_bytes);
      std::vector<uint8_t> buf(t.size_bytes);
      if (!dev.read(t.device, t.noc_addr, t.size_bytes, buf.data()) ||
          buf != expected) {
        ok = false;
      }
    }
  }
  std::cout << "[receiver] verification: " << (ok ? "PASS" : "FAIL") << "\n";
  return ok ? 0 : 1;
}

// --- gtest driver (host + builtin) -----------------------------------------
// Reaps a forked child on scope exit so an early ASSERT can't leak a zombie or
// leave the sender hung mid-connect. Disarm (pid = -1) after a clean waitpid.
struct ChildReaper {
  pid_t pid;
  ~ChildReaper() {
    if (pid > 0) {
      ::kill(pid, SIGTERM);
      int status = 0;
      ::waitpid(pid, &status, 0);
    }
  }
};

// fork()+execv() this same binary as the sender worker (`--role sender`), so
// the two Mooncake engines live in separate processes exactly as in production.
pid_t forkSelfAsSender(const Options& o,
                       const std::string& senderMooncakeName) {
  char exePath[PATH_MAX];
  const ssize_t n = ::readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  if (n <= 0) return -1;
  exePath[n] = '\0';

  // Resolve to explicit asymmetric coordinates so the child sees exactly what
  // the parent (receiver) does, whether the test set symmetric or shifted args.
  const MigrationRequest req = requestOf(o);
  const std::string port = std::to_string(o.control_port);
  const std::string ss = std::to_string(req.src_slot);
  const std::string ds = std::to_string(req.dst_slot);
  const std::string lb = std::to_string(req.layer_begin);
  const std::string le = std::to_string(req.layer_end);
  const std::string spb = std::to_string(req.src_position_begin);
  const std::string spe = std::to_string(req.src_position_end);
  const std::string dpb = std::to_string(req.dst_position_begin);
  const std::string dpe = std::to_string(req.dst_position_end);
  const std::string uuid = std::to_string(o.uuid);
  const std::string tmo = std::to_string(o.timeout_sec);

  const pid_t pid = ::fork();
  if (pid == 0) {
    ::prctl(PR_SET_PDEATHSIG, SIGTERM);  // die if the test process dies
    const std::vector<std::string> a = {exePath,
                                        "--role",
                                        "sender",
                                        "--mode",
                                        "host",
                                        "--transport",
                                        o.transport,
                                        "--table",
                                        "builtin",
                                        "--mooncake-name",
                                        senderMooncakeName,
                                        "--control-host",
                                        o.control_host,
                                        "--control-port",
                                        port,
                                        "--prefill-host",
                                        "prefill",
                                        "--decode-host",
                                        "decode",
                                        "--src-slot",
                                        ss,
                                        "--dst-slot",
                                        ds,
                                        "--layer-begin",
                                        lb,
                                        "--layer-end",
                                        le,
                                        "--src-pos-begin",
                                        spb,
                                        "--src-pos-end",
                                        spe,
                                        "--dst-pos-begin",
                                        dpb,
                                        "--dst-pos-end",
                                        dpe,
                                        "--uuid",
                                        uuid,
                                        "--timeout-sec",
                                        tmo};
    std::vector<char*> argv;
    argv.reserve(a.size() + 1);
    for (const auto& s : a) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    ::execv(exePath, argv.data());
    ::_exit(127);  // execv only returns on failure
  }
  return pid;  // <0 on fork failure
}

// In-process receiver for the host+builtin path: builds the decode side, serves
// the two control messages of one migration, and hands the populated device
// store + plan back so the test can verify bytes with gtest macros. Returns
// false (with `err`) on any setup/serve failure. Kept separate from the
// standalone runReceiver() worker so the test can assert on the result here.
bool serveReceiverHostBuiltin(const Options& o, HostDeviceIo& dev,
                              HostKvPlan& planOut, std::string& err) {
  auto decode = builtinDecode();
  const std::string decodeHost = "decode";
  const MigrationRequest req = requestOf(o);
  planOut = buildHostPlan(*decode, decodeHost, req.dstSlice());
  if (planOut.empty()) {
    err = "no decode chunks for request";
    return false;
  }

  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.local_server_name = o.mooncake_name;
  cfg.protocol = TransportProtocol::TCP;
  if (!engine->init(cfg)) {
    err = "Mooncake engine init failed";
    return false;
  }
  const std::string segmentName = engine->localServerName();

  auto control = std::make_shared<TcpControl>(o.timeout_sec);
  if (!control->initializeAsServer(o.control_port)) {
    err = "control bind/accept failed";
    return false;
  }
  KvControlChannel channel(control);

  MooncakeKvReceiver bounceReceiver(engine, dev, segmentName, bounceGeoOf(o));
  if (!bounceReceiver.registered()) {
    err = "bounce buffer registration failed";
    return false;
  }
  KvMigrationReceiver orch(channel, bounceReceiver);
  orch.run();  // serves Begin/Window*/Done until the sender closes post-Ack
  return true;
}

}  // namespace

// Full real-stack round trip with no hardware: two processes (this test = the
// decode receiver, a forked child = the prefill sender) over a real TCP control
// channel + real Mooncake TCP transfer, verifying the decode devices hold the
// agreed per-(layer,pos) pattern. `o` carries the request (symmetric or a
// position shift); ports are passed so cases don't collide on rerun.
void runHostBuiltinMigration(Options o, uint16_t controlPort,
                             const std::string& recvName,
                             const std::string& senderName) {
  o.role = "receiver";
  o.mode = "host";
  o.table = "builtin";
  o.control_host = "127.0.0.1";
  o.control_port = controlPort;
  o.mooncake_name = recvName;  // receiver (this process)

  const pid_t sender = forkSelfAsSender(o, senderName);
  ASSERT_GT(sender, 0) << "fork/execv of sender worker failed";
  ChildReaper reaper{sender};  // disarmed after the clean waitpid below

  HostDeviceIo dev;
  HostKvPlan plan;
  std::string err;
  ASSERT_TRUE(serveReceiverHostBuiltin(o, dev, plan, err)) << err;

  std::vector<uint8_t> buf(K_CHUNK);
  const uint32_t step = builtinConfig().chunk_n_tokens;
  const MigrationRequest req = requestOf(o);
  ASSERT_FALSE(plan.chunks.empty()) << "decode plan unexpectedly empty";
  for (const auto& chunk : plan.chunks) {
    const auto expected =
        expectedForDst(req, step, chunk.layer, chunk.position, K_CHUNK);
    for (const auto& t : chunk.targets) {
      ASSERT_TRUE(dev.read(t.device, t.noc_addr, K_CHUNK, buf.data()))
          << "decode read failed at layer=" << chunk.layer
          << " pos=" << chunk.position;
      EXPECT_EQ(buf, expected) << "byte mismatch at layer=" << chunk.layer
                               << " pos=" << chunk.position;
    }
  }

  int status = 0;
  ASSERT_EQ(sender, ::waitpid(sender, &status, 0));
  reaper.pid = -1;  // reaped cleanly; disarm the guard
  EXPECT_TRUE(WIFEXITED(status)) << "sender did not exit normally";
  EXPECT_EQ(0, WEXITSTATUS(status)) << "sender process reported failure";
}

// RDMA-over-host bounce buffer, whole slot, over a real Mooncake TCP transfer.
// A 1-slot / 64 B bounce buffer forces the multi-window credit handshake (8
// windows) over the wire, not just one batch — the streaming path end to end,
// no hardware.
TEST(TransportKvMigrationE2E, HostBuiltinRoundTripBounce) {
  Options o;
  o.transport = "bounce";
  o.bounce_section_count = 1;
  o.bounce_section_size = K_CHUNK;  // one chunk per slot -> many windows
  runHostBuiltinMigration(o, 18654, "127.0.0.1:17785", "127.0.0.1:17786");
}

// Bounce path with a position shift and a bounce buffer large enough to merge
// each layer's contiguous chunks into a single slot (one window) — guards
// asymmetric addressing + merged drains over the real transport.
TEST(TransportKvMigrationE2E, HostBuiltinPositionShiftBounce) {
  Options o;
  o.transport = "bounce";
  o.src_pos_begin = 0;
  o.src_pos_end = 64;
  o.dst_pos_begin = 64;
  o.dst_pos_end = 128;
  runHostBuiltinMigration(o, 18656, "127.0.0.1:17787", "127.0.0.1:17788");
}

int main(int argc, char** argv) {
  // Worker path: when a --role is given, run that single side and return its
  // exit code. This is how the gtest driver's forked child (sender) and the
  // device / real-table shell harness (sender + receiver) invoke the binary.
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--role") == 0) {
      Options o;
      if (!parseArgs(argc, argv, o)) {
        usage();
        return 2;
      }
      return o.role == "sender" ? runSender(o) : runReceiver(o);
    }
  }

  // No role: run the gtest suite (host + builtin round trip).
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
