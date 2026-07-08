// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Acceptance harness for the real KV-cache migration path.
//
// Two processes (a prefill "sender" and a decode "receiver") drive a full
// table-addressed migration over a real TCP control channel + real Mooncake:
//
//   prefill device DRAM --read(UMD)--> staging
//       --Mooncake one-sided WRITE(mirror_offset)--> decode mirror segment
//       --drain(UMD)--> decode device DRAM
//
//   control channel (TCP):  BeginMigration -> MirrorReady -> [WRITEs]
//                           -> DoneMarker -> Ack
//
// Unlike transport_migration_e2e (the dummy single-tensor PoC), this exercises
// the real address layer: KvCacheMirror / RemoteRegion built from a KV table,
// device-group fan-out, per-(device,channel) physical offsets, and the
// KvMigration{Sender,Receiver} orchestrators.
//
// Modes:
//   --mode host    : a host-backed device store stands in for device DRAM.
//                    Runs on any --mooncake build, no hardware. The byte
//                    content is a deterministic pattern keyed by (layer,pos),
//                    so the receiver verifies without an out-of-band payload.
//   --mode device  : real TT device DRAM via UMD (needs --mooncake +
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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/i_device_io.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_chunk_address_table_adapter.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_migration_orchestrator.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_kv_sender.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/multi_device_umd.hpp"
#include "transport/transfer_types.hpp"

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
  bool isConnected() const override { return conn >= 0; }
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
      if (r <= 0) return false;  // 0 = peer closed, <0 = error/timeout (EAGAIN)
      off += static_cast<std::size_t>(r);
    }
    return true;
  }

  int timeout_sec = 0;
  int listen = -1;
  int conn = -1;
};

// --- Host-backed device store (host mode) ----------------------------------
class HostDeviceIo : public IDeviceIo {
 public:
  bool read(LocalDeviceId d, NocAddr n, std::size_t size, void* host) override {
    const auto it = store.find({d, n});
    if (it == store.end() || it->second.size() < size) return false;
    std::memcpy(host, it->second.data(), size);
    return true;
  }
  bool write(LocalDeviceId d, NocAddr n, const void* host,
             std::size_t size) override {
    auto& v = store[{d, n}];
    const auto* p = static_cast<const uint8_t*>(host);
    v.assign(p, p + size);
    return true;
  }

 private:
  std::map<std::pair<LocalDeviceId, NocAddr>, std::vector<uint8_t>> store;
};

// --- Options ---------------------------------------------------------------
struct Options {
  std::string role;               // sender | receiver
  std::string mode = "host";      // host | device
  std::string table = "builtin";  // builtin | <protobuf path> (local table)
  std::string decode_table;       // sender, real-table mode: decode .pb path
  std::string control_host = "127.0.0.1";
  uint16_t control_port = 18650;
  std::string mooncake_name;  // this proc's Mooncake host:port
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
};

void usage() {
  std::cerr
      << "usage: transport_kv_migration_e2e --role sender|receiver "
         "--mooncake-name HOST:PORT\n"
         "  --control-host H (sender; default 127.0.0.1)  --control-port P "
         "(default 18650)\n"
         "  --mode host|device   --table builtin|<protobuf-path>\n"
         "  --prefill-host NAME  --decode-host NAME\n"
         "  --slot N  --layer-begin N --layer-end N  --pos-begin N --pos-end "
         "N\n"
         "  (asymmetric overrides) --src-slot N --dst-slot N "
         "--src-pos-begin N --src-pos-end N --dst-pos-begin N --dst-pos-end N\n"
         "  --uuid N  --timeout-sec S\n";
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
    if (a == "--table" && nextStr(o.table)) continue;
    if (a == "--decode-table" && nextStr(o.decode_table)) continue;
    if (a == "--control-host" && nextStr(o.control_host)) continue;
    if (a == "--mooncake-name" && nextStr(o.mooncake_name)) continue;
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
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if ((o.role != "sender" && o.role != "receiver") || o.mooncake_name.empty()) {
    return false;
  }
  return true;
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

std::vector<uint8_t> pattern(uint32_t layer, uint32_t pos) {
  std::vector<uint8_t> v(K_CHUNK);
  for (uint32_t i = 0; i < K_CHUNK; ++i) {
    v[i] = static_cast<uint8_t>(layer * 40 + pos + i);
  }
  return v;
}

// The bytes drained to a dst (layer, position) were seeded at the matching
// *src* position — chunks pair by ordinal within their ranges. Map dst -> src
// so verification matches under a position shift (symmetric => srcPos==dstPos).
std::vector<uint8_t> expectedForDst(const MigrationRequest& req, uint32_t step,
                                    uint32_t layer, uint32_t dstPos) {
  const uint32_t srcStart =
      req.src_position_begin - (req.src_position_begin % step);
  const uint32_t dstStart =
      req.dst_position_begin - (req.dst_position_begin % step);
  return pattern(layer, srcStart + (dstPos - dstStart));
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

// Build a real device-I/O over the devices touched by `plan` (chip id = the
// FabricNode chip id encoded in the LocalDeviceId low 16 bits).
std::unique_ptr<MultiDeviceUmd> makeUmd(const HostKvPlan& plan) {
  auto umd = std::make_unique<MultiDeviceUmd>();
  for (const auto& chunk : plan.chunks) {
    for (const auto& t : chunk.targets) {
      if (!umd->hasDevice(t.device)) {
        const int chip = static_cast<int>(t.device & 0xFFFFu);
        umd->addDevice(t.device, std::make_shared<UmdDeviceAccess>(chip));
      }
    }
  }
  return umd;
}

// --- Sender ----------------------------------------------------------------
int runSender(const Options& o) {
  // Local (prefill) table + the request.
  std::shared_ptr<const IKvTable> prefill;
  std::shared_ptr<const IKvTable> decode;
  std::string prefillHost = o.prefill_host;
  std::string decodeHost = o.decode_host;

  if (o.table == "builtin") {
    prefill = builtinPrefill();
    decode = builtinDecode();
    prefillHost = "prefill";
    decodeHost = "decode";
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

  // Control channel (connect to the receiver).
  auto control = std::make_shared<TcpControl>(o.timeout_sec);
  std::cout << "[sender] connecting control -> " << o.control_host << ":"
            << o.control_port << "\n";
  if (!control->initializeAsClient(o.control_host, o.control_port)) {
    std::cerr << "[sender] control connect failed\n";
    return 1;
  }
  KvControlChannel channel(control);

  // Device or host I/O for the prefill side.
  HostDeviceIo hostIo;
  std::unique_ptr<MultiDeviceUmd> umd;
  const HostKvPlan srcPlan =
      buildHostPlan(*prefill, prefillHost, req.srcSlice());
  if (srcPlan.empty()) {
    std::cerr << "[sender] no prefill chunks for request on host '"
              << prefillHost << "'\n";
    return 1;
  }
  if (o.mode == "device") umd = makeUmd(srcPlan);
  IDeviceIo& dev = (o.mode == "device") ? static_cast<IDeviceIo&>(*umd)
                                        : static_cast<IDeviceIo&>(hostIo);

  // Built-in mode: seed the source with the deterministic pattern so the
  // receiver can verify. (Real-table mode reads whatever is already on device.)
  if (o.table == "builtin") {
    for (const auto& chunk : srcPlan.chunks) {
      const auto bytes = pattern(chunk.layer, chunk.position);
      for (const auto& t : chunk.targets) {
        dev.write(t.device, t.noc_addr, bytes.data(), bytes.size());
      }
    }
  }

  // Mooncake engine (storage backend unused by the migrator; pass a host one).
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.local_server_name = o.mooncake_name;
  cfg.protocol = TransportProtocol::TCP;
  if (!engine->init(cfg)) {
    std::cerr << "[sender] Mooncake engine init failed\n";
    return 1;
  }

  MooncakeKvSender sender(engine, dev, prefill, decode, prefillHost,
                          decodeHost);
  KvMigrationSender orch(channel, sender);
  const bool ok = orch.migrate(o.uuid, req);
  std::cout << "[sender] migrate -> " << (ok ? "OK" : "FAIL") << "\n";
  return ok ? 0 : 1;
}

// --- Receiver --------------------------------------------------------------
int runReceiver(const Options& o) {
  std::shared_ptr<const IKvTable> decode;
  std::string decodeHost = o.decode_host;
  if (o.table == "builtin") {
    decode = builtinDecode();
    decodeHost = "decode";
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

  // Device or host I/O for the decode side.
  HostDeviceIo hostIo;
  std::unique_ptr<MultiDeviceUmd> umd;
  if (o.mode == "device") umd = makeUmd(plan);
  IDeviceIo& dev = (o.mode == "device") ? static_cast<IDeviceIo&>(*umd)
                                        : static_cast<IDeviceIo&>(hostIo);

  // Mooncake engine; its advertised segment name is the sender's WRITE target.
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig cfg;
  cfg.local_server_name = o.mooncake_name;
  cfg.protocol = TransportProtocol::TCP;
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

  MooncakeKvReceiver receiver(engine, dev, decode, decodeHost, segmentName);
  KvMigrationReceiver orch(channel, receiver);

  // Serve exactly the two control messages of one migration.
  if (!orch.serveOne()) {  // BeginMigration -> prepareMirror + MirrorReady
    std::cerr << "[receiver] control closed before BeginMigration\n";
    return 1;
  }
  if (!orch.serveOne()) {  // DoneMarker -> drain + Ack
    std::cerr << "[receiver] control closed before DoneMarker\n";
    return 1;
  }

  // Built-in mode: verify the decode devices hold the agreed pattern. Under a
  // position shift the bytes now at a dst position were seeded at the matching
  // *src* position, so map dst -> src by ordinal before computing the expected
  // pattern (srcStart + (dst_pos - dstStart)). Symmetric => srcPos == dst_pos.
  if (o.table == "builtin") {
    bool ok = true;
    std::vector<uint8_t> buf(K_CHUNK);
    const uint32_t step = decode->config().chunk_n_tokens;
    for (const auto& chunk : plan.chunks) {
      const auto expected =
          expectedForDst(req, step, chunk.layer, chunk.position);
      for (const auto& t : chunk.targets) {
        if (!dev.read(t.device, t.noc_addr, K_CHUNK, buf.data()) ||
            buf != expected) {
          ok = false;
        }
      }
    }
    std::cout << "[receiver] verification: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
  }
  std::cout
      << "[receiver] migration drained (real-table mode; no byte verify)\n";
  return 0;
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

  MooncakeKvReceiver receiver(engine, dev, decode, decodeHost, segmentName);
  KvMigrationReceiver orch(channel, receiver);
  if (!orch.serveOne()) {  // BeginMigration -> prepareMirror + MirrorReady
    err = "control closed before BeginMigration";
    return false;
  }
  if (!orch.serveOne()) {  // DoneMarker -> drain + Ack
    err = "control closed before DoneMarker";
    return false;
  }
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
        expectedForDst(req, step, chunk.layer, chunk.position);
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

// Symmetric whole-range migration (src coords == dst coords).
TEST(TransportKvMigrationE2E, HostBuiltinRoundTrip) {
  Options o;  // defaults: slot 5, layers 0..2, pos 0..128 on both sides
  runHostBuiltinMigration(o, 18650, "127.0.0.1:17777", "127.0.0.1:17778");
}

// Position shift: read src positions [0,64) and land them at dst positions
// [64,128) (same slot). Distinct source/destination device addresses, paired by
// ordinal — the bytes seeded at src 0/32 must verify at dst 64/96.
TEST(TransportKvMigrationE2E, HostBuiltinPositionShift) {
  Options o;
  o.src_pos_begin = 0;
  o.src_pos_end = 64;  // src chunks at positions 0, 32
  o.dst_pos_begin = 64;
  o.dst_pos_end = 128;  // dst chunks at positions 64, 96
  runHostBuiltinMigration(o, 18652, "127.0.0.1:17781", "127.0.0.1:17782");
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
