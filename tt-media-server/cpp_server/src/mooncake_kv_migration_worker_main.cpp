// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// mooncake_kv_migration_worker: the single migration-worker binary that joins
// the #4406 Kafka trigger to the #4173 data plane. One process, two roles:
//
//   --role prefill : consumes MigrationRequestMessages from Kafka and drives a
//                    real migration across the decode hosts via
//                    MooncakeMigrationExecutor → KvMigrationMultiHostSender,
//                    then publishes the ack. (KvMigrationWorker owns the loop.)
//
//   --role decode  : registers its KV mirror as a Mooncake segment and runs
//                    KvMigrationReceiverServer, answering the migration control
//                    protocol (prepareMirror / drain) for inbound migrations.
//
// This supersedes bringup_mooncake_worker (Kafka loop that only logged) and
// tt_kv_migration_consumer (StubMigrationExecutor): it is the first binary that
// actually moves KV on a Kafka trigger.
//
// Table source: loadKvTableFile for now (the .pb path); the engine→worker
// handoff (engine_table_handoff) swaps in behind the same IKvTable once the
// engine implements the producer. Device IO: MultiDeviceUmd; FabricNode→ASIC
// chip resolution comes from an optional --device-map file (the same contract
// the engine will hand over), falling back to the placeholder (device &
// 0xFFFF) for a single-mesh host when no map is given.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/defaults.hpp"
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "runtime/worker/kv_migration_worker.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map.hpp"  // DeviceMap (FabricNode -> UMD chip)
#include "transport/host_dram_storage_backend.hpp"
#include "transport/kv_migration_endpoints.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/kv_table_adapter.hpp"        // allHostLocations
#include "transport/kv_table_provisioning.hpp"   // loadKvTableFile
#include "transport/kv_table_view.hpp"           // IKvTable
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_migration_executor.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/multi_device_umd.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"
#include "utils/logger.hpp"

namespace {

using namespace tt::transport;

constexpr int K_IDLE_POLL_MS = 100;

std::atomic<bool> gStop{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "gStop must be lock-free to be signal-safe");
void onSignal(int /*sig*/) { gStop.store(true, std::memory_order_relaxed); }

std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string{v} : std::string{fallback};
}

enum class Role { Prefill, Decode };

struct WorkerConfig {
  Role role = Role::Prefill;
  std::string metadata_uri;  // Mooncake metadata service (or P2PHANDSHAKE).
  std::string name;          // this worker's Mooncake server/segment name.
  std::string host;          // this node's host tag in the table.
  std::string device_map_path;  // FabricNode->UMD chip map (both roles); empty
                                // => placeholder (device & 0xFFFF).

  // prefill:
  std::string prefill_table_path;
  std::string decode_table_path;
  std::unordered_map<std::string, KvControlChannelConnector::Endpoint> peers;

  // decode:
  std::string table_path;  // this decode node's table.
  std::string segment;     // advertised segment (defaults to `name`).
  uint16_t control_port = 0;
};

void usage() {
  std::cerr
      << "usage: mooncake_kv_migration_worker --role prefill|decode "
         "--metadata URI --name NAME --host HOST_TAG\n"
         "  prefill: --prefill-table P.pb --decode-table D.pb "
         "--peer-control NAME=host:port (repeatable)\n"
         "  decode:  --table D.pb --control-port N [--segment NAME]\n"
         "  both:    [--device-map FILE]  ('mesh chip umd' per line; needed "
         "when this host's table spans multiple meshes)\n"
         "  Kafka (prefill) via env: KAFKA_BROKERS, "
         "KAFKA_MIGRATION_REQUEST_TOPIC, KAFKA_MIGRATION_ACK_TOPIC, "
         "KAFKA_GROUP_ID\n";
}

// Parse "NAME=host:port" into (logical name, endpoint).
bool parsePeer(const std::string& spec, std::string& name,
               KvControlChannelConnector::Endpoint& ep) {
  const auto eq = spec.find('=');
  const auto colon = spec.rfind(':');
  if (eq == std::string::npos || colon == std::string::npos || colon < eq) {
    return false;
  }
  name = spec.substr(0, eq);
  ep.host = spec.substr(eq + 1, colon - eq - 1);
  try {
    ep.port = static_cast<uint16_t>(std::stoi(spec.substr(colon + 1)));
  } catch (...) {
    return false;
  }
  return !name.empty() && !ep.host.empty() && ep.port != 0;
}

bool parseConfig(int argc, char** argv, WorkerConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--role" && next(v)) {
      if (v == "prefill") {
        cfg.role = Role::Prefill;
      } else if (v == "decode") {
        cfg.role = Role::Decode;
      } else {
        std::cerr << "--role must be prefill|decode\n";
        return false;
      }
      continue;
    }
    if (a == "--metadata" && next(cfg.metadata_uri)) continue;
    if (a == "--name" && next(cfg.name)) continue;
    if (a == "--host" && next(cfg.host)) continue;
    if (a == "--prefill-table" && next(cfg.prefill_table_path)) continue;
    if (a == "--decode-table" && next(cfg.decode_table_path)) continue;
    if (a == "--table" && next(cfg.table_path)) continue;
    if (a == "--device-map" && next(cfg.device_map_path)) continue;
    if (a == "--segment" && next(cfg.segment)) continue;
    if (a == "--control-port" && next(v)) {
      try {
        cfg.control_port = static_cast<uint16_t>(std::stoi(v));
      } catch (...) {
        std::cerr << "--control-port invalid: " << v << "\n";
        return false;
      }
      continue;
    }
    if (a == "--peer-control" && next(v)) {
      std::string name;
      KvControlChannelConnector::Endpoint ep;
      if (!parsePeer(v, name, ep)) {
        std::cerr << "--peer-control must be NAME=host:port, got: " << v << "\n";
        return false;
      }
      cfg.peers[name] = ep;
      continue;
    }
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }

  if (cfg.metadata_uri.empty() || cfg.name.empty() || cfg.host.empty()) {
    std::cerr << "--metadata, --name and --host are required\n";
    return false;
  }
  if (cfg.role == Role::Prefill &&
      (cfg.prefill_table_path.empty() || cfg.decode_table_path.empty() ||
       cfg.peers.empty())) {
    std::cerr << "prefill needs --prefill-table, --decode-table and at least "
                 "one --peer-control\n";
    return false;
  }
  if (cfg.role == Role::Decode &&
      (cfg.table_path.empty() || cfg.control_port == 0)) {
    std::cerr << "decode needs --table and --control-port\n";
    return false;
  }
  // NB: segment defaults to the engine's real local server name at runtime
  // (runDecode), not cfg.name — under P2PHANDSHAKE the RPC port is auto-assigned
  // so engine->localServerName() != cfg.name, and the sender must open the
  // engine's actual segment. Only an explicit --segment overrides it.
  return true;
}

std::shared_ptr<MooncakeTransferEngine> makeEngine(const WorkerConfig& cfg) {
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig ec;
  ec.metadata_uri = cfg.metadata_uri;
  ec.local_server_name = cfg.name;
  ec.protocol = TransportProtocol::TCP;
  if (!engine->init(ec)) {
    TT_LOG_ERROR("[worker] engine init failed (metadata={})", cfg.metadata_uri);
    return nullptr;
  }
  return engine;
}

// Load a 'mesh chip umd' device map (the same format the e2e harness uses).
// Empty path => empty map => buildDeviceIo falls back to the placeholder.
DeviceMap loadDeviceMapFile(const std::string& path) {
  DeviceMap dm;
  if (path.empty()) return dm;
  std::ifstream f(path);
  if (!f.good()) {
    TT_LOG_WARN("[worker] cannot open --device-map {}; using placeholder chip ids",
                path);
    return dm;
  }
  uint32_t mesh = 0, chip = 0;
  uint64_t umd = 0;
  while (f >> mesh >> chip >> umd) dm.set(FabricNode{mesh, chip}, umd);
  TT_LOG_INFO("[worker] device-map: {} entries from {}", dm.size(), path);
  return dm;
}

// One UmdDeviceAccess per device this host owns in `table`. Chip resolution uses
// `device_map` (the Phase-4b seam the engine fills; the file-based map is the
// same contract) and falls back to the encodeDevice low bits (placeholder, as
// in the e2e harness) for any device the map doesn't cover. The placeholder is
// only correct when the host's table is single-mesh; a multi-mesh host (its KV
// aliases chip ids across meshes) needs the map or its replicas collide.
std::unique_ptr<MultiDeviceUmd> buildDeviceIo(const IKvTable& table,
                                              const std::string& host,
                                              const DeviceMap& device_map) {
  auto umd = std::make_unique<MultiDeviceUmd>();
  for (const auto& loc : allHostLocations(table, host)) {
    if (!umd->hasDevice(loc.device)) {
      const auto mapped = device_map.umdChip(loc.device);
      const int chip = mapped ? static_cast<int>(*mapped)
                              : static_cast<int>(loc.device & 0xFFFFu);
      umd->addDevice(loc.device, std::make_shared<UmdDeviceAccess>(chip));
    }
  }
  return umd;
}

std::shared_ptr<tt::sockets::ISocketTransport> makeClientTransport(
    const KvControlChannelConnector::Endpoint& ep) {
  auto t = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!t->initializeAsClient(ep.host, ep.port)) return nullptr;
  t->start();
  return t;
}

std::shared_ptr<tt::sockets::ISocketTransport> makeServerTransport(uint16_t port) {
  auto t = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!t->initializeAsServer(port)) return nullptr;
  t->start();
  return t;
}

int runPrefill(const WorkerConfig& cfg) {
  auto engine = makeEngine(cfg);
  if (!engine) return 1;

  auto prefill = loadKvTableFile(cfg.prefill_table_path);
  auto decode = loadKvTableFile(cfg.decode_table_path);
  if (!prefill || !decode) {
    TT_LOG_ERROR("[worker] failed to load prefill/decode table");
    return 1;
  }

  const DeviceMap device_map = loadDeviceMapFile(cfg.device_map_path);
  auto device = buildDeviceIo(*prefill->table, cfg.host, device_map);

  // Open one control channel per decode host (static resolution for now; the
  // map is the seam a discovery service fills later).
  KvControlChannelConnector connector(cfg.peers, makeClientTransport);
  if (!connector.connect()) {
    TT_LOG_WARN(
        "[worker] not every decode peer connected; involved-but-missing hosts "
        "will fail their slice");
  }

  KvMigrationMultiHostSender sender(engine, *device, prefill->table,
                                    decode->table, cfg.host,
                                    connector.channels());
  auto executor = std::make_unique<MooncakeMigrationExecutor>(sender);

  const std::string brokers = envOr("KAFKA_BROKERS", tt::config::defaults::KAFKA_BROKERS);
  const std::string reqTopic = envOr("KAFKA_MIGRATION_REQUEST_TOPIC",
                                     tt::config::defaults::KAFKA_MIGRATION_REQUEST_TOPIC);
  const std::string ackTopic = envOr("KAFKA_MIGRATION_ACK_TOPIC",
                                     tt::config::defaults::KAFKA_MIGRATION_ACK_TOPIC);
  const std::string group = envOr("KAFKA_GROUP_ID", tt::config::defaults::KAFKA_GROUP_ID);

  auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = brokers, .topic = reqTopic, .group_id = group});
  auto producer = std::make_unique<tt::messaging::KafkaProducer>(
      tt::messaging::KafkaProducerConfig{.brokers = brokers, .topic = ackTopic});

  tt::worker::KvMigrationWorker worker(std::move(consumer), std::move(producer),
                                       std::move(executor));
  TT_LOG_INFO(
      "[worker] prefill '{}' READY: {} decode peers, brokers={} req={} ack={}",
      cfg.name, connector.connectedCount(), brokers, reqTopic, ackTopic);
  worker.start();
  while (!gStop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_IDLE_POLL_MS));
  }
  worker.stop();
  TT_LOG_INFO("[worker] prefill '{}' stopping", cfg.name);
  return 0;
}

int runDecode(const WorkerConfig& cfg) {
  auto engine = makeEngine(cfg);
  if (!engine) return 1;

  auto decode = loadKvTableFile(cfg.table_path);
  if (!decode) {
    TT_LOG_ERROR("[worker] failed to load decode table {}", cfg.table_path);
    return 1;
  }

  const DeviceMap device_map = loadDeviceMapFile(cfg.device_map_path);
  auto device = buildDeviceIo(*decode->table, cfg.host, device_map);

  // The sender opens the engine's actual segment (its live local server name);
  // under P2PHANDSHAKE that's an auto-assigned endpoint, not cfg.name. An
  // explicit --segment overrides (e.g. a metadata-server deployment).
  const std::string segment =
      cfg.segment.empty() ? engine->localServerName() : cfg.segment;
  // The mirror is registered as the Mooncake segment inside MooncakeKvReceiver.
  MooncakeKvReceiver receiver(engine, *device, decode->table, cfg.host,
                              segment);
  if (!receiver.registered()) {
    TT_LOG_ERROR("[worker] decode '{}' failed to register mirror segment '{}'",
                 cfg.name, segment);
    return 1;
  }

  KvMigrationReceiverServer server(cfg.control_port, makeServerTransport,
                                   receiver);
  if (!server.start()) {
    TT_LOG_ERROR("[worker] decode '{}' failed to start control server on :{}",
                 cfg.name, cfg.control_port);
    return 1;
  }
  TT_LOG_INFO("[worker] decode '{}' READY: segment={} control_port={}", cfg.name,
              segment, cfg.control_port);
  while (!gStop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_IDLE_POLL_MS));
  }
  server.stop();
  TT_LOG_INFO("[worker] decode '{}' stopping", cfg.name);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("kv-migration-worker");

  WorkerConfig cfg;
  if (!parseConfig(argc, argv, cfg)) {
    usage();
    return 2;
  }
  std::signal(SIGTERM, onSignal);
  std::signal(SIGINT, onSignal);

  return cfg.role == Role::Prefill ? runPrefill(cfg) : runDecode(cfg);
}
