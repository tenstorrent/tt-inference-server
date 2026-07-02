// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// kv_seed_verify: standalone byte-level seed/verify for the production
// mooncake_kv_migration_worker. The worker moves real KV and does NOT seed a
// pattern or read it back (that was e2e-harness-only scaffolding). To get the
// same byte guarantee for a Kafka-triggered worker migration, bracket it:
//
//   1) --mode seed   on the PREFILL host, BEFORE triggering: write the
//      deterministic pattern into the source DRAM at the prefill table's real
//      addresses for the request.
//   2) trigger the migration (scripts/migration_cli.py produce -> the worker).
//   3) --mode verify on the DECODE host, AFTER the ack: read the dest DRAM at
//      the decode table's addresses and compare to the same pattern.
//
// Reuses the exact addressing (loadKvTableFile + buildHostPlan), the device
// map, and the pattern the e2e harness (transport_kv_migration_e2e) uses, so a
// VERIFY: PASS here means the same byte-for-byte guarantee. The device I/O
// (UmdDeviceAccess) needs a device build (--blaze); the .pb loading needs a
// --kv-table build.
//
// Symmetric request only (src slot/positions == dst), which is the whole-slot
// migration the worker performs; the pattern is a pure function of (layer,
// position) so seed and verify agree without a dst->src ordinal remap.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "transport/device_map.hpp"
#include "transport/i_device_io.hpp"
#include "transport/kv_table_adapter.hpp"       // buildHostPlan, KvSlice, HostKvPlan
#include "transport/kv_table_provisioning.hpp"  // loadKvTableFile
#include "transport/kv_table_view.hpp"          // IKvTable, FabricNode
#include "transport/multi_device_umd.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"

namespace {
using namespace tt::transport;

// Same pattern as transport_kv_migration_e2e's patternN: a chunk's byte i is
// (layer*40 + position + i) mod 256 — unique per (layer, position, offset).
std::vector<uint8_t> patternN(uint32_t layer, uint32_t pos, uint64_t size) {
  std::vector<uint8_t> v(size);
  for (uint64_t i = 0; i < size; ++i) {
    v[i] = static_cast<uint8_t>(layer * 40 + pos + i);
  }
  return v;
}

// 'mesh chip umd' per line -> DeviceMap (same format as the e2e --device-map).
DeviceMap loadDeviceMap(const std::string& path) {
  DeviceMap dm;
  if (path.empty()) return dm;
  std::ifstream f(path);
  if (!f.good()) {
    std::cerr << "[seed-verify] cannot open device-map " << path
              << "; using placeholder chip ids\n";
    return dm;
  }
  uint32_t mesh = 0, chip = 0;
  uint64_t umd = 0;
  while (f >> mesh >> chip >> umd) dm.set(FabricNode{mesh, chip}, umd);
  std::cerr << "[seed-verify] device-map: " << dm.size() << " entries\n";
  return dm;
}

// One UmdDeviceAccess per device the plan touches; chip via the map, else the
// placeholder (device & 0xFFFF). Must match the worker's resolution so we
// read/write the same physical chips.
std::unique_ptr<MultiDeviceUmd> makeUmd(const HostKvPlan& plan,
                                        const DeviceMap& dm) {
  auto umd = std::make_unique<MultiDeviceUmd>();
  for (const auto& chunk : plan.chunks) {
    for (const auto& t : chunk.targets) {
      if (umd->hasDevice(t.device)) continue;
      const auto mapped = dm.umdChip(t.device);
      const int chip = mapped ? static_cast<int>(*mapped)
                              : static_cast<int>(t.device & 0xFFFFu);
      umd->addDevice(t.device, std::make_shared<UmdDeviceAccess>(chip));
    }
  }
  return umd;
}

struct Args {
  std::string mode;         // seed | verify
  std::string table;        // this side's .pb (prefill for seed, decode for verify)
  std::string host;         // fabric_node_host tag
  std::string device_map;   // optional
  uint32_t slot = 0;
  uint32_t layer_begin = 0, layer_end = 0;
  uint32_t pos_begin = 0, pos_end = 0;
};

bool parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](std::string& d) {
      if (i + 1 < argc) { d = argv[++i]; return true; }
      return false;
    };
    auto nextU = [&](uint32_t& d) {
      if (i + 1 < argc) {
        d = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
        return true;
      }
      return false;
    };
    if (s == "--mode" && next(a.mode)) continue;
    if (s == "--table" && next(a.table)) continue;
    if (s == "--host" && next(a.host)) continue;
    if (s == "--device-map" && next(a.device_map)) continue;
    if (s == "--slot" && nextU(a.slot)) continue;
    if (s == "--layer-begin" && nextU(a.layer_begin)) continue;
    if (s == "--layer-end" && nextU(a.layer_end)) continue;
    if (s == "--pos-begin" && nextU(a.pos_begin)) continue;
    if (s == "--pos-end" && nextU(a.pos_end)) continue;
    std::cerr << "unknown/incomplete arg: " << s << "\n";
    return false;
  }
  if ((a.mode != "seed" && a.mode != "verify") || a.table.empty() ||
      a.host.empty()) {
    std::cerr
        << "usage: kv_seed_verify --mode seed|verify --table T.pb --host TAG "
           "[--device-map F]\n"
           "       --slot N --layer-begin N --layer-end N --pos-begin N "
           "--pos-end N\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse(argc, argv, a)) return 2;

  auto loaded = loadKvTableFile(a.table);
  if (!loaded || !loaded->table) {
    std::cerr << "[seed-verify] failed to load table " << a.table
              << " (needs an ENABLE_KV_TABLE build)\n";
    return 1;
  }
  const IKvTable& table = *loaded->table;

  const KvSlice slice{a.slot, a.layer_begin, a.layer_end, a.pos_begin,
                      a.pos_end};
  const HostKvPlan plan = buildHostPlan(table, a.host, slice);
  if (plan.empty()) {
    std::cerr << "[seed-verify] no chunks for host '" << a.host
              << "' in this request (check host tag / layers / positions)\n";
    return 1;
  }

  const DeviceMap dm = loadDeviceMap(a.device_map);
  auto umd = makeUmd(plan, dm);
  IDeviceIo& dev = *umd;

  const bool seed = (a.mode == "seed");
  uint64_t chunks = 0, replicas = 0, bytes = 0, bad = 0;
  for (const auto& chunk : plan.chunks) {
    ++chunks;
    for (const auto& t : chunk.targets) {
      ++replicas;
      const auto expected = patternN(chunk.layer, chunk.position, t.size_bytes);
      if (seed) {
        if (dev.write(t.device, t.noc_addr, expected.data(), expected.size())) {
          bytes += t.size_bytes;
        } else {
          ++bad;
        }
      } else {
        std::vector<uint8_t> buf(t.size_bytes);
        if (dev.read(t.device, t.noc_addr, t.size_bytes, buf.data()) &&
            buf == expected) {
          bytes += t.size_bytes;
        } else {
          ++bad;
        }
      }
    }
  }

  std::cout << "[seed-verify] mode=" << a.mode << " host=" << a.host
            << " chunks=" << chunks << " replicas=" << replicas
            << " bytes=" << bytes << " bad=" << bad << "\n";
  if (seed) {
    std::cout << (bad ? "SEED: FAIL\n" : "SEED: OK\n");
  } else {
    std::cout << (bad ? "VERIFY: FAIL\n" : "VERIFY: PASS\n");
  }
  return bad ? 1 : 0;
}
