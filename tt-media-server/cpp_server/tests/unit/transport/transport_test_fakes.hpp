// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

// Shared in-memory fakes + table builders for the transport data-plane and
// orchestration tests. Each test is its own binary, so a single include per
// binary; the free helpers are inline to be safe.

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/i_device_io.hpp"
#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/in_memory_kv_table.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {
namespace test {

constexpr uint32_t K_CHUNK = 64;  // small chunk for readable byte content

// A blocking, thread-safe channel end: receiveRawData blocks until a message
// arrives or the inbound queue is closed — like a real socket. Two of them with
// crossed queues form a connected pair across threads.
struct Pipe {
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::vector<uint8_t>> q;
  bool closed = false;
};

inline void closePipe(const std::shared_ptr<Pipe>& p) {
  {
    std::lock_guard<std::mutex> lk(p->m);
    p->closed = true;
  }
  p->cv.notify_all();
}

class BlockingFakeTransport : public sockets::ISocketTransport {
 public:
  BlockingFakeTransport(std::shared_ptr<Pipe> in, std::shared_ptr<Pipe> out)
      : in(std::move(in)), out(std::move(out)) {}

  bool initializeAsServer(uint16_t) override { return true; }
  bool initializeAsClient(const std::string&, uint16_t) override {
    return true;
  }
  void start() override {}
  // Models a real transport teardown: closing the inbound queue unblocks a
  // pending receiveRawData() so a receiver loop waiting on it returns.
  void stop() override { closePipe(in); }
  // Reflects pipe closure so KvControlChannel::receive() can tell an
  // empty-on-close read apart from "no data yet" (it disambiguates via
  // isConnected()), like a real socket dropping its connection.
  bool isConnected() const override {
    std::lock_guard<std::mutex> lk(in->m);
    return !in->closed;
  }
  std::string getStatus() const override { return "blocking-fake"; }

  bool sendRawData(std::span<const uint8_t> data) override {
    {
      std::lock_guard<std::mutex> lk(out->m);
      out->q.emplace_back(data.begin(), data.end());
    }
    out->cv.notify_one();
    return true;
  }
  std::vector<uint8_t> receiveRawData() override {
    std::unique_lock<std::mutex> lk(in->m);
    in->cv.wait(lk, [&] { return !in->q.empty() || in->closed; });
    if (in->q.empty()) return {};  // closed
    std::vector<uint8_t> front = std::move(in->q.front());
    in->q.pop_front();
    return front;
  }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}

 private:
  std::shared_ptr<Pipe> in;
  std::shared_ptr<Pipe> out;
};

// Shared segment registry: an engine's advertised name -> its registered host
// buffer (base, length). Models the cluster-wide segment directory.
struct FakeRegistry {
  std::map<std::string, std::pair<uint8_t*, std::size_t>> segs;
};

// Transfer engine whose WRITE memcpies the local buffer into the *target's*
// registered buffer at target_offset — faithfully simulating Mooncake's
// one-sided write landing in the receiver's mirror, with no network.
class FakeTransferEngine : public ITransferEngine {
 public:
  FakeTransferEngine(std::shared_ptr<FakeRegistry> reg, std::string name)
      : reg(std::move(reg)), name(std::move(name)) {}

  StorageMedium storageMedium() const override {
    return StorageMedium::DEVICE_DRAM;
  }
  std::shared_ptr<IStorageBackend> storage() const override { return nullptr; }
  bool init(const EngineConfig&) override { return true; }

  bool registerLocalMemory(void* addr, std::size_t length) override {
    reg->segs[name] = {static_cast<uint8_t*>(addr), length};
    return true;
  }
  bool unregisterLocalMemory(void*) override {
    reg->segs.erase(name);
    return true;
  }
  SegmentHandle openSegment(const std::string& name) override {
    const auto it = reg->segs.find(name);
    if (it == reg->segs.end()) return K_INVALID_SEGMENT;
    const SegmentHandle h = next++;
    opened[h] = it->second.first;
    return h;
  }
  TransferStatus submitAndWait(const TransferRequest& r) override {
    if (r.op != TransferOp::WRITE) return {TransferState::FAILED, 0};
    const auto it = opened.find(r.target);
    if (it == opened.end()) return {TransferState::FAILED, 0};
    std::memcpy(it->second + r.target_offset, r.local_addr, r.length);
    return {TransferState::COMPLETED, r.length};
  }

 private:
  std::shared_ptr<FakeRegistry> reg;
  std::string name;
  std::map<SegmentHandle, uint8_t*> opened;
  SegmentHandle next = 1;
};

// In-memory device DRAM keyed by (device, noc_addr).
class FakeDeviceIo : public IDeviceIo {
 public:
  void put(LocalDeviceId d, NocAddr n, std::vector<uint8_t> bytes) {
    store[{d, n}] = std::move(bytes);
  }
  std::vector<uint8_t> get(LocalDeviceId d, NocAddr n) const {
    const auto it = store.find({d, n});
    return it == store.end() ? std::vector<uint8_t>{} : it->second;
  }
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

// Deterministic chunk content keyed by logical (layer, position).
inline std::vector<uint8_t> makeContent(uint32_t layer, uint32_t pos) {
  std::vector<uint8_t> v(K_CHUNK);
  for (uint32_t i = 0; i < K_CHUNK; ++i) {
    v[i] = static_cast<uint8_t>(layer * 40 + pos + i);
  }
  return v;
}

inline KvTableConfig reducedConfig() {
  KvTableConfig c;
  c.num_layers = 2;
  c.num_slots = 8;
  c.max_sequence_length = 128;  // -> positions 0,32,64,96
  c.chunk_n_tokens = 32;
  c.chunk_size_bytes = K_CHUNK;
  return c;
}

// One-host table: 2 layers, each on a 2-replica group, 4 contiguous positions.
// base/channel differ per layer so prefill and decode tables get distinct
// addrs.
inline InMemoryKvTable buildTable(const std::string& host, FabricNode l0a,
                                  FabricNode l0b, FabricNode l1a,
                                  FabricNode l1b, uint64_t base0, uint32_t ch0,
                                  uint64_t base1, uint32_t ch1,
                                  uint32_t slot = 5) {
  InMemoryKvTable t(reducedConfig());
  const uint32_t g0 = t.addDeviceGroup({l0a, l0b});
  const uint32_t g1 = t.addDeviceGroup({l1a, l1b});
  for (const auto& n : {l0a, l0b, l1a, l1b}) t.setHost(n, host);
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    t.setChunk(slot, 0, p,
               {makeNocAddr(ch0, base0 + idx * K_CHUNK), K_CHUNK, g0});
    t.setChunk(slot, 1, p,
               {makeNocAddr(ch1, base1 + idx * K_CHUNK), K_CHUNK, g1});
  }
  return t;
}

// Like buildTable but splits the two layers across two decode hosts (layer 0 ->
// host0, layer 1 -> host1) — a reduced 1-prefill -> 2-decode-host fan-out for
// the multi-host sender. Each host's receiver loads this full table but mirrors
// only its own layer.
inline InMemoryKvTable buildTableSplitHosts(
    const std::string& host0, const std::string& host1, FabricNode l0a,
    FabricNode l0b, FabricNode l1a, FabricNode l1b, uint64_t base0,
    uint32_t ch0, uint64_t base1, uint32_t ch1, uint32_t slot = 5) {
  InMemoryKvTable t(reducedConfig());
  const uint32_t g0 = t.addDeviceGroup({l0a, l0b});
  const uint32_t g1 = t.addDeviceGroup({l1a, l1b});
  t.setHost(l0a, host0);
  t.setHost(l0b, host0);
  t.setHost(l1a, host1);
  t.setHost(l1b, host1);
  for (uint32_t p = 0; p < 128; p += 32) {
    const uint32_t idx = p / 32;
    t.setChunk(slot, 0, p,
               {makeNocAddr(ch0, base0 + idx * K_CHUNK), K_CHUNK, g0});
    t.setChunk(slot, 1, p,
               {makeNocAddr(ch1, base1 + idx * K_CHUNK), K_CHUNK, g1});
  }
  return t;
}

// Symmetric request: src coords == dst coords.
inline MigrationRequest symmetricReq(uint32_t slot, uint32_t layer_begin,
                                     uint32_t layer_end, uint32_t pos_begin,
                                     uint32_t pos_end) {
  return MigrationRequest{slot,      slot,    layer_begin, layer_end,
                          pos_begin, pos_end, pos_begin,   pos_end};
}

// Asymmetric request: a position shift and/or a cross-slot migration.
inline MigrationRequest asymmetricReq(uint32_t src_slot, uint32_t dst_slot,
                                      uint32_t layer_begin, uint32_t layer_end,
                                      uint32_t src_pos_begin,
                                      uint32_t src_pos_end,
                                      uint32_t dst_pos_begin,
                                      uint32_t dst_pos_end) {
  return MigrationRequest{src_slot,      dst_slot,      layer_begin,
                          layer_end,     src_pos_begin, src_pos_end,
                          dst_pos_begin, dst_pos_end};
}

inline MigrationRequest wholeSlot5() { return symmetricReq(5, 0, 2, 0, 128); }

}  // namespace test
}  // namespace tt::transport
