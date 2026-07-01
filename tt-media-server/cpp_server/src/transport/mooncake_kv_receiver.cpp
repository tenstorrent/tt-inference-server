// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_kv_receiver.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

MooncakeKvReceiver::MooncakeKvReceiver(
    std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
    std::shared_ptr<const IKvTable> localTable, std::string host,
    std::string advertisedSegmentName)
    : engine_(std::move(engine)),
      device_(device),
      local_table_(std::move(localTable)),
      host_(std::move(host)),
      advertised_segment_name_(std::move(advertisedSegmentName)) {
  if (!engine_ || !local_table_) {
    TT_LOG_ERROR("[MooncakeKvReceiver] no engine/table; mirror not registered");
    return;
  }
  // One mirror over the *full* local table, registered once. Its offsets are a
  // stable function of (device, noc_addr) for the receiver's lifetime, so every
  // migration — even concurrent ones to disjoint chunks — addresses the same
  // segment with the same offsets the sender computes from the exchanged table.
  mirror_ = KvCacheMirror(allHostLocations(*local_table_, host_));
  if (mirror_.totalBytes() == 0) {
    TT_LOG_WARN(
        "[MooncakeKvReceiver] local table holds no chunks for host {}; mirror "
        "not registered",
        host_);
    return;
  }
  if (!engine_->registerLocalMemory(mirror_.base(), mirror_.totalBytes())) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] registerLocalMemory({} bytes) failed; mirror not "
        "registered",
        mirror_.totalBytes());
    return;
  }
  registered_ = true;
  TT_LOG_INFO(
      "[MooncakeKvReceiver] registered mirror: {} bytes, {} regions, "
      "segment={}",
      mirror_.totalBytes(), mirror_.layout().numRegions(),
      advertised_segment_name_);
}

MooncakeKvReceiver::~MooncakeKvReceiver() {
  if (registered_ && engine_) {
    engine_->unregisterLocalMemory(mirror_.base());
  }
}

std::optional<std::string> MooncakeKvReceiver::prepareMirror(
    const KvSlice& slice, uint64_t uuid) {
  if (!registered_) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] prepareMirror(uuid={}): mirror not registered",
        uuid);
    return std::nullopt;
  }

  // Reject a duplicate uuid: emplace below would not overwrite, so the stale
  // Pending would linger and the second drain would be lost.
  if (pending_.find(uuid) != pending_.end()) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] prepareMirror(uuid={}): already prepared", uuid);
    return std::nullopt;
  }

  // All-or-nothing: every requested chunk must resolve in the table. A
  // partially-satisfiable request would otherwise migrate only the found subset
  // and ACK success, leaving stale KV at the gaps on decode.
  if (const auto missing = firstUnresolvedChunk(*local_table_, slice)) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] prepareMirror(uuid={}): requested chunk "
        "(slot={}, layer={}, pos={}) absent from table; rejecting whole "
        "request (layers=[{},{}), pos=[{},{}))",
        uuid, slice.slot, missing->layer, missing->position, slice.layer_begin,
        slice.layer_end, slice.position_begin, slice.position_end);
    return std::nullopt;
  }

  HostKvPlan plan = buildHostPlan(*local_table_, host_, slice);
  if (plan.empty()) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] prepareMirror(uuid={}): no local chunks for "
        "request (slot={}, layers=[{},{}), pos=[{},{}))",
        uuid, slice.slot, slice.layer_begin, slice.layer_end,
        slice.position_begin, slice.position_end);
    return std::nullopt;
  }

  // Every requested chunk must fall inside the pre-registered full-table
  // mirror; otherwise the sender's WRITE (and our drain) would address outside
  // it.
  for (const KvChunkLocation& loc : plan.locations) {
    if (!mirror_.offsetOf(loc.device, loc.noc_addr)) {
      TT_LOG_ERROR(
          "[MooncakeKvReceiver] prepareMirror(uuid={}): chunk device={:#x} "
          "noc={:#x} outside registered mirror",
          uuid, loc.device, loc.noc_addr);
      return std::nullopt;
    }
  }

  const std::size_t n = plan.chunks.size();
  pending_.emplace(uuid, std::move(plan));
  TT_LOG_INFO(
      "[MooncakeKvReceiver] prepareMirror(uuid={}) ready: {} chunks, "
      "segment={}",
      uuid, n, advertised_segment_name_);
  return advertised_segment_name_;
}

bool MooncakeKvReceiver::drain(uint64_t uuid) {
  const auto it = pending_.find(uuid);
  if (it == pending_.end()) {
    TT_LOG_ERROR("[MooncakeKvReceiver] drain(uuid={}): no prepared mirror",
                 uuid);
    return false;
  }

  const HostKvPlan& plan = it->second;
  bool ok = true;
  // Selective: write back only this migration's chunks (fan-out: each replica
  // on this host) from the shared mirror, never the untouched span of the
  // mirror.
  for (const ChunkPlan& chunk : plan.chunks) {
    for (const ChunkTarget& target : chunk.targets) {
      const auto offset = mirror_.offsetOf(target.device, target.noc_addr);
      if (!offset) {
        TT_LOG_ERROR(
            "[MooncakeKvReceiver] drain(uuid={}): no mirror offset for "
            "device={:#x} noc={:#x}",
            uuid, target.device, target.noc_addr);
        ok = false;
        continue;
      }
      const uint8_t* src = mirror_.base() + *offset;
      if (!device_.write(target.device, target.noc_addr, src,
                         target.size_bytes)) {
        TT_LOG_ERROR(
            "[MooncakeKvReceiver] drain(uuid={}): device write failed "
            "device={:#x} noc={:#x}",
            uuid, target.device, target.noc_addr);
        ok = false;
      }
    }
  }

  // On success, forget the uuid. On failure, KEEP the per-uuid plan: the sender
  // already WROTE the bytes into the persistent mirror, so a re-sent DoneMarker
  // can re-drive the same drain with no re-transfer (forward recovery). The
  // shared mirror segment stays registered either way. drain() is NOT atomic
  // across chunks — on failure the decode device may hold a partial mix of new
  // and stale KV, so the caller must not consume the slot until a drain
  // succeeds (see README "Contract for a higher-layer caller").
  if (ok) {
    pending_.erase(it);
  }
  TT_LOG_INFO("[MooncakeKvReceiver] drain(uuid={}) -> {}", uuid,
              ok ? "OK" : "PARTIAL (retryable)");
  return ok;
}

}  // namespace tt::transport
