// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_kv_sender.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "transport/worker_health.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {
// Snap a token position down to its chunk boundary.
uint32_t chunkStart(uint32_t begin, uint32_t step) {
  return begin - (begin % step);
}

// Number of chunks a [begin, end) token range covers (start snapped down),
// matching the adapter's forEachRequestedChunk stepping.
uint32_t numChunks(uint32_t begin, uint32_t end, uint32_t step) {
  if (step == 0) return 0;
  const uint32_t start = chunkStart(begin, step);
  return end <= start ? 0u : (end - start + step - 1) / step;
}

// Pair src and dst chunks by (layer, ordinal-within-range): chunk K of the src
// position range maps to chunk K of the dst range. Keying by ordinal (not by
// absolute position) is what makes a position shift (src pos != dst pos) line
// up — layers map 1:1, so the layer is part of the key.
uint64_t loKey(uint32_t layer, uint32_t ordinal) {
  return (static_cast<uint64_t>(layer) << 32) | ordinal;
}
}  // namespace

MooncakeKvSender::MooncakeKvSender(std::shared_ptr<ITransferEngine> engine,
                                   IDeviceIo& device,
                                   std::shared_ptr<const IKvTable> prefillTable,
                                   std::shared_ptr<const IKvTable> decodeTable,
                                   std::string prefillHost,
                                   std::string decodeHost, WorkerHealth* health)
    : engine_(std::move(engine)),
      device_(device),
      health_(health),
      prefill_table_(std::move(prefillTable)),
      decode_table_(std::move(decodeTable)),
      prefill_host_(std::move(prefillHost)),
      decode_host_(std::move(decodeHost)) {
  if (decode_table_) {
    dst_layout_ = KvCacheLayout(allHostLocations(*decode_table_, decode_host_));
  }
}

void MooncakeKvSender::refreshPeerSegment(const std::string& segmentName) {
  if (health_) {
    health_->onTransferFailure();
    health_->onReresolveAttempt();
  }
  TT_LOG_WARN(
      "[MooncakeKvSender] transfer to segment '{}' failed; force-refreshing its "
      "descriptor so the next request re-resolves the peer's current address "
      "(peer may have restarted on a fresh port)",
      segmentName);
  if (engine_->refreshSegment(segmentName) == K_INVALID_SEGMENT && health_) {
    health_->onReresolveFailure();
  }
}

bool MooncakeKvSender::transferSlot(const MigrationRequest& request,
                                    const std::string& segmentName) {
  if (!engine_ || !prefill_table_ || !decode_table_) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: no engine/tables");
    return false;
  }

  const uint32_t step = decode_table_->config().chunk_n_tokens;
  if (step == 0) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: chunk_n_tokens=0");
    return false;
  }
  // The src and dst position ranges must cover the same number of chunks: the
  // sender pairs src chunk K with dst chunk K by ordinal, so a count mismatch
  // has no 1:1 mapping (a chunk would be left unsent or unmatched).
  const uint32_t srcChunks =
      numChunks(request.src_position_begin, request.src_position_end, step);
  const uint32_t dstChunks =
      numChunks(request.dst_position_begin, request.dst_position_end, step);
  if (srcChunks != dstChunks) {
    TT_LOG_ERROR(
        "[MooncakeKvSender] transferSlot: src/dst position ranges cover "
        "different chunk counts (src={} dst={}); they must align",
        srcChunks, dstChunks);
    return false;
  }

  const KvSlice srcSlice = request.srcSlice();
  const KvSlice dstSlice = request.dstSlice();
  const uint32_t srcStart = chunkStart(request.src_position_begin, step);
  const uint32_t dstStart = chunkStart(request.dst_position_begin, step);

  // Destination layout from the decode table (built identically to the
  // receiver's mirror) + the opened segment.
  const HostKvPlan decodePlan =
      buildHostPlan(*decode_table_, decode_host_, dstSlice);
  if (decodePlan.empty()) {
    TT_LOG_ERROR(
        "[MooncakeKvSender] transferSlot: no decode chunks for request");
    return false;
  }
  const SegmentHandle segment = engine_->openSegment(segmentName);
  if (segment == K_INVALID_SEGMENT) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: openSegment({}) failed",
                 segmentName);
    refreshPeerSegment(segmentName);
    return false;
  }

  // Source plan: where to read each chunk from on the prefill side, keyed by
  // (layer, ordinal) so the dst chunk K below finds src chunk K under a shift.
  const HostKvPlan prefillPlan =
      buildHostPlan(*prefill_table_, prefill_host_, srcSlice);
  std::unordered_map<uint64_t, const ChunkPlan*> source;
  source.reserve(prefillPlan.chunks.size());
  for (const ChunkPlan& chunk : prefillPlan.chunks) {
    const uint32_t ordinal = (chunk.position - srcStart) / step;
    source.emplace(loKey(chunk.layer, ordinal), &chunk);
  }

  // One registered staging buffer, sized to the largest chunk, reused per
  // chunk.
  uint64_t maxChunk = 0;
  for (const ChunkPlan& chunk : decodePlan.chunks) {
    for (const ChunkTarget& t : chunk.targets)
      maxChunk = std::max(maxChunk, t.size_bytes);
  }
  std::vector<uint8_t> staging(maxChunk, 0);
  if (!engine_->registerLocalMemory(staging.data(), staging.size())) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: registerLocalMemory failed");
    return false;
  }

  bool ok = true;
  // A failed WRITE (vs a planning error like a missing source or size mismatch)
  // is the symptom of a stale segment descriptor after a peer restart, so track
  // it separately and only force-refresh for that case.
  bool transferFailed = false;
  for (const ChunkPlan& chunk : decodePlan.chunks) {
    const uint32_t ordinal = (chunk.position - dstStart) / step;
    const auto srcIt = source.find(loKey(chunk.layer, ordinal));
    if (srcIt == source.end() || srcIt->second->targets.empty()) {
      TT_LOG_ERROR(
          "[MooncakeKvSender] no source for dst chunk (dst_slot={}, layer={}, "
          "dst_pos={}, ordinal={})",
          chunk.slot, chunk.layer, chunk.position, ordinal);
      ok = false;
      continue;
    }
    // Read once from the primary source replica. The staging buffer is sized to
    // the largest decode chunk, so the prefill source chunk must fit in it,
    // otherwise overflow can happen
    const ChunkTarget& src = srcIt->second->targets.front();
    if (src.size_bytes > staging.size()) {
      TT_LOG_ERROR(
          "[MooncakeKvSender] prefill chunk larger than staging buffer "
          "(slot={}, layer={}, pos={}): src={} staging={}",
          chunk.slot, chunk.layer, chunk.position, src.size_bytes,
          staging.size());
      ok = false;
      continue;
    }
    if (!device_.read(src.device, src.noc_addr, src.size_bytes,
                      staging.data())) {
      TT_LOG_ERROR("[MooncakeKvSender] read failed device={:#x} noc={:#x}",
                   src.device, src.noc_addr);
      ok = false;
      continue;
    }
    // Push to every decode replica's mirror offset (fan-out). The WRITE length
    // is the decode size, it must match the bytes we just read from the source
    // (same model), or we'd ship staging bytes that were never read.
    for (const ChunkTarget& dst : chunk.targets) {
      // The WRITE length is the decode size, it must equal the bytes we just
      // read from the source (same model), or we'd ship staging bytes that
      // were never read. Enforce at runtime: a mismatch would otherwise leak
      // stale staging contents to the decode replica's mirror.
      if (dst.size_bytes != src.size_bytes) {
        TT_LOG_ERROR(
            "[MooncakeKvSender] decode/prefill chunk size mismatch "
            "(slot={}, layer={}, pos={}): src={} dst={}",
            chunk.slot, chunk.layer, chunk.position, src.size_bytes,
            dst.size_bytes);
        ok = false;
        continue;
      }
      const auto offset = dst_layout_.offsetOf(dst.device, dst.noc_addr);
      if (!offset) {
        TT_LOG_ERROR(
            "[MooncakeKvSender] no mirror offset device={:#x} noc={:#x}",
            dst.device, dst.noc_addr);
        ok = false;
        continue;
      }
      TransferRequest req;
      req.op = TransferOp::WRITE;
      req.local_addr = staging.data();
      req.target = segment;
      req.target_offset = *offset;
      req.length = dst.size_bytes;
      if (engine_->submitAndWait(req).state != TransferState::COMPLETED) {
        TT_LOG_ERROR(
            "[MooncakeKvSender] WRITE failed device={:#x} noc={:#x} offset={}",
            dst.device, dst.noc_addr, *offset);
        ok = false;
        transferFailed = true;
      }
    }
  }

  engine_->unregisterLocalMemory(staging.data());
  // Re-resolve for the NEXT request only; this one still fails and acks FAILED.
  if (transferFailed) refreshPeerSegment(segmentName);
  return ok;
}

}  // namespace tt::transport
