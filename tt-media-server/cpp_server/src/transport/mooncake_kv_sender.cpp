// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_kv_sender.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {
// Read an unsigned tunable from the environment, falling back to `fallback` on
// unset/empty/unparseable/zero.
uint64_t envU64(const char* key, uint64_t fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  try {
    const unsigned long long parsed = std::stoull(v);
    return parsed == 0 ? fallback : static_cast<uint64_t>(parsed);
  } catch (...) {
    return fallback;
  }
}

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

uint64_t defaultStagingBytes() {
  return envU64("TT_MOONCAKE_STAGING_BYTES", K_DEFAULT_STAGING_BYTES);
}

uint32_t stagingWindowDivisor() {
  // Clamp to >= 1: a 0 divisor is meaningless (envU64 already maps 0 ->
  // default).
  return static_cast<uint32_t>(std::max<uint64_t>(
      1, envU64("TT_MOONCAKE_WINDOW_DIVISOR", K_DEFAULT_WINDOW_DIVISOR)));
}

KvStagingPool::KvStagingPool(std::shared_ptr<ITransferEngine> engine,
                             uint64_t bufferBytes)
    : engine_(std::move(engine)) {
  if (!engine_) {
    TT_LOG_ERROR("[KvStagingPool] no engine; not registered");
    return;
  }
  for (auto& b : buffers_) {
    b.resize(bufferBytes);  // contents are overwritten by device reads
    if (!engine_->registerLocalMemory(b.data(), b.size())) {
      TT_LOG_ERROR("[KvStagingPool] registerLocalMemory({} bytes) failed",
                   b.size());
      return;  // registered_ stays false; already-registered freed in dtor
    }
    ++registered_count_;
  }
  registered_ = true;
  TT_LOG_INFO("[KvStagingPool] registered {} buffers x {} bytes", kBuffers,
              bufferBytes);
}

KvStagingPool::~KvStagingPool() {
  if (!engine_) return;
  for (int i = 0; i < registered_count_; ++i) {
    engine_->unregisterLocalMemory(buffers_[i].data());
  }
}

MooncakeKvSender::MooncakeKvSender(std::shared_ptr<ITransferEngine> engine,
                                   IDeviceIo& device,
                                   std::shared_ptr<const IKvTable> prefillTable,
                                   std::shared_ptr<const IKvTable> decodeTable,
                                   std::string prefillHost,
                                   std::string decodeHost,
                                   std::shared_ptr<KvStagingPool> staging)
    : engine_(std::move(engine)),
      device_(device),
      prefill_table_(std::move(prefillTable)),
      decode_table_(std::move(decodeTable)),
      prefill_host_(std::move(prefillHost)),
      decode_host_(std::move(decodeHost)),
      staging_(std::move(staging)) {
  if (decode_table_) {
    dst_layout_ = KvCacheLayout(allHostLocations(*decode_table_, decode_host_));
  }
  // If no shared pool was injected, own one and register it now — every caller
  // has already initialized the engine, so paying the (one-time) registration
  // here keeps it off the first migration's critical path.
  if (!staging_ && engine_) {
    staging_ = std::make_shared<KvStagingPool>(engine_);
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

  // Plan every decode chunk up front: resolve its source replica, validate the
  // src/dst sizes, and pre-compute each fan-out replica's mirror offset. Doing
  // the validation here keeps the transfer loop below a straight
  // read-into-staging → batch, and lets us size the staging buffer to the real
  // per-chunk sizes.
  struct PlannedChunk {
    const ChunkTarget* src;  // source replica to read on the prefill side
    uint64_t size;           // src == dst chunk size (validated equal)
    std::vector<uint64_t> dst_offsets;  // mirror offset per decode replica
    uint32_t slot, layer, position;     // diagnostics only
  };

  bool ok = true;
  std::vector<PlannedChunk> planned;
  planned.reserve(decodePlan.chunks.size());
  uint64_t totalBytes = 0;
  uint64_t maxChunk = 0;
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
    // Read once from the primary source replica; fan out to every decode
    // replica from those same staged bytes.
    const ChunkTarget& src = srcIt->second->targets.front();
    PlannedChunk pc;
    pc.src = &src;
    pc.size = src.size_bytes;
    pc.slot = chunk.slot;
    pc.layer = chunk.layer;
    pc.position = chunk.position;
    for (const ChunkTarget& dst : chunk.targets) {
      // The WRITE length is the decode size; it must equal the bytes we read
      // from the source (same model), or we'd ship staging bytes that were
      // never read. A mismatch would leak stale staging to the decode mirror.
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
      pc.dst_offsets.push_back(*offset);
    }
    if (pc.dst_offsets.empty()) continue;  // nothing writable for this chunk
    totalBytes += pc.size;
    maxChunk = std::max(maxChunk, pc.size);
    planned.push_back(std::move(pc));
  }

  if (planned.empty()) {
    // No writable chunks: either the request resolved to none or every chunk
    // failed validation above (ok already reflects that).
    return ok;
  }

  // Staging buffers are registered ONCE (at construction) and reused across
  // every migration (see KvStagingPool): registration — especially RDMA
  // ibv_reg_mr — is far too costly to repeat per slot.
  if (!staging_ || !staging_->registered()) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: no registered staging");
    return false;
  }
  const uint64_t bufBytes = staging_->bufferBytes();
  if (maxChunk > bufBytes) {
    TT_LOG_ERROR(
        "[MooncakeKvSender] transferSlot: chunk ({} bytes) exceeds staging "
        "buffer ({} bytes)",
        maxChunk, bufBytes);
    return false;
  }

  // Merge chunks into contiguous segments. Both KV layouts store a
  // (layer, channel) span contiguously by position, and src chunk K pairs with
  // dst chunk K by ordinal, so a run of consecutive positions is contiguous on
  // BOTH sides at once: one device read fills it, and each fan-out replica is a
  // single WRITE of the whole run. This is the key to throughput — Mooncake's
  // TCP transport (and each device read) pays a large per-op cost, so
  // collapsing thousands of ~KB ops into a handful of large ones is where the
  // time goes.
  //
  // Sort by source (device, noc_addr) so contiguous chunks are adjacent, then
  // extend the current segment while every one of these holds:
  //   * same source device,
  //   * source address contiguous (prev.noc + accumulated == cur.noc),
  //   * same replica count, and each replica's mirror offset contiguous,
  //   * the segment still fits one (pre-registered) staging buffer.
  // Any break starts a new segment. Contiguity is checked, never assumed, so a
  // layout that doesn't line up simply yields shorter segments — never wrong
  // bytes.
  std::sort(planned.begin(), planned.end(),
            [](const PlannedChunk& a, const PlannedChunk& b) {
              if (a.src->device != b.src->device)
                return a.src->device < b.src->device;
              return a.src->noc_addr < b.src->noc_addr;
            });

  // --- diagnostic: why chunks (don't) merge ------------------------------
  // Over the sorted list, count adjacent pairs that are source-contiguous vs
  // dest(mirror)-contiguous vs both. Zero segments merged => srcAdj and/or
  // dstAdj are ~0, which tells us whether the source DRAM, the dest mirror, or
  // both are strided. Also dump the first few chunks so we can see the stride.
  {
    uint64_t srcAdj = 0, dstAdj = 0, bothAdj = 0;
    for (std::size_t i = 1; i < planned.size(); ++i) {
      const PlannedChunk& a = planned[i - 1];
      const PlannedChunk& b = planned[i];
      const bool s = a.src->device == b.src->device &&
                     a.src->noc_addr + a.size == b.src->noc_addr;
      bool d = a.dst_offsets.size() == b.dst_offsets.size();
      for (std::size_t j = 0; d && j < a.dst_offsets.size(); ++j) {
        if (a.dst_offsets[j] + a.size != b.dst_offsets[j]) d = false;
      }
      srcAdj += s ? 1 : 0;
      dstAdj += d ? 1 : 0;
      bothAdj += (s && d) ? 1 : 0;
    }
    TT_LOG_INFO(
        "[MooncakeKvSender] contiguity: {} pairs | srcAdj={} dstAdj={} both={}",
        planned.empty() ? 0 : planned.size() - 1, srcAdj, dstAdj, bothAdj);
    for (std::size_t i = 0; i < planned.size() && i < 6; ++i) {
      const PlannedChunk& p = planned[i];
      TT_LOG_INFO(
          "[MooncakeKvSender]   [{}] dev={:#x} noc={:#x} size={} dst0={:#x} "
          "L{} pos={}",
          i, p.src->device, p.src->noc_addr, p.size,
          p.dst_offsets.empty() ? 0 : p.dst_offsets[0], p.layer, p.position);
    }
  }

  struct Segment {
    LocalDeviceId device;               // source device to read from
    NocAddr noc_addr;                   // source start address
    uint64_t size;                      // total contiguous bytes
    std::vector<uint64_t> dst_offsets;  // mirror start offset per replica
  };
  std::vector<Segment> segments;
  segments.reserve(planned.size());
  uint64_t maxSegment = 0;
  for (const PlannedChunk& pc : planned) {
    bool extend = false;
    if (!segments.empty()) {
      Segment& s = segments.back();
      extend = s.device == pc.src->device &&
               s.noc_addr + s.size == pc.src->noc_addr &&
               s.dst_offsets.size() == pc.dst_offsets.size() &&
               s.size + pc.size <= bufBytes;
      for (std::size_t j = 0; extend && j < s.dst_offsets.size(); ++j) {
        if (s.dst_offsets[j] + s.size != pc.dst_offsets[j]) extend = false;
      }
    }
    if (extend) {
      segments.back().size += pc.size;
    } else {
      segments.push_back(
          Segment{pc.src->device, pc.src->noc_addr, pc.size, pc.dst_offsets});
    }
    maxSegment = std::max(maxSegment, segments.back().size);
  }
  TT_LOG_INFO(
      "[MooncakeKvSender] merged {} chunks -> {} segments (max segment {}B)",
      planned.size(), segments.size(), maxSegment);

  // Double-buffered pipeline. The slot moves in windows; while one window's
  // batch runs on the engine's threads, the next window is staged from device
  // DRAM into the *other* buffer — overlapping device reads with network
  // writes. At most one batch in flight. Window size targets `divisor` windows
  // per slot (so there is something to overlap), floored at one whole segment
  // and capped at the (fixed, pre-registered) buffer size. Both knobs are
  // env-tunable.
  const uint64_t divisor = stagingWindowDivisor();
  const uint64_t windowCap = std::min(
      bufBytes,
      std::max<uint64_t>(maxSegment, (totalBytes + divisor - 1) / divisor));
  TT_LOG_INFO(
      "[MooncakeKvSender] staging: buf={}B divisor={} window={}B (~{} windows) "
      "total={}B",
      bufBytes, divisor, windowCap, (totalBytes + windowCap - 1) / windowCap,
      totalBytes);

  int buf = 0;                         // buffer we are currently filling
  uint64_t used = 0;                   // bytes used in buffer `buf`
  std::vector<TransferRequest> batch;  // fan-out WRITEs staged in buffer `buf`
  TransferHandle inflight;             // the *previous* window's transfer

  // --- read/write split instrumentation (diagnostic) ----------------------
  // Accumulate wall time actually spent inside device DRAM reads vs network
  // submit/wait, so we can tell which side is the bottleneck. With the pipeline
  // overlapping the two, readNs >> waitNs means device-read bound (network
  // hidden); waitNs >> readNs means network bound (reads hidden).
  using DiagClock = std::chrono::steady_clock;
  auto diagNs = [](DiagClock::time_point a, DiagClock::time_point b) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  uint64_t readNs = 0, submitNs = 0, waitNs = 0;
  uint64_t readBytes = 0, writeBytes = 0;
  uint64_t readOps = 0, batchOps = 0, writeOps = 0;

  // Finish the current window: wait for the previous window's transfer (this
  // bounds concurrency to one in flight and frees its buffer for reuse), then
  // dispatch this window asynchronously and switch to the other buffer. The
  // reads that fill the next window then overlap this dispatch.
  auto dispatch = [&]() {
    if (batch.empty()) return;
    if (inflight.valid) {
      const auto w0 = DiagClock::now();
      const bool bad =
          engine_->waitBatch(inflight).state != TransferState::COMPLETED;
      waitNs += diagNs(w0, DiagClock::now());
      if (bad) {
        TT_LOG_ERROR("[MooncakeKvSender] batch WRITE failed");
        ok = false;
      }
    }
    const auto s0 = DiagClock::now();
    inflight = engine_->submitBatch(batch);
    submitNs += diagNs(s0, DiagClock::now());
    ++batchOps;
    if (!inflight.valid) {
      TT_LOG_ERROR("[MooncakeKvSender] submitBatch of {} transfers failed",
                   batch.size());
      ok = false;
    }
    batch.clear();
    used = 0;
    buf ^= 1;
  };

  for (const Segment& seg : segments) {
    // windowCap >= maxSegment >= seg.size, so after a dispatch the segment
    // fits.
    if (used + seg.size > windowCap) dispatch();
    uint8_t* dst = staging_->buffer(buf) + used;
    // One device read fills the whole contiguous segment.
    const auto r0 = DiagClock::now();
    const bool readOk = device_.read(seg.device, seg.noc_addr, seg.size, dst);
    readNs += diagNs(r0, DiagClock::now());
    readBytes += seg.size;
    ++readOps;
    if (!readOk) {
      TT_LOG_ERROR("[MooncakeKvSender] read failed device={:#x} noc={:#x}",
                   seg.device, seg.noc_addr);
      ok = false;
      continue;  // leave `used` unchanged; the slot is free for the next
                 // segment
    }
    // One WRITE per replica covers the whole segment.
    for (const uint64_t offset : seg.dst_offsets) {
      TransferRequest req;
      req.op = TransferOp::WRITE;
      req.local_addr = dst;
      req.target = segment;
      req.target_offset = offset;
      req.length = seg.size;
      batch.push_back(req);
      writeBytes += seg.size;
      ++writeOps;
    }
    used += seg.size;
  }
  dispatch();  // submit the final window (waits the prior one)
  if (inflight.valid) {
    const auto w0 = DiagClock::now();
    const bool bad =
        engine_->waitBatch(inflight).state != TransferState::COMPLETED;
    waitNs += diagNs(w0, DiagClock::now());
    if (bad) {
      TT_LOG_ERROR("[MooncakeKvSender] final batch WRITE failed");
      ok = false;
    }
  }

  // Report the read/write split. read_* is device DRAM; wait_* is time blocked
  // on network completion (submit is async, so submitNs is just enqueue cost).
  // Because reads overlap the previous window's network transfer, the dominant
  // term names the bottleneck: reads hidden under network => wait >> read.
  const auto mbps = [](uint64_t bytes, uint64_t ns) {
    return ns == 0 ? 0.0
                   : (static_cast<double>(bytes) / 1e6) /
                         (static_cast<double>(ns) / 1e9);
  };
  TT_LOG_INFO(
      "[MooncakeKvSender] split: read={}ms ({} ops, {}B, {:.1f} MB/s) | "
      "waitBatch={}ms ({}B over {} batch(es), {:.1f} MB/s) | "
      "submit={}ms ({} writes enqueued)",
      readNs / 1'000'000, readOps, readBytes, mbps(readBytes, readNs),
      waitNs / 1'000'000, writeBytes, batchOps, mbps(writeBytes, waitNs),
      submitNs / 1'000'000, writeOps);

  // Staging stays registered for the next migration — no per-slot unregister.
  return ok;
}

}  // namespace tt::transport
