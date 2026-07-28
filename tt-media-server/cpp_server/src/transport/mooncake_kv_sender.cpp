// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_kv_sender.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "transport/worker_health.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {
// Snap a token position down to its chunk boundary.
uint32_t chunkStart(uint32_t begin, uint32_t step) {
  return begin - (begin % step);
}

// Number of chunks a [begin, end) token range covers (start snapped down).
uint32_t numChunks(uint32_t begin, uint32_t end, uint32_t step) {
  if (step == 0) return 0;
  const uint32_t start = chunkStart(begin, step);
  return end <= start ? 0u : (end - start + step - 1) / step;
}

// Pair src and dst chunks by (layer, ordinal-within-range).
uint64_t loKey(uint32_t layer, uint32_t ordinal) {
  return (static_cast<uint64_t>(layer) << 32) | ordinal;
}

// A single logical chunk: where to read it on the prefill side, and the decode
// device targets it fans out to. `size` is validated equal on both sides.
struct PlannedChunk {
  LocalDeviceId src_device = 0;
  NocAddr src_noc = 0;
  uint64_t size = 0;
  std::vector<DrainTarget> targets;  // decode replicas
};

// A source-contiguous run of chunks that shares one bounce section: one device
// read, one WRITE into the slot, and per-replica contiguous device targets.
struct Segment {
  LocalDeviceId src_device = 0;
  NocAddr src_noc = 0;
  uint64_t size = 0;
  std::vector<DrainTarget> targets;  // base addr per replica
};
}  // namespace

MooncakeKvSender::MooncakeKvSender(std::shared_ptr<ITransferEngine> engine,
                                   IDeviceIo& device,
                                   std::shared_ptr<const IKvTable> prefillTable,
                                   std::shared_ptr<const IKvTable> decodeTable,
                                   std::string prefillHost,
                                   std::string decodeHost,
                                   std::shared_ptr<KvStagingPool> staging,
                                   WorkerHealth* health)
    : engine_(std::move(engine)),
      device_(device),
      health_(health),
      prefill_table_(std::move(prefillTable)),
      decode_table_(std::move(decodeTable)),
      prefill_host_(std::move(prefillHost)),
      decode_host_(std::move(decodeHost)),
      staging_(std::move(staging)) {
  if (!staging_ && engine_) {
    staging_ = std::make_shared<KvStagingPool>(engine_);
  }
}

void MooncakeKvSender::refreshPeerSegment(const std::string& segmentName) {
  if (health_) {
    health_->onTransferFailure();
    health_->onReresolveAttempt();
  }
  TT_LOG_WARN(
      "[MooncakeKvSender] transfer to segment '{}' failed; force-refreshing "
      "its descriptor for the next request",
      segmentName);
  if (engine_->refreshSegment(segmentName) == K_INVALID_SEGMENT && health_) {
    health_->onReresolveFailure();
  }
}

bool MooncakeKvSender::transferSlot(uint64_t uuid,
                                    const MigrationRequest& request,
                                    const std::string& segmentName,
                                    const BounceGeometry& geometry,
                                    const WindowSink& sink) {
  if (!engine_ || !prefill_table_ || !decode_table_) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: no engine/tables");
    return false;
  }
  if (!geometry.valid()) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: invalid bounce geometry");
    return false;
  }
  if (!staging_ || !staging_->registered()) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: no registered staging");
    return false;
  }

  const uint32_t step = decode_table_->config().chunk_n_tokens;
  if (step == 0) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: chunk_n_tokens=0");
    return false;
  }
  const uint32_t srcChunks =
      numChunks(request.src_position_begin, request.src_position_end, step);
  const uint32_t dstChunks =
      numChunks(request.dst_position_begin, request.dst_position_end, step);
  if (srcChunks != dstChunks) {
    TT_LOG_ERROR(
        "[MooncakeKvSender] transferSlot: src/dst chunk counts differ "
        "(src={} dst={})",
        srcChunks, dstChunks);
    return false;
  }

  const KvSlice srcSlice = request.srcSlice();
  const KvSlice dstSlice = request.dstSlice();
  const uint32_t srcStart = chunkStart(request.src_position_begin, step);
  const uint32_t dstStart = chunkStart(request.dst_position_begin, step);

  const HostKvPlan decodePlan =
      buildHostPlan(*decode_table_, decode_host_, dstSlice);
  if (decodePlan.empty()) {
    TT_LOG_ERROR("[MooncakeKvSender] transferSlot: no decode chunks");
    return false;
  }

  // Source plan keyed by (layer, ordinal) so dst chunk k finds src chunk k.
  const HostKvPlan prefillPlan =
      buildHostPlan(*prefill_table_, prefill_host_, srcSlice);
  std::unordered_map<uint64_t, const ChunkPlan*> source;
  source.reserve(prefillPlan.chunks.size());
  for (const ChunkPlan& chunk : prefillPlan.chunks) {
    const uint32_t ordinal = (chunk.position - srcStart) / step;
    source.emplace(loKey(chunk.layer, ordinal), &chunk);
  }

  // A merged run must fit one bounce section AND one staging buffer.
  const uint64_t sectionCap =
      std::min<uint64_t>(geometry.section_size, staging_->bufferBytes());

  // Plan each decode chunk: resolve its source replica, validate sizes, keep
  // the per-replica device targets (device, noc_addr) the receiver drains to.
  bool ok = true;
  std::vector<PlannedChunk> planned;
  planned.reserve(decodePlan.chunks.size());
  for (const ChunkPlan& chunk : decodePlan.chunks) {
    const uint32_t ordinal = (chunk.position - dstStart) / step;
    const auto srcIt = source.find(loKey(chunk.layer, ordinal));
    if (srcIt == source.end() || srcIt->second->targets.empty()) {
      TT_LOG_ERROR(
          "[MooncakeKvSender] no source for dst chunk (slot={}, layer={}, "
          "pos={}, ordinal={})",
          chunk.slot, chunk.layer, chunk.position, ordinal);
      ok = false;
      continue;
    }
    const ChunkTarget& src = srcIt->second->targets.front();
    if (src.size_bytes > sectionCap) {
      TT_LOG_ERROR(
          "[MooncakeKvSender] chunk ({} B) exceeds bounce section / staging "
          "({} B)",
          src.size_bytes, sectionCap);
      return false;
    }
    PlannedChunk pc;
    pc.src_device = src.device;
    pc.src_noc = src.noc_addr;
    pc.size = src.size_bytes;
    for (const ChunkTarget& dst : chunk.targets) {
      if (dst.size_bytes != src.size_bytes) {
        TT_LOG_ERROR(
            "[MooncakeKvSender] src/dst chunk size mismatch (slot={}, "
            "layer={}, pos={}): src={} dst={}",
            chunk.slot, chunk.layer, chunk.position, src.size_bytes,
            dst.size_bytes);
        ok = false;
        continue;
      }
      pc.targets.push_back(DrainTarget{dst.device, dst.noc_addr});
    }
    if (pc.targets.empty()) continue;
    planned.push_back(std::move(pc));
  }
  if (planned.empty()) return ok;  // ok already reflects any validation failure

  // Sort by source (device, noc) so contiguous chunks are adjacent, then merge
  // runs that are contiguous on BOTH the source read and every replica target,
  // capped at one bounce section. Contiguity is always checked, never assumed.
  std::sort(planned.begin(), planned.end(),
            [](const PlannedChunk& a, const PlannedChunk& b) {
              if (a.src_device != b.src_device)
                return a.src_device < b.src_device;
              return a.src_noc < b.src_noc;
            });

  std::vector<Segment> segments;
  segments.reserve(planned.size());
  for (const PlannedChunk& pc : planned) {
    bool extend = false;
    if (!segments.empty()) {
      Segment& s = segments.back();
      extend = s.src_device == pc.src_device &&
               s.src_noc + s.size == pc.src_noc &&
               s.targets.size() == pc.targets.size() &&
               s.size + pc.size <= sectionCap;
      for (std::size_t j = 0; extend && j < s.targets.size(); ++j) {
        if (s.targets[j].device != pc.targets[j].device ||
            s.targets[j].noc_addr + s.size != pc.targets[j].noc_addr) {
          extend = false;
        }
      }
    }
    if (extend) {
      segments.back().size += pc.size;
    } else {
      Segment s;
      s.src_device = pc.src_device;
      s.src_noc = pc.src_noc;
      s.size = pc.size;
      s.targets = pc.targets;  // base addr per replica
      segments.push_back(std::move(s));
    }
  }
  TT_LOG_INFO(
      "[MooncakeKvSender] uuid={} planned {} chunks -> {} segments; bounce "
      "buffer "
      "{} sections x {} B",
      uuid, planned.size(), segments.size(), geometry.section_count,
      geometry.section_size);

  const SegmentHandle segment = engine_->openSegment(segmentName);
  if (segment == K_INVALID_SEGMENT) {
    TT_LOG_ERROR("[MooncakeKvSender] openSegment({}) failed", segmentName);
    refreshPeerSegment(segmentName);
    return false;
  }

  // Partition the segments into windows, each bounded by BOTH the bounce
  // section count and the staging buffer size. (seg.size <= sectionCap <=
  // bufBytes, so a segment always fits an empty window — no window is ever
  // empty/undersized.)
  const uint64_t bufBytes = staging_->bufferBytes();
  struct WindowPlan {
    std::vector<const Segment*> segs;
    uint64_t bytes = 0;
  };
  std::vector<WindowPlan> windows;
  for (const Segment& seg : segments) {
    if (windows.empty() ||
        windows.back().segs.size() >= geometry.section_count ||
        windows.back().bytes + seg.size > bufBytes) {
      windows.emplace_back();
    }
    windows.back().segs.push_back(&seg);
    windows.back().bytes += seg.size;
  }

  // Depth-2 software pipeline: stage window i's DEVICE READS into
  // staging buffer i%2 WHILE window i-1's WRITE batch is in flight on the
  // network (submitBatch is async); then wait i-1, drain it (sink -> receiver
  // drain -> frees i-1's bounce sections), and submit window i. So each
  // window's device reads overlap the previous window's network transfer — on
  // DRISC the reads are true NOC-DMA (readAsync); on MMIO/host they run inline
  // (the async interface's synchronous default) and only the network overlap
  // remains.
  BounceSectionAllocator alloc(geometry);
  bool transferFailed = false;

  using DiagClock = std::chrono::steady_clock;
  auto diagNs = [](DiagClock::time_point a, DiagClock::time_point b) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  uint64_t readNs = 0, netNs = 0, readBytes = 0, writeBytes = 0;
  uint64_t readOps = 0, windowOps = 0;

  TransferHandle inflight;  // the PREVIOUS window's in-flight WRITE batch
  std::vector<BounceSectionDescriptor> inflightWindow;

  // Wait the in-flight window's network WRITEs, drain it on the receiver
  // (sink), and release its bounce sections. @return false on a transfer/drain
  // failure.
  auto retireInflight = [&]() -> bool {
    if (!inflight.valid) return true;
    const auto w0 = DiagClock::now();
    const bool bad =
        engine_->waitBatch(inflight).state != TransferState::COMPLETED;
    netNs += diagNs(w0, DiagClock::now());
    if (bad) {
      TT_LOG_ERROR("[MooncakeKvSender] window WRITE batch failed");
      transferFailed = true;
      inflight = TransferHandle{};
      return false;
    }
    const bool cont = sink(uuid, inflightWindow);
    alloc.release(static_cast<uint32_t>(inflightWindow.size()));
    inflight = TransferHandle{};
    inflightWindow.clear();
    return cont;
  };

  for (std::size_t i = 0; ok && i < windows.size(); ++i) {
    const WindowPlan& win = windows[i];
    uint8_t* const buf = staging_->buffer(static_cast<int>(i % 2));

    // Stage W's segment reads into buf at distinct offsets. Issued via the
    // async interface so a DRISC backend DMAs them in the background
    // (overlapping the previous window's network); a synchronous backend
    // completes each inline.
    std::vector<std::pair<uint64_t, const Segment*>> staged;
    staged.reserve(win.segs.size());
    uint64_t off = 0;
    const auto r0 = DiagClock::now();
    for (const Segment* seg : win.segs) {
      bool issued = false;
      while (!(issued = device_.readAsync(seg->src_device, seg->src_noc,
                                          seg->size, buf + off))) {
        // Couldn't issue: retire an in-flight device read and retry. Nothing in
        // flight to retire => this was a hard read failure, not backpressure.
        if (!device_.tryPopCompleted() && device_.asyncInFlight() == 0) break;
      }
      if (!issued) {
        TT_LOG_ERROR("[MooncakeKvSender] read failed device={:#x} noc={:#x}",
                     seg->src_device, seg->src_noc);
        ok = false;
        break;
      }
      staged.emplace_back(off, seg);
      readBytes += seg->size;
      ++readOps;
      off += seg->size;
    }
    // Drain W's device reads to completion (sync backends: nothing in flight).
    while (ok && device_.asyncInFlight() > 0) device_.tryPopCompleted();
    readNs += diagNs(r0, DiagClock::now());
    if (!ok) break;

    // Retire the PREVIOUS window (its network overlapped W's reads above),
    // which frees its bounce sections for W.
    if (!retireInflight()) {
      ok = false;
      break;
    }

    // Assign bounce sections to W and submit its WRITE batch asynchronously.
    std::vector<TransferRequest> batch;
    std::vector<BounceSectionDescriptor> wdesc;
    batch.reserve(staged.size());
    wdesc.reserve(staged.size());
    for (const auto& [soff, seg] : staged) {
      const auto section = alloc.alloc();
      if (!section) {
        TT_LOG_ERROR("[MooncakeKvSender] no bounce section for window {}", i);
        ok = false;
        break;
      }
      TransferRequest req;
      req.op = TransferOp::WRITE;
      req.local_addr = buf + soff;
      req.target = segment;
      req.target_offset = *section;
      req.length = seg->size;
      batch.push_back(req);
      writeBytes += seg->size;
      wdesc.push_back(
          BounceSectionDescriptor{*section, seg->size, seg->targets});
    }
    if (!ok) break;
    const auto s0 = DiagClock::now();
    inflight = engine_->submitBatch(batch);
    netNs += diagNs(s0, DiagClock::now());  // enqueue cost; waitBatch dominates
    ++windowOps;
    if (!inflight.valid) {
      TT_LOG_ERROR("[MooncakeKvSender] submitBatch of {} transfers failed",
                   batch.size());
      ok = false;
      transferFailed = true;
      break;
    }
    inflightWindow = std::move(wdesc);
  }

  // Drain the final in-flight window; on abort, still wait a dispatched batch
  // so the engine isn't left reading freed staging.
  if (ok) {
    if (!retireInflight()) ok = false;
  } else if (inflight.valid) {
    engine_->waitBatch(inflight);
  }

  const auto mbps = [](uint64_t bytes, uint64_t ns) {
    return ns == 0 ? 0.0
                   : (static_cast<double>(bytes) / 1e6) /
                         (static_cast<double>(ns) / 1e9);
  };
  TT_LOG_INFO(
      "[MooncakeKvSender] uuid={} split: read={}ms ({} ops, {}B, {:.1f} "
      "MB/s) | net={}ms ({} window(s), {}B, {:.1f} MB/s) [pipelined]",
      uuid, readNs / 1'000'000, readOps, readBytes, mbps(readBytes, readNs),
      netNs / 1'000'000, windowOps, writeBytes, mbps(writeBytes, netNs));

  if (transferFailed) refreshPeerSegment(segmentName);
  return ok;
}

}  // namespace tt::transport
