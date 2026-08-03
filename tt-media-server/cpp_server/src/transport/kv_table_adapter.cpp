// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_table_adapter.hpp"

#include <algorithm>
#include <optional>
#include <set>

namespace tt::transport {

namespace {

// Walk every (layer, position) the request addresses, invoking `fn(layer, pos)`
// once per chunk — independent of whether the table actually holds it. The
// single source of truth for the request's chunk-stepping, so the planning walk
// (forEachReplica) and the presence check (firstUnresolvedChunk) cover the
// exact same set of chunks and can never drift.
template <typename Fn>
void forEachRequestedChunk(const IKvTable& table, const KvSlice& slice,
                           Fn&& fn) {
  const uint32_t step = table.config().chunk_n_tokens;
  if (step == 0) return;
  // Snap the start down to a chunk boundary so token ranges that don't begin on
  // a chunk edge still address whole chunks.
  const uint32_t start = slice.position_begin - (slice.position_begin % step);
  for (uint32_t layer = slice.layer_begin; layer < slice.layer_end; ++layer) {
    for (uint32_t pos = start; pos < slice.position_end; pos += step) {
      fn(layer, pos);
    }
  }
}

// Walk the request, invoking `fn(layer, position, loc, node)` for every
// device-group replica of every chunk the table actually holds. Chunks absent
// from the table are skipped here — callers that must not tolerate gaps gate on
// firstUnresolvedChunk first.
template <typename Fn>
void forEachReplica(const IKvTable& table, const KvSlice& slice, Fn&& fn) {
  forEachRequestedChunk(table, slice, [&](uint32_t layer, uint32_t pos) {
    const std::optional<ChunkLoc> loc = table.lookup(slice.slot, layer, pos);
    if (!loc) return;
    for (const FabricNode& node : table.deviceGroup(loc->device_group_index)) {
      fn(layer, pos, *loc, node);
    }
  });
}

}  // namespace

HostKvPlan buildHostPlan(const IKvTable& table, const std::string& host,
                         const KvSlice& slice) {
  HostKvPlan plan;

  // Group replicas by their logical (layer, position) so each ChunkPlan carries
  // all of this host's targets for that chunk (the fan-out width). Replicas of
  // a chunk arrive adjacent (layer outer, position inner, replicas innermost),
  // so a running (layer, position) is enough to detect chunk boundaries.
  bool haveCurrent = false;
  uint32_t curLayer = 0;
  uint32_t curPos = 0;

  forEachReplica(table, slice,
                 [&](uint32_t layer, uint32_t pos, const ChunkLoc& loc,
                     const FabricNode& node) {
                   if (table.hostOf(node) != host) return;

                   const ChunkTarget target{encodeDevice(node), loc.noc_addr,
                                            loc.size_bytes};
                   plan.locations.push_back(KvChunkLocation{
                       target.device, target.noc_addr, target.size_bytes});

                   if (!haveCurrent || curLayer != layer || curPos != pos) {
                     ChunkPlan cp;
                     cp.slot = slice.slot;
                     cp.layer = layer;
                     cp.position = pos;
                     plan.chunks.push_back(std::move(cp));
                     haveCurrent = true;
                     curLayer = layer;
                     curPos = pos;
                   }
                   plan.chunks.back().targets.push_back(target);
                 });

  return plan;
}

std::optional<MissingChunk> firstUnresolvedChunk(const IKvTable& table,
                                                 const KvSlice& slice) {
  std::optional<MissingChunk> missing;
  forEachRequestedChunk(table, slice, [&](uint32_t layer, uint32_t pos) {
    if (missing) return;  // Already found one; report the first.
    if (!table.lookup(slice.slot, layer, pos)) {
      missing = MissingChunk{layer, pos};
    }
  });
  return missing;
}

std::vector<std::string> hostsForRequest(const IKvTable& table,
                                         const KvSlice& slice) {
  std::set<std::string> hosts;
  forEachReplica(
      table, slice,
      [&](uint32_t, uint32_t, const ChunkLoc&, const FabricNode& node) {
        const std::string& host = table.hostOf(node);
        if (!host.empty()) hosts.insert(host);
      });
  return {hosts.begin(), hosts.end()};
}

std::vector<KvChunkLocation> allHostLocations(const IKvTable& table,
                                              const std::string& host) {
  // Walk the whole table (every slot/layer/position the config can hold), not a
  // single request's range, so the layout — and thus the registered mirror
  // segment — is the same for every migration. Mirrors forEachReplica's
  // ordering so the packing is byte-identical to the per-request plan's.
  const KvTableConfig& c = table.config();
  std::vector<KvChunkLocation> out;
  if (c.chunk_n_tokens == 0) return out;
  for (uint32_t slot = 0; slot < c.num_slots; ++slot) {
    for (uint32_t layer = 0; layer < c.num_layers; ++layer) {
      for (uint32_t pos = 0; pos < c.max_sequence_length;
           pos += c.chunk_n_tokens) {
        const std::optional<ChunkLoc> loc = table.lookup(slot, layer, pos);
        if (!loc) continue;
        for (const FabricNode& node :
             table.deviceGroup(loc->device_group_index)) {
          if (table.hostOf(node) != host) continue;
          out.push_back(KvChunkLocation{encodeDevice(node), loc->noc_addr,
                                        loc->size_bytes});
        }
      }
    }
  }
  return out;
}

}  // namespace tt::transport
