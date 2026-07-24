// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "transport/kv_cache_layout.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief One side's coordinates: a slot, a layer range, a token-position range.
 *
 * The unit the table walk (buildHostPlan / firstUnresolvedChunk / hostsFor
 * Request) addresses against a single table. Ranges are half-open [begin, end);
 * positions are in tokens, walked in chunk_n_tokens steps (the start is snapped
 * down to a chunk boundary).
 */
struct KvSlice {
  uint32_t slot = 0;
  uint32_t layer_begin = 0;
  uint32_t layer_end = 0;
  uint32_t position_begin = 0;
  uint32_t position_end = 0;
};

/**
 * @brief What to migrate, prefill (src) -> decode (dst).
 *
 * The migration is asymmetric: a layer maps 1:1 (layer L -> layer L, so the
 * layer range is shared), but the *slot* and the *token-position range* may
 * differ between sides — a position shift. The two position ranges must cover
 * the **same number of chunks** (the sender pairs src chunk k with dst chunk k
 * by ordinal, not by absolute position). A symmetric whole-slot migration is
 * src_slot==dst_slot and src positions==dst positions.
 *
 * srcSlice()/dstSlice() project the per-side coordinates the table walk uses:
 * the sender reads with srcSlice() against the prefill table and computes
 * destination device targets with dstSlice() against the decode table; the
 * receiver drains from the sender's per-window descriptors (it holds no table).
 */
struct MigrationRequest {
  uint32_t src_slot = 0;
  uint32_t dst_slot = 0;
  uint32_t layer_begin = 0;
  uint32_t layer_end = 0;
  uint32_t src_position_begin = 0;
  uint32_t src_position_end = 0;
  uint32_t dst_position_begin = 0;
  uint32_t dst_position_end = 0;

  KvSlice srcSlice() const {
    return {src_slot, layer_begin, layer_end, src_position_begin,
            src_position_end};
  }
  KvSlice dstSlice() const {
    return {dst_slot, layer_begin, layer_end, dst_position_begin,
            dst_position_end};
  }
};

/// One device that a chunk lands on (a replica within its device group).
struct ChunkTarget {
  LocalDeviceId device = 0;  ///< encodeDevice(fabric node).
  NocAddr noc_addr = 0;
  uint64_t size_bytes = 0;
};

/// A single logical chunk and the on-host device(s) it maps to (fan-out).
struct ChunkPlan {
  uint32_t slot = 0;
  uint32_t layer = 0;
  uint32_t position = 0;  ///< In tokens.
  std::vector<ChunkTarget> targets;
};

/**
 * @brief A migration plan restricted to one host.
 *
 * `chunks` keeps the logical (slot, layer, position) coordinates and their
 * per-device targets (for the data plane: the sender fills bounce sections, the
 * receiver drains each to its device target). `locations` is the same targets
 * flattened (used for emptiness checks and iteration).
 */
struct HostKvPlan {
  std::vector<ChunkPlan> chunks;
  std::vector<KvChunkLocation> locations;

  bool empty() const { return locations.empty(); }
};

/**
 * @brief Build the migration plan for the chunks of `request` that live on
 *        `host`.
 *
 * For each (layer, position) in range, looks up the chunk, and for every
 * replica in its device group that resides on `host`, emits a target (the
 * device-group fan-out) keyed by encodeDevice. Chunks absent from the table or
 * with no replica on `host` are skipped.
 *
 * Run it on the decode table for the decode host to get the destination device
 * targets; run it on the prefill table for the prefill host to find the source
 * addresses to read.
 */
HostKvPlan buildHostPlan(const IKvTable& table, const std::string& host,
                         const KvSlice& slice);

/// A requested chunk coordinate the table does not hold.
struct MissingChunk {
  uint32_t layer = 0;
  uint32_t position = 0;  ///< In tokens.
};

/**
 * @brief The first requested (layer, position) that `table` does not hold, or
 *        std::nullopt if every requested chunk resolves.
 *
 * Host-independent: it checks table presence (lookup succeeds), not placement,
 * so a chunk owned by another host still resolves. Out-of-range requests
 * (layer_end > num_layers, position_end > max_sequence_length) are covered for
 * free — those coordinates are walked and lookup returns nullopt.
 *
 * Call before buildHostPlan to reject a partially-satisfiable request
 * wholesale: migrating only the found subset would silently leave stale KV at
 * the gaps on decode while still reporting success.
 */
std::optional<MissingChunk> firstUnresolvedChunk(const IKvTable& table,
                                                 const KvSlice& slice);

/// Distinct hosts that hold any chunk of `slice` (decode cluster fan-out).
std::vector<std::string> hostsForRequest(const IKvTable& table,
                                         const KvSlice& slice);

/**
 * @brief Every KvChunkLocation `table` places on `host`, across the *whole*
 *        table (all slots/layers/positions), with device-group fan-out.
 *
 * Used at worker startup to enumerate the host's local devices to open (each
 * distinct `LocalDeviceId` this host owns). (Contrast buildHostPlan, which is
 * the per-request *subset* used to drive reads/writes/drains.)
 */
std::vector<KvChunkLocation> allHostLocations(const IKvTable& table,
                                              const std::string& host);

}  // namespace tt::transport
