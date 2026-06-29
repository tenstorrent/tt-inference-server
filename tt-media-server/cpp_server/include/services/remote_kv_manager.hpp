// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <ctime>
#include <vector>

namespace tt::services {

// ---------------------------------------------------------------------------
// Slot-to-slot migration
// ---------------------------------------------------------------------------

enum class MigrationStatus {
  UNKNOWN,
  IN_PROGRESS,
  SUCCESSFUL,
  FAILED,
};

struct Migration {
  uint64_t migration_id;
  std::time_t time_created;
  MigrationStatus status;
};

struct MigrationRequest {
  uint32_t src_slot;
  uint32_t dst_slot;
  uint32_t layer_id;
  uint32_t position_start;
  uint32_t position_end;
};

// ---------------------------------------------------------------------------
// Mooncake-store offload / download
// ---------------------------------------------------------------------------

/**
 * One KV-cache block identified by:
 *   - blockHash:   content-addressable key in the Mooncake store. Matches
 *                  the hash produced by getPrefixCacheHashesByBlocks() so
 *                  download/offload, prefix lookup and routing all agree
 *                  on identity.
 *   - positionId:  absolute token position of this block's FIRST token
 *                  inside the destination (download) or source (offload)
 *                  slot. Because slots are pre-reserved to max context and
 *                  blocks are fixed-size, the worker uses positionId to
 *                  locate the exact byte range to read/write inside the
 *                  slot's KV layout — no allocator round-trip needed.
 *   - tokenCount:  number of tokens this block covers. Lets the schema
 *                  carry the "first block is larger" pattern
 *                  (e.g. tokenCount=128 for block 0, 32 for the rest)
 *                  without committing to a fixed layout on the wire.
 *
 * Blocks in a request are dependency-chained (block i+1 is only meaningful
 * if blocks [0..i] are also present), so the order of the request vector
 * MUST be the same order they occupy in the slot — index 0 = oldest token
 * position, last entry = most recent.
 *
 * positionId and tokenCount are deliberately over-determined: a well-formed
 * chain satisfies `blocks[i+1].positionId == blocks[i].positionId +
 * blocks[i].tokenCount`. The redundancy lets the worker validate the chain
 * in a single pass and lets the wire schema represent non-uniform block
 * sizes in the future without another schema change.
 */
struct KVCacheBlockRef {
  uint64_t blockHash;
  uint32_t positionId;
  uint32_t tokenCount;
};

/**
 * Download the listed blocks from the Mooncake store into dstSlot. The
 * manager fans the request out to every migration worker; each worker
 * fetches only the layers it owns in its pipeline rank, then reports back
 * how far the leading contiguous prefix it could successfully download
 * goes. See KVTransferResult for the aggregation rule.
 */
struct DownloadKVRequest {
  uint32_t dstSlot;
  std::vector<KVCacheBlockRef> blocks;
};

/**
 * Offload the listed blocks from srcSlot into the Mooncake store. The
 * manager fans the request out to every migration worker; each worker
 * publishes only the layers it owns. Offload is fire-and-forget by design
 * — there is no result-poll method, the impl will not track the
 * submission past send.
 */
struct OffloadKVRequest {
  uint32_t srcSlot;
  std::vector<KVCacheBlockRef> blocks;
};

/**
 * Lifecycle of a download transfer. Parallel in spirit to MigrationStatus
 * but kept separate because the "did it work" answer here is not a single
 * bit: a COMPLETED transfer may still have downloaded zero blocks. Read
 * KVTransferResult::usablePrefixCount for the actual outcome.
 */
enum class KVTransferStatus {
  UNKNOWN,      ///< Id was never issued, or has been garbage-collected.
  IN_PROGRESS,  ///< At least one worker has not yet reported back.
  COMPLETED,    ///< All workers reported; usablePrefixCount is final.
  FAILED,       ///< Operation could not run to completion (e.g. timeout,
                ///< worker error). usablePrefixCount is 0.
};

/**
 * Result of a download, polled via getDownloadResult().
 *
 * usablePrefixCount: the largest k such that blocks [0..k) were
 * successfully transferred on EVERY worker that participated, AND those k
 * blocks form a contiguous leading prefix on each worker. Per the
 * dependency-chain rule, only that prefix is safe for the scheduler to
 * consume — any block that arrived later than a hole is unusable because
 * its parent block is missing.
 *
 * Worked example. 4 prefill workers, 6 blocks requested (A→B→C→D→E→F):
 *
 *   worker 0 got  : A B C D E F           per-worker prefix = 6
 *   worker 1 got  : A B C D E             per-worker prefix = 5
 *   worker 2 got  : A B C   E F           per-worker prefix = 3
 *                                         (D missing breaks the chain;
 *                                          E and F are useless for w2)
 *   worker 3 got  : A B C D E F           per-worker prefix = 6
 *   ============================================================
 *   usablePrefixCount = min(6, 5, 3, 6) = 3   →  scheduler reuses A,B,C
 *
 * A returned 0 means nothing was usable from the store; the scheduler
 * must compute from scratch for the entire requested range.
 */
struct KVTransferResult {
  KVTransferStatus status;
  uint32_t usablePrefixCount;
};

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

/**
 * Async client to the pool of migration workers. The scheduler-facing
 * surface for issuing KV-cache migrations and Mooncake-store transfers.
 * Publishes requests on Kafka and tracks completion via per-operation ACK
 * topics.
 *
 * When to use which entry point:
 *
 *   migrate()            slot→slot point-to-point copy between two known
 *                        peers. Use for the prefill→decode handoff in
 *                        latency-tight scenarios where the destination is
 *                        already chosen. No store interaction. Caller
 *                        polls getStatus() until terminal.
 *
 *   downloadFromStore()  hash-keyed fetch from the Mooncake store into a
 *                        local slot. Use to hydrate a slot's prefix when
 *                        local prefix-cache lookup missed and the
 *                        scheduler needs to know how much it can reuse
 *                        before computing the tail. Synchronous in
 *                        spirit: callers are expected to poll
 *                        getDownloadResult() until COMPLETED or FAILED
 *                        and act on usablePrefixCount before scheduling
 *                        compute on the missing tail.
 *
 *   offloadToStore()     hash-keyed publish to the Mooncake store from a
 *                        local slot. Use to persist newly computed blocks
 *                        (write-through during decode) or to save a slot
 *                        before eviction. Fire-and-forget by design: the
 *                        impl does not retain per-submission state and
 *                        there is no result-poll method. The returned id
 *                        is for log correlation only and may be
 *                        discarded.
 *
 * Thread-safety: all methods are safe to call from any thread.
 */
class IRemoteKVManager {
 public:
  virtual ~IRemoteKVManager() = default;

  // -------------------------------------------------------------------------
  // Slot-to-slot migration (existing)
  // -------------------------------------------------------------------------

  /**
   * Migrate KV cache between two local slots. Returns immediately with a
   * new unique id; the actual transfer happens asynchronously on a remote
   * worker. Poll with getStatus().
   */
  [[nodiscard]] virtual uint64_t migrate(const MigrationRequest& request) = 0;

  /**
   * Look up the current status of a previously submitted migration.
   * Returns MigrationStatus::UNKNOWN if the id was never issued by
   * migrate() or has been garbage-collected.
   */
  virtual MigrationStatus getStatus(uint64_t migrationId) const = 0;

  // -------------------------------------------------------------------------
  // Mooncake-store offload / download
  // -------------------------------------------------------------------------

  /**
   * Ask the migration workers to download the listed blocks from the
   * Mooncake store into request.dstSlot. Returns immediately with a fresh
   * id; poll with getDownloadResult().
   *
   * The block list MUST be in dependency order (oldest token position
   * first). The request is fanned out to every worker; each worker
   * handles its own pipeline-rank layers. The aggregated outcome is
   * exposed as a single usable-prefix count — see KVTransferResult.
   */
  [[nodiscard]] virtual uint64_t downloadFromStore(
      const DownloadKVRequest& request) = 0;

  /**
   * Look up the current result of a previously submitted
   * downloadFromStore() call. Returns
   * {KVTransferStatus::UNKNOWN, 0} if the id was never issued or has
   * been garbage-collected.
   */
  virtual KVTransferResult getDownloadResult(uint64_t transferId) const = 0;

  /**
   * Ask the migration workers to offload the listed blocks from
   * request.srcSlot into the Mooncake store. Fire-and-forget: returns a
   * fresh id for log correlation only and the impl does not retain
   * per-submission state. There is no corresponding getOffloadResult()
   * — by design.
   */
  virtual uint64_t offloadToStore(const OffloadKVRequest& request) = 0;
};

}  // namespace tt::services
