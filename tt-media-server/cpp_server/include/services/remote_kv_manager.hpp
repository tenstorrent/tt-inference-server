// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <ctime>
#include <vector>

namespace tt::services {

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

/**
 * One KV-cache block in a Mooncake-store offload / download request.
 *
 * blockHash matches the hash produced by getPrefixCacheHashesByBlocks().
 * positionId is the absolute token position of the block's first token
 * inside the slot. tokenCount covers the "first block is larger" pattern
 * (e.g. 128 then 32, 32, ...) without committing to a fixed layout.
 *
 * Blocks in a request are dependency-chained: index 0 = oldest, last
 * entry = most recent. blocks[i+1].positionId == blocks[i].positionId +
 * blocks[i].tokenCount.
 */
struct KVCacheBlockRef {
  uint64_t blockHash;
  uint32_t positionId;
  uint32_t tokenCount;
};

struct DownloadKVRequest {
  uint32_t dstSlot;
  std::vector<KVCacheBlockRef> blocks;
};

struct OffloadKVRequest {
  uint32_t srcSlot;
  std::vector<KVCacheBlockRef> blocks;
};

enum class KVTransferStatus {
  UNKNOWN,
  IN_PROGRESS,
  COMPLETED,  ///< All workers reported; usablePrefixCount is final.
  FAILED,     ///< Operation could not run to completion; usablePrefixCount=0.
};

/**
 * Result of a download, polled via getDownloadResult().
 *
 * usablePrefixCount: the largest k such that blocks [0..k) were
 * successfully transferred on EVERY worker that participated, and form
 * a contiguous leading prefix on each worker. Per the dependency-chain
 * rule, any block after a hole is unusable.
 *
 * Worked example. 4 workers, 6 blocks requested (A→B→C→D→E→F):
 *   worker 0 got: A B C D E F          per-worker prefix = 6
 *   worker 1 got: A B C D E            per-worker prefix = 5
 *   worker 2 got: A B C   E F          per-worker prefix = 3 (D-hole)
 *   worker 3 got: A B C D E F          per-worker prefix = 6
 *   usablePrefixCount = min(6, 5, 3, 6) = 3
 */
struct KVTransferResult {
  KVTransferStatus status;
  uint32_t usablePrefixCount;
};

/**
 * Async client to the pool of migration workers.
 *
 * Entry points:
 *   migrate()            point-to-point slot→slot copy. Latency-tight
 *                        prefill→decode handoff; no store interaction.
 *   downloadFromStore()  hash-keyed fetch into a slot. Use on prefix
 *                        miss; caller polls getDownloadResult().
 *   offloadToStore()     hash-keyed publish from a slot. Fire-and-forget;
 *                        returns void.
 *
 * Thread-safety: all methods are safe to call from any thread.
 */
class IRemoteKVManager {
 public:
  virtual ~IRemoteKVManager() = default;

  /**
   * Returns immediately with a new unique id; the actual transfer happens
   * asynchronously on a remote worker. Poll with getStatus().
   */
  [[nodiscard]] virtual uint64_t migrate(const MigrationRequest& request) = 0;

  /**
   * Returns MigrationStatus::UNKNOWN if the id was never issued by
   * migrate() or has been garbage-collected.
   */
  virtual MigrationStatus getStatus(uint64_t migrationId) const = 0;

  /**
   * Returns immediately with a fresh id; poll with getDownloadResult().
   * The block list MUST be in dependency order (oldest token position
   * first). Fanned out to every worker; result aggregates as a single
   * usable-prefix count.
   */
  [[nodiscard]] virtual uint64_t downloadFromStore(
      const DownloadKVRequest& request) = 0;

  /**
   * Returns {KVTransferStatus::UNKNOWN, 0} if the id was never issued or
   * has been garbage-collected.
   */
  virtual KVTransferResult getDownloadResult(uint64_t transferId) const = 0;

  /**
   * Fire-and-forget. The impl does not retain per-submission state and
   * there is no getOffloadResult().
   */
  virtual void offloadToStore(const OffloadKVRequest& request) = 0;
};

}  // namespace tt::services
