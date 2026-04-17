// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace tt::worker {

/**
 * POSIX shared-memory transport for worker metrics.
 *
 * Workers publish operational signals into a fixed-size segment created by
 * the main process at startup. Each worker owns one cache-line-aligned slot;
 * the main process aggregates all slots at scrape time.
 *
 * The slot intentionally exposes only a minimal dispatch tag (pid +
 * metrics_layout + generation). The remaining 32-u64 scratch area is
 * interpreted entirely by the layout, so different worker types (LLM, SDXL,
 * embedding, ...) can publish unrelated signals through the same transport
 * without negotiating a common header.
 */

constexpr uint32_t WORKER_METRICS_SHM_MAGIC = 0x54545744;  // "TTWD"
constexpr size_t MAX_WORKER_SLOTS = 32;
constexpr size_t WORKER_SCRATCH_U64_COUNT = 32;

/**
 * Names what a slot's scratch area means. Each worker writes its own value
 * during initialize() so the main-side aggregator can dispatch to the
 * correct renderer. Append-only; never renumber existing values.
 */
enum class MetricsLayout : uint32_t {
  UNKNOWN = 0,
  SP_PIPELINE_RUNNER = 1,
  SDXL = 2,
  EMBEDDING = 3,
};

/**
 * Per-worker slot. The dispatch tag (pid, metrics_layout, generation) is
 * the only contract the transport itself enforces. The 32 atomic u64 scratch
 * cells are interpreted by the layout-specific writer (worker side) and
 * renderer (main side) sharing a layout header (e.g.
 * sp_pipeline_metrics_layout.hpp).
 *
 * Slot is cache-line aligned to avoid false sharing across workers.
 */
struct alignas(64) WorkerSlot {
  std::atomic<int32_t> pid;             // 0 = unclaimed
  std::atomic<uint32_t> metrics_layout;  // MetricsLayout
  std::atomic<uint32_t> generation;      // bumped on slot reclaim
  uint32_t reserved;
  std::atomic<uint64_t> scratch[WORKER_SCRATCH_U64_COUNT];
};
static_assert(sizeof(WorkerSlot) == 320, "WorkerSlot layout drift");

struct WorkerMetricsShmRegion {
  std::atomic<uint32_t> magic;
  std::atomic<uint32_t> num_workers;
  WorkerSlot slots[MAX_WORKER_SLOTS];
};

/**
 * Create the shared-memory region (main process only).
 *
 * Defensively unlinks any stale segment of the same name first, then
 * shm_open(O_CREAT|O_RDWR), ftruncate to sizeof(WorkerMetricsShmRegion),
 * mmap, and initialize the header. Returns nullptr on failure (logged).
 */
WorkerMetricsShmRegion* createSharedRegion(const std::string& name,
                                           size_t numWorkers);

/**
 * Attach to an existing region (worker process). Validates magic
 * and returns nullptr (with a log line) on mismatch.
 */
WorkerMetricsShmRegion* openSharedRegion(const std::string& name);

/**
 * Tear down the region created by createSharedRegion (main process only).
 * munmap + shm_unlink. Safe to call with a nullptr region.
 */
void destroySharedRegion(WorkerMetricsShmRegion* region,
                         const std::string& name);

}  // namespace tt::worker
