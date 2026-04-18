// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
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

constexpr size_t MAX_WORKER_SLOTS = 32;
constexpr size_t WORKER_SCRATCH_U64_COUNT = 32;

/**
 * Names what a slot's scratch area means. Each worker writes its own value
 * during initialize() so the main-side aggregator can dispatch to the
 * correct renderer. Append-only; never renumber existing values.
 */
enum class MetricsLayout : uint8_t {
  UNKNOWN = 0,
  SP_PIPELINE_RUNNER = 1,
  SDXL = 2,
  EMBEDDING = 3,
};

/**
 * RAII owner of the POSIX shared-memory segment backing worker metrics.
 *
 * All access to the segment (metadata reads/writes, per-slot scratch
 * load/store/increment) goes through this class's inline accessors; the
 * underlying bytes are treated as an implementation detail, so callers never
 * hold a raw pointer into shm RAM.
 *
 * Two factories distinguish the two roles of the single binary:
 *   - create(): main process. Allocates the segment, zero-initializes it,
 *     stamps the magic/num_workers header. Destructor munmaps and
 *     shm_unlinks the name (main owns the segment lifecycle).
 *   - open():   worker process. Attaches to the segment main already
 *     created and validates the magic. Destructor only munmaps; the
 *     segment name stays alive for other workers and for main.
 */
class WorkerMetricsShm {
 public:
  /** Main-side factory. Returns nullptr on failure (logged). */
  static std::unique_ptr<WorkerMetricsShm> create(std::string name,
                                                  size_t numWorkers);

  /** Worker-side factory. Returns nullptr on failure (logged). */
  static std::unique_ptr<WorkerMetricsShm> open(std::string name);

  ~WorkerMetricsShm();

  WorkerMetricsShm(const WorkerMetricsShm&) = delete;
  WorkerMetricsShm& operator=(const WorkerMetricsShm&) = delete;

  // ----- metadata accessors -------------------------------------------------

  size_t numWorkers() const {
    return region_->num_workers.load(std::memory_order_acquire);
  }

  // ----- per-slot metadata accessors ---------------------------------------

  void setPid(size_t workerId, int32_t pid) {
    region_->slots[workerId].pid.store(pid, std::memory_order_release);
  }

  void setLayout(size_t workerId, MetricsLayout layout) {
    region_->slots[workerId].metrics_layout.store(static_cast<uint8_t>(layout),
                                                  std::memory_order_release);
  }

  MetricsLayout layout(size_t workerId) const {
    return static_cast<MetricsLayout>(
        region_->slots[workerId].metrics_layout.load(
            std::memory_order_acquire));
  }

  void bumpGeneration(size_t workerId) {
    region_->slots[workerId].generation.fetch_add(1, std::memory_order_acq_rel);
  }

  // ----- per-slot scratch accessors ----------------------------------------

  uint64_t loadScratch(size_t workerId, size_t idx) const {
    return region_->slots[workerId].scratch[idx].load(
        std::memory_order_relaxed);
  }

  void storeScratch(size_t workerId, size_t idx, uint64_t value) {
    region_->slots[workerId].scratch[idx].store(value,
                                                std::memory_order_relaxed);
  }

  /** Returns the previous value. */
  uint64_t fetchAddScratch(size_t workerId, size_t idx, uint64_t delta) {
    return region_->slots[workerId].scratch[idx].fetch_add(
        delta, std::memory_order_relaxed);
  }

  /** Returns the previous value. */
  uint64_t fetchSubScratch(size_t workerId, size_t idx, uint64_t delta) {
    return region_->slots[workerId].scratch[idx].fetch_sub(
        delta, std::memory_order_relaxed);
  }

 private:
  /**
   * Per-worker slot. The dispatch tag (pid, metrics_layout, generation) is
   * the only contract the transport itself enforces. The 32 atomic u64
   * scratch cells are interpreted by the layout-specific writer and
   * renderer that share a layout header (e.g. sp_pipeline_metrics_layout).
   *
   * Slot is cache-line aligned to avoid false sharing across workers.
   */
  struct alignas(64) Slot {
    std::atomic<int32_t> pid;             // 0 = unclaimed
    std::atomic<uint8_t> metrics_layout;  // MetricsLayout
    // 3 bytes of implicit padding follow before `generation` (4-byte aligned).
    std::atomic<uint32_t> generation;  // bumped on slot reclaim
    uint32_t reserved;
    std::atomic<uint64_t> scratch[WORKER_SCRATCH_U64_COUNT];
  };
  static_assert(sizeof(Slot) == 320, "Slot layout drift");

  struct Region {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> num_workers;
    Slot slots[MAX_WORKER_SLOTS];
  };

  WorkerMetricsShm(Region* region, std::string name, bool owns);

  Region* region_{nullptr};
  std::string name_;
  bool owns_{false};  // true = created by main; false = opened by worker
};

}  // namespace tt::worker
