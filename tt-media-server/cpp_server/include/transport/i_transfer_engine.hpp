// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "transport/i_storage_backend.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief The Transfer Engine: moves tensor bytes between galaxies (issue
 *        #3890).
 *
 * Per #3890's core assumption, a Transfer Engine defines two mechanisms:
 *   - a *transport* mechanism (TCP, RDMA, ...) — how bytes move between
 *     registered host staging buffers on different hosts, modelled by this
 *     interface (init → registerLocalMemory → openSegment → submitAndWait),
 *     mirroring the Mooncake Transfer Engine surface; and
 *   - a *storage* mechanism (host DRAM, device DRAM, ...) — how bytes are
 *     staged to/from the backing store, modelled by IStorageBackend, with the
 *     UMD device-DRAM backend being the custom backend #3890 targets.
 *
 * Memory model: regions are registered, and transfers are addressed, by *host*
 * virtual address. The transport cannot DMA directly from TT device DRAM, so
 * the storage backend stages device DRAM into the registered host buffer
 * (see mooncake/poc-transfer-engine/adr-mooncake-backend.md). Implemented by
 * MooncakeTransferEngine.
 */
class ITransferEngine {
 public:
  virtual ~ITransferEngine() = default;

  /// The storage mechanism this engine stages transfers through.
  virtual StorageMedium storageMedium() const = 0;

  /**
   * @brief The storage backend transfers are staged through.
   *
   * Exposed so a driver (e.g. MooncakeMigrationWorker) can run the
   * bounce-buffer flow — stage device DRAM into the registered host buffer with
   * the backend, then move the host buffer with this engine. May be null.
   */
  virtual std::shared_ptr<IStorageBackend> storage() const = 0;

  /**
   * @brief Bring up the engine and install its transport.
   * @return true on success.
   */
  virtual bool init(const EngineConfig& config) = 0;

  /**
   * @brief Register a local host buffer so it can be the source/target of a
   *        transfer. The buffer must outlive its registration.
   * @return true on success.
   */
  virtual bool registerLocalMemory(void* addr, std::size_t length) = 0;

  /**
   * @brief Stop advertising a previously registered local buffer.
   * @return true on success.
   */
  virtual bool unregisterLocalMemory(void* addr) = 0;

  /**
   * @brief Resolve a peer's advertised segment by name.
   * @return a handle, or K_INVALID_SEGMENT on failure.
   */
  virtual SegmentHandle openSegment(const std::string& segmentName) = 0;

  /**
   * @brief Submit a single transfer and block until it completes or fails.
   *
   * Convenience over the batched Mooncake API for the PoC's one-tensor path.
   */
  virtual TransferStatus submitAndWait(const TransferRequest& request) = 0;

  /**
   * @brief Submit many transfers as one batch and return immediately.
   *
   * The requests are issued inside a single underlying batch (one
   * allocateBatchID / submitTransfer) and run concurrently on the engine's own
   * threads; this call does NOT block. The caller can do other work — e.g.
   * stage the next window from device DRAM into a second buffer — and later
   * block on the returned handle with waitBatch(). That read/transfer overlap
   * is the point of the async pair. Every request's `local_addr` must stay
   * registered and unmodified until the matching waitBatch() returns.
   *
   * @return a valid handle to await, or an invalid handle (valid==false) if the
   *         batch could not be dispatched (bad request / not initialized).
   */
  virtual TransferHandle submitBatch(
      const std::vector<TransferRequest>& requests) = 0;

  /**
   * @brief Block until a submitBatch() batch finishes, then release it.
   *
   * @return COMPLETED with the summed transferred_bytes iff *every* request in
   *         the batch completed; FAILED otherwise (including an invalid
   * handle).
   */
  virtual TransferStatus waitBatch(TransferHandle handle) = 0;

  /**
   * @brief Submit a batch and block until it completes. Convenience over
   *        submitBatch + waitBatch for callers that do not overlap with other
   *        work. An empty request list is COMPLETED with 0 bytes.
   */
  virtual TransferStatus submitBatchAndWait(
      const std::vector<TransferRequest>& requests) {
    if (requests.empty()) return TransferStatus{TransferState::COMPLETED, 0};
    const TransferHandle handle = submitBatch(requests);
    if (!handle.valid) return TransferStatus{TransferState::FAILED, 0};
    return waitBatch(handle);
  }
};

}  // namespace tt::transport
