// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

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
   * @return a handle, or kInvalidSegment on failure.
   */
  virtual SegmentHandle openSegment(const std::string& segmentName) = 0;

  /**
   * @brief Submit a single transfer and block until it completes or fails.
   *
   * Convenience over the batched Mooncake API for the PoC's one-tensor path.
   */
  virtual TransferStatus submitAndWait(const TransferRequest& request) = 0;
};

}  // namespace tt::transport
