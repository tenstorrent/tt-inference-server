// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
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
   * @return a handle, or K_INVALID_SEGMENT on failure.
   */
  virtual SegmentHandle openSegment(const std::string& segmentName) = 0;

  /**
   * @brief Force-refresh a peer's segment descriptor from the metadata service
   *        and return its (possibly new) handle.
   *
   * After a peer restarts on a fresh dynamic port it re-publishes under the
   * same logical name with a new address. RDMA force-updates the cached
   * descriptor on its own retry path, but TCP reads a stale cached descriptor
   * and would keep targeting the dead address. Senders call this after a
   * transfer fails to pick up the peer's current address before retrying.
   * @return a usable handle, or K_INVALID_SEGMENT if the peer is unresolvable.
   */
  virtual SegmentHandle refreshSegment(const std::string& segmentName) = 0;

  /**
   * @brief Resolve a peer's routable host from the metadata service, given its
   *        server name.
   *
   * The metadata service stores each worker's routable address in its rpc_meta
   * registry, keyed by the worker's server name (which it published via
   * MC_TCP_BIND_ADDRESS). This surfaces that host so a caller can discover a
   * peer at bring-up instead of hard-coding it — e.g. the prefill worker
   * resolving where each decode host lives before opening its control channel.
   * With a metadata server @p serverName may be a LOGICAL tag (e.g.
   * "decode-0"); under P2PHANDSHAKE Mooncake parses host:port from the name.
   * The returned host is where the peer's control server also lives (same
   * node); the peer's Mooncake rpc_port is deliberately dropped since the
   * caller pairs the host with the separate KV control port.
   *
   * @return the peer's host (IP or hostname), or empty string if unresolvable.
   *         The base implementation returns empty so engines without a metadata
   *         service (test fakes) need not override it.
   */
  virtual std::string resolveServerName(const std::string& /*serverName*/) {
    return {};
  }

  /**
   * @brief Publish an arbitrary fact about this worker into the metadata
   *        service so peers can discover it — the *same* store openSegment /
   *        resolveServerName read from.
   *
   * Mooncake's segment and rpc_meta registries only carry the data plane (a
   * peer's segment + its Mooncake rpc host:port). Anything else a peer must
   * learn at bring-up — e.g. a worker's KV *control* endpoint — has nowhere to
   * live otherwise, forcing a hard-coded convention. This routes those facts
   * through the metadata service too, so discovery stays the single source of
   * truth. Keys are raw; the caller namespaces them (e.g.
   * "kv_control/decode-0").
   *
   * @return true on success. The base implementation is a no-op returning false
   *         so engines without a metadata service (P2PHANDSHAKE / test fakes)
   *         need not override it; the caller then falls back to a static
   *         convention.
   */
  virtual bool publishMetadata(const std::string& /*key*/,
                               const std::string& /*value*/) {
    return false;
  }

  /**
   * @brief Look up a value previously stored with publishMetadata.
   * @return the value, or std::nullopt if the key is absent / unresolvable /
   *         there is no metadata service. The base implementation returns
   *         std::nullopt.
   */
  virtual std::optional<std::string> lookupMetadata(
      const std::string& /*key*/) {
    return std::nullopt;
  }

  /**
   * @brief Submit a single transfer and block until it completes or fails.
   *
   * Convenience over the batched Mooncake API for the PoC's one-tensor path.
   */
  virtual TransferStatus submitAndWait(const TransferRequest& request) = 0;
};

}  // namespace tt::transport
