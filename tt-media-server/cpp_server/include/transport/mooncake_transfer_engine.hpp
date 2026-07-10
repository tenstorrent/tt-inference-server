// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "transport/i_storage_backend.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief Mooncake-backed Transfer Engine: Mooncake transport + a pluggable
 *        storage backend.
 *
 * Wraps the Mooncake Transfer Engine (`mooncake::TransferEngine`) behind the
 * generic ITransferEngine interface (init → installTransport →
 * registerLocalMemory → openSegment → submit), and composes it with an
 * IStorageBackend so the two mechanisms #3890 calls out — transport (Mooncake
 * TCP/RDMA) and storage (host/device DRAM) — are wired together here. Passing a
 * DeviceDramStorageBackend yields the custom UMD-backed device-DRAM path.
 *
 * The wrapped engine lives behind a pimpl so this header stays free of Mooncake
 * includes; the real type is pulled in only by the .cpp.
 *
 * @note The real Mooncake backend is compiled in when built with
 *       `TT_TRANSPORT_WITH_MOONCAKE` (CMake `-DENABLE_MOONCAKE=ON`); otherwise
 *       the methods log and report failure so transport_lib still builds in
 *       every configuration. See
 * mooncake/poc-transfer-engine/adr-mooncake-backend.md for the
 *       storage/transport split.
 */
class MooncakeTransferEngine : public ITransferEngine {
 public:
  /**
   * @param storage The storage mechanism transfers are staged through. The
   *        device-DRAM backend is the custom backend #3890 targets.
   */
  explicit MooncakeTransferEngine(std::shared_ptr<IStorageBackend> storage);
  ~MooncakeTransferEngine() override;

  MooncakeTransferEngine(const MooncakeTransferEngine&) = delete;
  MooncakeTransferEngine& operator=(const MooncakeTransferEngine&) = delete;
  MooncakeTransferEngine(MooncakeTransferEngine&&) noexcept;
  MooncakeTransferEngine& operator=(MooncakeTransferEngine&&) noexcept;

  StorageMedium storageMedium() const override;
  std::shared_ptr<IStorageBackend> storage() const override;

  /**
   * @brief This engine's actual advertised segment name ("host:port").
   *
   * Under P2PHANDSHAKE the transport binds a *random* port and rewrites its
   * name to use it, so the port passed in EngineConfig::local_server_name is
   * not the one peers must connect to. A peer opens this engine's segment by
   * this returned name. Empty before init() or when built without Mooncake.
   */
  std::string localServerName() const;

  bool init(const EngineConfig& config) override;
  bool registerLocalMemory(void* addr, std::size_t length) override;
  bool unregisterLocalMemory(void* addr) override;
  SegmentHandle openSegment(const std::string& segmentName) override;
  TransferStatus submitAndWait(const TransferRequest& request) override;
  TransferHandle submitBatch(
      const std::vector<TransferRequest>& requests) override;
  TransferStatus waitBatch(TransferHandle handle) override;

 private:
  std::shared_ptr<IStorageBackend> storage_;

  // Hides the underlying mooncake::TransferEngine from this header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::transport
