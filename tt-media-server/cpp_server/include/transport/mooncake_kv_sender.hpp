// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>

#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_cache_layout.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"

namespace tt::transport {

/**
 * @brief Prefill-host (sender) side of a KV migration.
 *
 * Drives the sender half of the data plane. For each chunk of the slot it:
 *   1. reads the bytes from its local (prefill) device DRAM (one source
 *      replica), and
 *   2. computes the full destination addressing from the decode table it
 *      received at init — the mirror offset for each decode replica — and
 * pushes the bytes there with a one-sided Mooncake WRITE (the fan-out).
 *
 * Because the sender owns all destination addressing (mirror offset today, the
 * decode device NocAddr already known for future RDMA-direct), the receiver
 * only drains its mirror — see MooncakeKvReceiver.
 *
 * Holds both tables: the prefill table (local, for reads) and the decode table
 * (remote, exchanged at init, for destination offsets).
 */
class MooncakeKvSender {
 public:
  MooncakeKvSender(std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
                   std::shared_ptr<const IKvTable> prefillTable,
                   std::shared_ptr<const IKvTable> decodeTable,
                   std::string prefillHost, std::string decodeHost);

  /**
   * @brief Transfer one slot's chunks into the decode mirror segment.
   * @param request      what to migrate (slot, layer/position ranges).
   * @param segment_name the receiver's advertised segment (from MirrorReady).
   * @return true if every chunk read + wrote successfully.
   */
  bool transferSlot(const MigrationRequest& request,
                    const std::string& segmentName);

 private:
  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  std::shared_ptr<const IKvTable> prefill_table_;
  std::shared_ptr<const IKvTable> decode_table_;
  std::string prefill_host_;
  std::string decode_host_;
  // Destination addressing built once from the *full* decode table, so the
  // mirror offsets are byte-identical to the receiver's full-table mirror and
  // stable across migrations (see MooncakeKvReceiver / allHostLocations).
  KvCacheLayout dst_layout_;
};

}  // namespace tt::transport
