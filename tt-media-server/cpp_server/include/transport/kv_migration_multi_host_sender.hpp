// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "transport/double_pinned_buffer.hpp"
#include "transport/i_device_io.hpp"
#include "transport/i_transfer_engine.hpp"
#include "transport/kv_control_channel.hpp"
#include "transport/kv_staging_pool.hpp"
#include "transport/kv_table_adapter.hpp"
#include "transport/kv_table_view.hpp"
#include "transport/mooncake_kv_sender.hpp"

namespace tt::transport {

class WorkerHealth;

/**
 * @brief Sender-side fan-out across the N decode hosts a slot spans, over the
 *        RDMA-over-host bounce-buffer protocol (n->m).
 *
 * Routing (which decode hosts a request touches, from hostsForRequest on the
 * shared decode table) and host->control-channel resolution are injected; each
 * per-host leg runs the windowed bounce protocol (MooncakeKvSender +
 * KvMigrationSender: Begin -> BounceReady -> [WindowReady/WindowAck]* -> Done
 * -> Ack). Each decode host is its own bounce-buffer receiver process with its
 * own bounce buffer + segment + control channel.
 *
 * A single staging pool is shared across all per-host senders (the fan-out is
 * serial). When `deviceMap` is set (DRISC), the staging is double-pinned (the
 * same buffer the engine ibv_reg_mr's is NOC-mapped).
 */
class KvMigrationMultiHostSender {
 public:
  KvMigrationMultiHostSender(
      std::shared_ptr<ITransferEngine> engine, IDeviceIo& device,
      std::shared_ptr<const IKvTable> prefillTable,
      std::shared_ptr<const IKvTable> decodeTable, std::string prefillHost,
      std::unordered_map<std::string, std::shared_ptr<KvControlChannel>>
          channels,
      DeviceMapFn deviceMap = {}, WorkerHealth* health = nullptr);

  /// Upsert a decode host and bind/refresh its control channel (thread-safe vs
  /// migrate()). @return true if present after the call; false if @p channel is
  /// null.
  bool addHost(const std::string& host,
               std::shared_ptr<KvControlChannel> channel);

  /// Drive the bounce migration to every decode host the request touches
  /// (sorted order). A missing/failed host fails the whole call, but the rest
  /// are still attempted (comprehensive report). @return true iff every
  /// involved host completed (all-or-nothing retry contract: re-drive the same
  /// request).
  bool migrate(uint64_t uuid, const MigrationRequest& request);

  std::size_t hostCount() const;

  /// True iff the shared staging pool registered with the engine (and, on
  /// DRISC, NOC-mapped). False means every migration will fail at runtime, so
  /// the worker must fail bring-up rather than advertise Ready — mirrors the
  /// receiver's registered() on the decode side.
  bool registered() const { return staging_ && staging_->registered(); }

 private:
  std::shared_ptr<ITransferEngine> engine_;
  IDeviceIo& device_;
  std::shared_ptr<const IKvTable> prefill_table_;
  std::shared_ptr<const IKvTable> decode_table_;
  std::string prefill_host_;
  WorkerHealth* health_ = nullptr;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<KvControlChannel>> channels_;
  // Shared staging pool (declared before senders_ so it outlives them).
  std::shared_ptr<KvStagingPool> staging_;
  std::unordered_map<std::string, std::unique_ptr<MooncakeKvSender>> senders_;
};

}  // namespace tt::transport
