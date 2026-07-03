// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>
#include <unordered_map>
#include <unordered_set>

#include "services/remote_kv_manager.hpp"

namespace tt::services {

/**
 * Adapter that implements MigrationClientInterface using IRemoteKVManager.
 *
 * This allows PrefillScheduler to use our Kafka-backed RemoteKVManager for
 * both KV cache migration entry points:
 *
 *   1) migrate()            — ALLOCATE prefix-cache loopback (intra-endpoint
 *                             slot->slot KV pre-warm).
 *   2) start_burst() /
 *      enqueue_migration_in_burst() /
 *      finish_burst()       — cross-endpoint Prefill->Decode KV migration on
 *                             a migrating SUBMIT (61 layers x N chunks all
 *                             aggregated under one burst uuid).
 *
 * Aggregation (master-worker parity):
 * Every scheduler-facing "unit of work" (one migrate() call, or one start_burst
 * .. finish_burst pair) fans out to N per-layer requests on IRemoteKVManager
 * (one Kafka request per layer per position range). The N per-layer Kafka acks
 * are aggregated inside poll() and delivered to the scheduler as EXACTLY ONE
 * terminal callback (onComplete_ or onFailed_) per group token. This mirrors
 * the "master migration worker" semantics: multiple sub-completions collapse
 * into a single on_migration_complete / on_migration_failed event.
 *
 * Group token:
 * - migrate() mints an adapter-owned token (high bit set to keep its space
 *   disjoint from caller-supplied uuids).
 * - start_burst(uuid) uses the caller-supplied uuid directly as both the
 *   BurstId and the terminal event's token — finish_burst returns that same
 *   uuid so the scheduler can correlate the eventual on_migration_complete
 *   with its own token_to_slot_ map.
 *
 * Completion ordering (why finish_burst DOES NOT fire callbacks):
 * PrefillScheduler emplaces the token returned by finish_burst() into its
 * token_to_slot_ map AFTER the finish_burst() call returns; on_migration_
 * complete_ looks that token up. Firing the callback synchronously inside
 * finish_burst() would race that emplace and silently drop the burst. The
 * adapter therefore only fires the terminal event inside poll(), and only
 * when a group is both `closed` AND has an empty pending set.
 *
 * Failure policy: fail-fast. The first per-layer FAILED status fires onFailed_
 * immediately (once per token). Remaining Kafka acks for that group are still
 * drained so the reverse map does not leak; no further callback fires.
 *
 * Threading:
 * - poll() should be called from the scheduler's ack_reader_thread.
 * - migrate() / start_burst() / enqueue_migration_in_burst() / finish_burst()
 *   are thread-safe but expected to be driven from the same owner thread as
 *   poll() (that is the MigrationClientInterface contract).
 * - shutdown(drain=true) must be called from a thread OTHER than the poll()
 *   owner. The caller must ensure poll() continues to run on its owner thread
 *   until shutdown returns, otherwise the drain will time out.
 *
 * Callback registration: Register all callbacks during setup, before the first
 * migrate() / start_burst() / poll() call. Callbacks are read without
 * synchronization from poll() on the owner thread.
 */
class RemoteKVManagerAdapter
    : public tt_llm_engine::scheduler::MigrationClientInterface {
 public:
  using MigrationToken = tt_llm_engine::scheduler::MigrationToken;
  using MigrationCompleteEvent =
      tt_llm_engine::scheduler::MigrationCompleteEvent;
  using MigrationReceivedEvent =
      tt_llm_engine::scheduler::MigrationReceivedEvent;
  using MigrationFailedEvent = tt_llm_engine::scheduler::MigrationFailedEvent;
  using EndpointDisconnectedEvent =
      tt_llm_engine::scheduler::EndpointDisconnectedEvent;

  /**
   * @param kvManager The underlying KV manager for migrations.
   * @param shutdownTimeout Max time shutdown(drain=true) waits for in-flight
   *   migrations to complete. Default 30s.
   */
  explicit RemoteKVManagerAdapter(
      std::unique_ptr<IRemoteKVManager> kvManager,
      std::chrono::milliseconds shutdownTimeout = std::chrono::seconds(30));

  ~RemoteKVManagerAdapter() override = default;

  RemoteKVManagerAdapter(const RemoteKVManagerAdapter&) = delete;
  RemoteKVManagerAdapter& operator=(const RemoteKVManagerAdapter&) = delete;

  // --- Burst lifecycle (cross-endpoint P->D migration on migrating SUBMIT) ---
  // BurstId == the caller-supplied uuid. Fan-out and completion aggregation
  // use the same MigrationGroup machinery as migrate(); poll() only emits the
  // terminal event after finish_burst() has closed the group AND all fanned
  // out per-layer Kafka acks have landed.
  BurstId start_burst(MigrationToken uuid) override;

  void enqueue_migration_in_burst(BurstId burst, int remote_endpoint_id,
                                  uint32_t src_slot, uint32_t dst_slot,
                                  uint32_t layer_start,
                                  uint32_t layer_end_exclusive,
                                  uint32_t pos_start,
                                  uint32_t pos_end_exclusive) override;

  MigrationToken finish_burst(BurstId burst) override;

  // Kafka's producer has no fixed-size cmd queue in the sense the shmem
  // adapter has; return UINT32_MAX so PrefillScheduler's ack-drain never
  // throttles the burst enqueue path on backpressure. If the Kafka producer
  // does hit its own send buffer, MigrationRequestMessage will fail-fast via
  // RemoteKVManagerImpl and appear as an immediate FAILED status on poll().
  uint32_t cmd_queue_write_space() const override { return UINT32_MAX; }

  // --- Single migration (ALLOCATE prefix-cache loopback pre-warm) ---
  // One-shot: the returned group is `closed` from the start, so poll() fires
  // the terminal event as soon as all per-layer Kafka acks have landed.
  MigrationToken migrate(int remote_endpoint_id, uint32_t src_slot,
                         uint32_t dst_slot, uint32_t layer_start,
                         uint32_t layer_end_exclusive, uint32_t pos_start,
                         uint32_t pos_end_exclusive) override;

  // --- Polling ---
  int poll() override;

  void on_migration_complete(
      std::function<void(const MigrationCompleteEvent&)> cb) override;

  void on_migration_received(
      std::function<void(const MigrationReceivedEvent&)> cb) override;

  void on_migration_failed(
      std::function<void(const MigrationFailedEvent&)> cb) override;

  void on_endpoint_disconnected(
      std::function<void(const EndpointDisconnectedEvent&)> cb) override;

  void on_connection_received(
      std::function<void(int remote_endpoint_id)> cb) override;

  // --- Setup / lifecycle ---
  void connect_to(int remote_endpoint_id, const std::string& role,
                  const std::string& service_name) override;

  void wait_ready(int timeout_ms = 5000) override;

  void shutdown(bool drain = true) override;

 private:
  // Aggregation state for ONE scheduler-facing unit of work — either a single
  // migrate() call OR a start_burst()..finish_burst() lifecycle. Owns the set
  // of per-layer Kafka migration_ids that must all terminate before poll()
  // fires the group's single MigrationComplete callback.
  //
  // `closed` gates completion emission:
  //   - migrate(): closed=true at construction (one-shot, no further fan-out).
  //   - start_burst() / enqueue_.. / finish_burst(): closed=false until
  //     finish_burst() flips it, so enqueue calls can keep adding pending ids
  //     without a premature "done" firing.
  //
  // `failedReported` guards against a second callback: once a per-layer FAILED
  // is observed we call onFailed_ and never fire again for this token, but
  // keep draining remaining Kafka acks so kafkaToGroup_ does not leak.
  struct MigrationGroup {
    MigrationToken token;
    std::unordered_set<uint64_t> pendingKafkaIds;
    bool closed{false};
    bool failed{false};
    bool failedReported{false};
  };

  // Token minter for adapter-owned MigrationTokens. Bit 63 is set to keep the
  // adapter's token space disjoint from any caller-supplied uuid space (same
  // convention as MigrationLayerClientAdapter::migrate).
  MigrationToken mintToken();

  std::unique_ptr<IRemoteKVManager> kvManager_;

  mutable std::mutex mtx_;
  std::condition_variable drainCv_;

  // Group state keyed by the adapter-minted MigrationToken returned to the
  // scheduler; poll() aggregates into these.
  std::unordered_map<MigrationToken, MigrationGroup> groups_;
  // Reverse index: Kafka migration_id -> owning group's token. Rebuilt on
  // every migrate(); trimmed as per-layer acks land in poll().
  std::unordered_map<uint64_t, MigrationToken> kafkaToGroup_;

  std::atomic<uint64_t> nextTokenSuffix_{0};
  bool shutdownRequested_{false};

  std::chrono::milliseconds shutdownTimeout_;

  std::function<void(const MigrationCompleteEvent&)> onComplete_;
  std::function<void(const MigrationReceivedEvent&)> onReceived_;
  std::function<void(const MigrationFailedEvent&)> onFailed_;
  std::function<void(const EndpointDisconnectedEvent&)> onDisconnected_;
  std::function<void(int)> onConnectionReceived_;
};

}  // namespace tt::services
