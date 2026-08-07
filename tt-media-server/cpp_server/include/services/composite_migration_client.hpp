// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tt_llm_engine/scheduler/migration_client_interface.hpp>

namespace tt::services {

/**
 * CompositeMigrationClient routes MigrationClientInterface calls across two
 * concrete backends, each of which itself implements MigrationClientInterface:
 *
 *   burst_    — handles the cross-endpoint burst path
 *               (start_burst / enqueue_migration_in_burst / finish_burst).
 *               Typically a RemoteKVManagerAdapter (Kafka) when the
 *               PREFILL_USE_REMOTE_KV_MANAGER path is enabled.
 *
 *   loopback_ — handles the single-shot migrate() path used by
 *               ALLOCATE(migrate_from_slot) prefix-cache slot copies (which
 *               the PrefillScheduler still emits when the input side asks
 *               for a resident-prefix reuse). Typically a
 *               MigrationLayerClientAdapter (shmem) or a MockMigrationClient
 *               — i.e. whatever the non-Kafka path would have picked for
 *               the current runner_type.
 *
 * The routing is a data-structural property: which backend runs a given
 * call is fixed by which member holds the pointer, not by any runtime
 * dispatch inside CompositeMigrationClient. Everything the interface does
 * not naturally partition — poll(), callback registration, backpressure,
 * lifecycle — fans out to both backends so the scheduler sees a single,
 * complete MigrationClient regardless of which backend produced a given
 * event.
 *
 * Why this class exists at all:
 * PrefillScheduler drives two semantically-different migration operations
 * through the SAME interface pointer (ml_). The Kafka RemoteKVManagerAdapter
 * intentionally implements only one of them (burst) and throws on the other
 * (migrate). Wiring only the Kafka adapter into ml_ would therefore crash
 * the process the first time an IS request carries migrate_from_slot. The
 * composite lets you keep RemoteKVManagerAdapter single-purpose while
 * still delivering a fully-functional MigrationClientInterface to the
 * scheduler.
 *
 * Token disjointness (no cross-talk between the two backends):
 *   - burst_ uses caller-supplied MigrationTokens (from start_burst(uuid)).
 *   - loopback_ mints its own tokens internally from migrate().
 * Both are 64-bit; the collision probability is astronomically small.
 * The scheduler's on_migration_complete_ demultiplexes by looking each
 * token up in either its in_flight_alloc_migrations_ map (loopback) or
 * its token_to_slot_ map (burst) — the composite does not need any
 * demux logic of its own.
 *
 * Threading:
 * poll() is invoked by the scheduler's owner thread (ack_reader_thread
 * for prefill). It drains burst_->poll() and loopback_->poll() serially
 * on that thread, so callbacks fire synchronously in-thread — no new
 * synchronization surface is introduced by the composite.
 */
class CompositeMigrationClient final
    : public tt_llm_engine::scheduler::MigrationClientInterface {
 public:
  using MI = tt_llm_engine::scheduler::MigrationClientInterface;
  using BurstId = MI::BurstId;
  using MigrationToken = tt_llm_engine::scheduler::MigrationToken;
  using MigrationCompleteEvent =
      tt_llm_engine::scheduler::MigrationCompleteEvent;
  using MigrationReceivedEvent =
      tt_llm_engine::scheduler::MigrationReceivedEvent;
  using MigrationFailedEvent = tt_llm_engine::scheduler::MigrationFailedEvent;
  using EndpointDisconnectedEvent =
      tt_llm_engine::scheduler::EndpointDisconnectedEvent;

  /**
   * @param burst     Backend for start_burst / enqueue_migration_in_burst /
   *                  finish_burst. Must be non-null.
   * @param loopback  Backend for the single-shot migrate() path. Must be
   *                  non-null — this is what makes the composite exist.
   *                  Callers that don't need loopback support should use
   *                  the burst backend directly instead of this class.
   */
  CompositeMigrationClient(std::unique_ptr<MI> burst,
                           std::unique_ptr<MI> loopback);

  ~CompositeMigrationClient() override = default;

  CompositeMigrationClient(const CompositeMigrationClient&) = delete;
  CompositeMigrationClient& operator=(const CompositeMigrationClient&) = delete;

  // --- Burst path -> burst_ only ---
  BurstId start_burst(MigrationToken uuid) override;

  void enqueue_migration_in_burst(BurstId burst, int remote_endpoint_id,
                                  uint32_t src_slot, uint32_t dst_slot,
                                  uint32_t layer_start,
                                  uint32_t layer_end_exclusive,
                                  uint32_t pos_start,
                                  uint32_t pos_end_exclusive) override;

  MigrationToken finish_burst(BurstId burst) override;

  // Backpressure: report the tighter of the two backends. Kafka reports
  // UINT32_MAX (unbounded), so in the typical wiring this reduces to the
  // loopback's real cmd queue and the scheduler's kCmdQueueBackpressureMargin
  // throttle keeps working exactly as before.
  uint32_t cmd_queue_write_space() const override;

  // --- Single migrate -> loopback_ only ---
  MigrationToken migrate(int remote_endpoint_id, uint32_t src_slot,
                         uint32_t dst_slot, uint32_t layer_start,
                         uint32_t layer_end_exclusive, uint32_t pos_start,
                         uint32_t pos_end_exclusive) override;

  // --- Fan-out ---
  int poll() override;

  void on_migration_complete(
      std::function<void(const MigrationCompleteEvent&)> cb) override;
  void on_migration_received(
      std::function<void(const MigrationReceivedEvent&)> cb) override;
  void on_migration_failed(
      std::function<void(const MigrationFailedEvent&)> cb) override;
  void on_endpoint_disconnected(
      std::function<void(const EndpointDisconnectedEvent&)> cb) override;
  void on_connection_received(std::function<void(int)> cb) override;

  void connect_to(int remote_endpoint_id, const std::string& role,
                  const std::string& service_name) override;
  void wait_ready(int timeout_ms = 5000) override;
  void shutdown(bool drain = true) override;

 private:
  std::unique_ptr<MI> burst_;
  std::unique_ptr<MI> loopback_;
};

}  // namespace tt::services
