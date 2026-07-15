// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/composite_migration_client.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "utils/logger.hpp"

namespace tt::services {

CompositeMigrationClient::CompositeMigrationClient(std::unique_ptr<MI> burst,
                                                   std::unique_ptr<MI> loopback)
    : burst_(std::move(burst)), loopback_(std::move(loopback)) {
  // Both slots are required by construction. A caller who only needs one
  // backend should hand that backend directly to whatever holds the
  // MigrationClientInterface pointer — using the composite in that case
  // would just add indirection with no routing benefit, and would open
  // the door to null-deref crashes on the "missing" method group.
  if (!burst_) {
    throw std::invalid_argument(
        "[CompositeMigrationClient] burst backend must be non-null "
        "(handles start_burst / enqueue_migration_in_burst / finish_burst)");
  }
  if (!loopback_) {
    throw std::invalid_argument(
        "[CompositeMigrationClient] loopback backend must be non-null "
        "(handles migrate() for ALLOCATE(migrate_from_slot) prefix-cache "
        "slot copies)");
  }
  TT_LOG_INFO(
      "[CompositeMigrationClient] wired: burst + loopback backends both "
      "installed; migrate() routes to loopback, start_burst/enqueue/finish "
      "route to burst");
}

// --- Burst path: forward verbatim to burst_ ---

CompositeMigrationClient::BurstId CompositeMigrationClient::start_burst(
    MigrationToken uuid) {
  return burst_->start_burst(uuid);
}

void CompositeMigrationClient::enqueue_migration_in_burst(
    BurstId burst, int remote_endpoint_id, uint32_t src_slot, uint32_t dst_slot,
    uint32_t layer_start, uint32_t layer_end_exclusive, uint32_t pos_start,
    uint32_t pos_end_exclusive) {
  burst_->enqueue_migration_in_burst(burst, remote_endpoint_id, src_slot,
                                     dst_slot, layer_start, layer_end_exclusive,
                                     pos_start, pos_end_exclusive);
}

CompositeMigrationClient::MigrationToken CompositeMigrationClient::finish_burst(
    BurstId burst) {
  return burst_->finish_burst(burst);
}

// --- Single migrate: forward verbatim to loopback_ ---

CompositeMigrationClient::MigrationToken CompositeMigrationClient::migrate(
    int remote_endpoint_id, uint32_t src_slot, uint32_t dst_slot,
    uint32_t layer_start, uint32_t layer_end_exclusive, uint32_t pos_start,
    uint32_t pos_end_exclusive) {
  return loopback_->migrate(remote_endpoint_id, src_slot, dst_slot, layer_start,
                            layer_end_exclusive, pos_start, pos_end_exclusive);
}

// --- Backpressure: min so the scheduler respects the tighter constraint ---

uint32_t CompositeMigrationClient::cmd_queue_write_space() const {
  // In the typical Kafka+shmem wiring: burst_ (Kafka) returns UINT32_MAX,
  // so min() collapses to loopback_->cmd_queue_write_space(). That is what
  // the PrefillScheduler's kCmdQueueBackpressureMargin throttle needs to
  // see so it defers enqueuing when the shmem cmd queue is almost full.
  return std::min(burst_->cmd_queue_write_space(),
                  loopback_->cmd_queue_write_space());
}

// --- Fan-out: poll both; sum returned so the scheduler still observes
//     "some work happened this tick" and does not DS_PAUSE prematurely
//     when only one backend actually made progress this iteration.

int CompositeMigrationClient::poll() {
  const int a = burst_->poll();
  const int b = loopback_->poll();
  return a + b;
}

// --- Callback registration: mirror the SAME callback onto both backends.
//     Each backend fires with its own tokens; the scheduler's registered
//     handler receives all events through one std::function and demuxes
//     via its own token maps (in_flight_alloc_migrations_ for loopback
//     tokens, token_to_slot_ for burst tokens). The composite therefore
//     does not need to know or care which backend a given event came from.
//     We copy the callback into burst_'s slot and move it into loopback_'s
//     slot so we do not perform a second heap allocation for the same
//     std::function target.

void CompositeMigrationClient::on_migration_complete(
    std::function<void(const MigrationCompleteEvent&)> cb) {
  burst_->on_migration_complete(cb);
  loopback_->on_migration_complete(std::move(cb));
}

void CompositeMigrationClient::on_migration_received(
    std::function<void(const MigrationReceivedEvent&)> cb) {
  burst_->on_migration_received(cb);
  loopback_->on_migration_received(std::move(cb));
}

void CompositeMigrationClient::on_migration_failed(
    std::function<void(const MigrationFailedEvent&)> cb) {
  burst_->on_migration_failed(cb);
  loopback_->on_migration_failed(std::move(cb));
}

void CompositeMigrationClient::on_endpoint_disconnected(
    std::function<void(const EndpointDisconnectedEvent&)> cb) {
  burst_->on_endpoint_disconnected(cb);
  loopback_->on_endpoint_disconnected(std::move(cb));
}

void CompositeMigrationClient::on_connection_received(
    std::function<void(int)> cb) {
  burst_->on_connection_received(cb);
  loopback_->on_connection_received(std::move(cb));
}

// --- Lifecycle: fan out to both backends. The Kafka adapter's connect_to
//     and wait_ready are no-ops (Mooncake / Kafka discovery is handled at
//     construction time), so the observable behavior is dominated by the
//     loopback backend's real handshake.

void CompositeMigrationClient::connect_to(int remote_endpoint_id,
                                          const std::string& role,
                                          const std::string& service_name) {
  burst_->connect_to(remote_endpoint_id, role, service_name);
  loopback_->connect_to(remote_endpoint_id, role, service_name);
}

void CompositeMigrationClient::wait_ready(int timeout_ms) {
  burst_->wait_ready(timeout_ms);
  loopback_->wait_ready(timeout_ms);
}

void CompositeMigrationClient::shutdown(bool drain) {
  // Order: burst first (drains outstanding Kafka acks via its own poll
  // during shutdown), loopback second (its shmem migration worker gets a
  // clean SHUTDOWN cmd afterwards). Not load-bearing — both must complete
  // before the composite is destroyed — but this matches a "far-side
  // first" teardown convention.
  burst_->shutdown(drain);
  loopback_->shutdown(drain);
}

}  // namespace tt::services
