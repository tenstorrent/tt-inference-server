// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <utility>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/prefill_selector.hpp"

namespace tt::gateway {

Dispatcher::Dispatcher(PrefillRegistry& registry, AffinityCache& affinity_cache,
                       Senders senders)
    : registry_(registry),
      affinity_cache_(affinity_cache),
      senders_(std::move(senders)) {}

void Dispatcher::onPrefillRequest(const tt::sockets::PrefillRequestMessage& msg) {
  auto prefills = registry_.snapshot();
  auto sticky = (msg.registration_hash != 0)
                    ? affinity_cache_.lookup(msg.registration_hash)
                    : std::nullopt;

  SelectionResult decision =
      selectPrefill(prefills, msg.registration_hash, sticky, round_robin_cursor_);

  if (!decision.server_id.has_value()) {
    failTaskToDecode(msg.task_id, "no_prefill_available");
    return;
  }

  const std::string& chosen = *decision.server_id;

  registry_.incrementInflight(chosen);
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    in_flight_task_to_prefill_[msg.task_id] = chosen;
    in_flight_task_to_hash_[msg.task_id] = msg.registration_hash;
  }

  // Send assignment first so decode can prep KV-transfer ahead of the result.
  tt::sockets::PrefillAssignmentMessage assignment;
  assignment.task_id = msg.task_id;
  assignment.server_id = chosen;
  if (senders_.sendAssignmentToDecode) {
    senders_.sendAssignmentToDecode(assignment);
  }

  bool sent = false;
  if (senders_.sendRequestToPrefill) {
    sent = senders_.sendRequestToPrefill(chosen, msg);
  }

  if (!sent) {
    registry_.decrementInflight(chosen);
    {
      std::lock_guard<std::mutex> lock(inflight_mutex_);
      in_flight_task_to_prefill_.erase(msg.task_id);
      in_flight_task_to_hash_.erase(msg.task_id);
    }
    failTaskToDecode(msg.task_id, "prefill_send_failed");
  }
}

void Dispatcher::onPrefillResult(const std::string& from_server_id,
                                 const tt::sockets::PrefillResultMessage& msg) {
  std::string assigned;
  size_t hash = 0;
  bool was_tracked = false;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    auto it = in_flight_task_to_prefill_.find(msg.task_id);
    if (it != in_flight_task_to_prefill_.end()) {
      assigned = it->second;
      in_flight_task_to_prefill_.erase(it);
      was_tracked = true;
    }
    auto hit = in_flight_task_to_hash_.find(msg.task_id);
    if (hit != in_flight_task_to_hash_.end()) {
      hash = hit->second;
      in_flight_task_to_hash_.erase(hit);
    }
  }

  // Decrement against the responder, not the original assignee, so a stray
  // result still decrements the right counter.
  registry_.decrementInflight(from_server_id);

  // Don't cache failures — they'd resend to the same broken prefill.
  if (was_tracked && !msg.error && hash != 0) {
    affinity_cache_.record(hash, from_server_id);
  }

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(msg);
  }
  (void)assigned;
}

void Dispatcher::onCacheBlocksAdded(
    const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
  registry_.addCachedBlocks(msg.server_id, msg.block_hashes);
}

void Dispatcher::onCacheBlocksEvicted(
    const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
  registry_.evictCachedBlocks(msg.server_id, msg.block_hashes);
}

void Dispatcher::onPrefillDown(const std::string& server_id) {
  affinity_cache_.evictPrefill(server_id);

  std::vector<uint32_t> orphaned;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    for (auto it = in_flight_task_to_prefill_.begin();
         it != in_flight_task_to_prefill_.end();) {
      if (it->second == server_id) {
        orphaned.push_back(it->first);
        in_flight_task_to_hash_.erase(it->first);
        it = in_flight_task_to_prefill_.erase(it);
      } else {
        ++it;
      }
    }
  }

  for (uint32_t task_id : orphaned) {
    failTaskToDecode(task_id, "prefill_down");
  }
}

void Dispatcher::failTaskToDecode(uint32_t task_id, const std::string& reason) {
  tt::sockets::PrefillResultMessage err(task_id);
  err.error = true;
  err.finished = true;
  err.generated_text = reason;

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(err);
  }
}

}  // namespace tt::gateway
