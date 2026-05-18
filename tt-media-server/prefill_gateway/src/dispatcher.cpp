// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <optional>
#include <utility>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/prefill_selector.hpp"

namespace tt::gateway {

Dispatcher::Dispatcher(PrefillRegistry& registry, AffinityCache& affinityCache,
                       Senders senders)
    : registry_(registry),
      affinity_cache_(affinityCache),
      senders_(std::move(senders)) {}

void Dispatcher::onPrefillRequest(
    const tt::sockets::PrefillRequestMessage& msg) {
  auto prefills = registry_.snapshot();
  auto sticky = (msg.registration_hash != 0)
                    ? affinity_cache_.lookup(msg.registration_hash)
                    : std::nullopt;

  auto chosenOpt = selectPrefill(prefills, msg.registration_hash, sticky,
                                 round_robin_cursor_);
  if (!chosenOpt.has_value()) {
    failTaskToDecode(msg.task_id, "no_prefill_available");
    return;
  }

  const std::string& chosen = *chosenOpt;

  registry_.incrementInflight(chosen);
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    in_flight_[msg.task_id] = {chosen, msg.registration_hash};
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
      in_flight_.erase(msg.task_id);
    }
    failTaskToDecode(msg.task_id, "prefill_send_failed");
  }
}

void Dispatcher::onPrefillResult(const std::string& fromServerId,
                                 const tt::sockets::PrefillResultMessage& msg) {
  std::optional<InFlightEntry> entry;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    auto it = in_flight_.find(msg.task_id);
    if (it != in_flight_.end()) {
      entry = std::move(it->second);
      in_flight_.erase(it);
    }
  }

  // Decrement against the responder, not the original assignee, so a stray
  // result still decrements the right counter.
  registry_.decrementInflight(fromServerId);

  // Don't cache failures — they'd resend to the same broken prefill.
  if (entry && !msg.error && entry->registration_hash != 0) {
    affinity_cache_.record(entry->registration_hash, fromServerId);
  }

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(msg);
  }
}

void Dispatcher::onCacheBlocksAdded(
    const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
  registry_.addCachedBlocks(msg.server_id, msg.block_hashes);
}

void Dispatcher::onCacheBlocksEvicted(
    const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
  registry_.evictCachedBlocks(msg.server_id, msg.block_hashes);
}

void Dispatcher::onPrefillDown(const std::string& serverId) {
  affinity_cache_.evictPrefill(serverId);

  std::vector<uint32_t> orphaned;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    for (auto it = in_flight_.begin(); it != in_flight_.end();) {
      if (it->second.prefill_id == serverId) {
        orphaned.push_back(it->first);
        it = in_flight_.erase(it);
      } else {
        ++it;
      }
    }
  }

  for (uint32_t taskId : orphaned) {
    failTaskToDecode(taskId, "prefill_down");
  }
}

void Dispatcher::failTaskToDecode(uint32_t taskId, const std::string& reason) {
  tt::sockets::PrefillResultMessage err(taskId);
  err.error = true;
  err.finished = true;
  err.generated_text = reason;

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(err);
  }
}

}  // namespace tt::gateway
