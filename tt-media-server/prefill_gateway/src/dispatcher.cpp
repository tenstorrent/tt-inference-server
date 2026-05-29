// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <chrono>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/gateway_metrics.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/prefill_selector.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {

Dispatcher::Dispatcher(PrefillRegistry& registry, AffinityCache& affinityCache,
                       Senders senders)
    : Dispatcher(registry, affinityCache, std::move(senders),
                 Options{std::chrono::minutes(5), std::chrono::minutes(1),
                         std::chrono::seconds(30), 3}) {}

Dispatcher::Dispatcher(PrefillRegistry& registry, AffinityCache& affinityCache,
                       Senders senders, Options options)
    : registry_(registry),
      affinity_cache_(affinityCache),
      senders_(std::move(senders)),
      options_(options) {}

void Dispatcher::onPrefillRequest(
    const tt::sockets::PrefillRequestMessage& msg) {
  auto prefills = registry_.snapshot();
  const uint64_t affinityKey =
      msg.registration_hashes.empty() ? 0 : msg.registration_hashes.front();
  auto sticky =
      (affinityKey != 0) ? affinity_cache_.lookup(affinityKey) : std::nullopt;

  auto selection = selectPrefillWithReason(prefills, affinityKey, sticky,
                                           round_robin_cursor_);
  GatewayMetrics::instance().recordRoutingDecision(
      routingReasonName(selection.reason));
  GatewayMetrics::instance().setRoutingTableSize(affinity_cache_.size());
  if (!selection.server_id.has_value()) {
    const auto summary = summarizePrefillEligibility(prefills);
    TT_LOG_WARN(
        "[Dispatcher] taskId={} no eligible prefill (total={}, healthy={}, "
        "accepting={}, capacity_available={})",
        msg.task_id, summary.total, summary.healthy, summary.accepting,
        summary.capacity_available);
    failTaskToDecode(msg.task_id, "no_prefill_available");
    return;
  }

  const std::string& chosen = *selection.server_id;
  const bool usedSticky = selection.reason == PrefillRoutingReason::PrefixMatch;
  if (usedSticky) {
    GatewayMetrics::instance().observePrefixMatchDepth(
        msg.registration_hashes.size());
  }
  TT_LOG_INFO(
      "[Dispatcher] taskId={} route prefill='{}' reason={} sticky={} hash={}",
      msg.task_id, chosen, routingReasonName(selection.reason), usedSticky,
      affinityKey);

  registry_.incrementInflight(chosen);
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    in_flight_[msg.task_id] = {chosen, affinityKey, Clock::now()};
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
    TT_LOG_ERROR(
        "[Dispatcher] taskId={} send to prefill='{}' failed, failing task",
        msg.task_id, chosen);
    registry_.decrementInflight(chosen);
    InFlightEntry failedEntry;
    {
      std::lock_guard<std::mutex> lock(inflight_mutex_);
      auto it = in_flight_.find(msg.task_id);
      if (it != in_flight_.end()) {
        failedEntry = it->second;
      }
      in_flight_.erase(msg.task_id);
    }
    failTaskToDecode(msg.task_id, "prefill_send_failed", &failedEntry);
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

  if (!entry) {
    TT_LOG_WARN(
        "[Dispatcher] Dropping result for unknown taskId={} from prefill='{}'",
        msg.task_id, fromServerId);
    return;
  }

  // Decrement against the responder, not the original assignee, so a stray
  // result still decrements the right counter.
  registry_.decrementInflight(fromServerId);
  const auto latency = Clock::now() - entry->started_at;

  // Don't cache failures — they'd resend to the same broken prefill.
  if (!msg.error && entry->affinity_key != 0) {
    affinity_cache_.record(entry->affinity_key, fromServerId);
    GatewayMetrics::instance().setRoutingTableSize(affinity_cache_.size());
  }

  if (msg.error) {
    TT_LOG_ERROR("[Dispatcher] taskId={} result error from prefill='{}'",
                 msg.task_id, fromServerId);
    GatewayMetrics::instance().recordRequestFailed("prefill_result_error");
    GatewayMetrics::instance().recordRequestCompleted(fromServerId, "error",
                                                      latency);
  } else {
    TT_LOG_INFO("[Dispatcher] taskId={} result ok from prefill='{}' tokens={}",
                msg.task_id, fromServerId, msg.tokens_generated);
    GatewayMetrics::instance().recordRequestCompleted(fromServerId, "success",
                                                      latency);
  }

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(msg);
  }
}

void Dispatcher::onPrefillCancel(const tt::sockets::CancelPrefillMessage& msg) {
  std::optional<InFlightEntry> entry;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    auto it = in_flight_.find(msg.task_id);
    if (it != in_flight_.end()) {
      entry = std::move(it->second);
      in_flight_.erase(it);
    }
  }

  if (!entry) {
    TT_LOG_DEBUG("[Dispatcher] Ignoring cancel for unknown taskId={}",
                 msg.task_id);
    return;
  }

  registry_.decrementInflight(entry->prefill_id);

  bool sent = false;
  if (senders_.sendCancelToPrefill) {
    sent = senders_.sendCancelToPrefill(entry->prefill_id, msg);
  }
  GatewayMetrics::instance().recordCancel(sent);
  if (sent) {
    TT_LOG_INFO("[Dispatcher] taskId={} cancel -> prefill='{}'", msg.task_id,
                entry->prefill_id);
    return;
  }
  TT_LOG_WARN(
      "[Dispatcher] taskId={} cancel send to prefill='{}' failed; "
      "cancellation is best-effort",
      msg.task_id, entry->prefill_id);
}

void Dispatcher::onCacheBlocksAdded(
    const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
  registry_.addCachedBlocks(msg.server_id, msg.block_hashes);
  GatewayMetrics::instance().recordCacheBlocksAdded(msg.block_hashes.size());
}

void Dispatcher::onCacheBlocksEvicted(
    const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
  registry_.evictCachedBlocks(msg.server_id, msg.block_hashes);
  GatewayMetrics::instance().recordCacheBlocksEvicted(msg.block_hashes.size());
}

void Dispatcher::onPrefillDown(const std::string& serverId) {
  affinity_cache_.evictPrefill(serverId);

  std::vector<std::pair<uint32_t, InFlightEntry>> orphaned;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    std::erase_if(in_flight_, [&orphaned, &serverId](const auto& task) {
      if (task.second.prefill_id != serverId) {
        return false;
      }
      orphaned.push_back(task);
      return true;
    });
  }

  if (!orphaned.empty()) {
    TT_LOG_WARN("[Dispatcher] prefill='{}' down, failing {} in-flight tasks",
                serverId, orphaned.size());
    GatewayMetrics::instance().recordPrefillDownTasks(orphaned.size());
  }

  for (const auto& [taskId, entry] : orphaned) {
    failTaskToDecode(taskId, "prefill_down", &entry);
  }
}

void Dispatcher::onRequestTimeouts(Clock::time_point now) {
  std::vector<std::string> recoveredPrefills;
  {
    std::lock_guard<std::mutex> lock(timeout_state_mutex_);
    std::erase_if(prefill_blocked_until_,
                  [&recoveredPrefills, now](const auto& blockedPrefill) {
                    if (now < blockedPrefill.second) {
                      return false;
                    }
                    recoveredPrefills.push_back(blockedPrefill.first);
                    return true;
                  });
  }
  for (const auto& prefillId : recoveredPrefills) {
    TT_LOG_INFO(
        "[Dispatcher] prefill='{}' accepting tasks after timeout "
        "cooldown",
        prefillId);
    registry_.setAcceptingTasks(prefillId, true);
  }

  if (options_.request_timeout.count() <= 0) {
    return;
  }

  std::vector<std::pair<uint32_t, InFlightEntry>> timedOut;
  {
    std::lock_guard<std::mutex> lock(inflight_mutex_);
    std::erase_if(in_flight_, [&timedOut, now, this](auto& task) {
      if (now - task.second.started_at < options_.request_timeout) {
        return false;
      }
      timedOut.emplace_back(task.first, std::move(task.second));
      return true;
    });
  }

  for (const auto& [taskId, entry] : timedOut) {
    TT_LOG_WARN("[Dispatcher] taskId={} timed out on prefill='{}'", taskId,
                entry.prefill_id);
    registry_.decrementInflight(entry.prefill_id);
    GatewayMetrics::instance().recordTimeout(entry.prefill_id);

    tt::sockets::CancelPrefillMessage cancel;
    cancel.task_id = taskId;
    if (senders_.sendCancelToPrefill &&
        !senders_.sendCancelToPrefill(entry.prefill_id, cancel)) {
      TT_LOG_WARN(
          "[Dispatcher] taskId={} timeout cancel send to prefill='{}' "
          "failed",
          taskId, entry.prefill_id);
    }

    if (options_.timeout_threshold > 0 && options_.timeout_window.count() > 0 &&
        options_.timeout_cooldown.count() > 0) {
      std::lock_guard<std::mutex> lock(timeout_state_mutex_);
      auto& history = prefill_timeout_history_[entry.prefill_id];
      history.push_back(now);
      while (!history.empty() &&
             now - history.front() > options_.timeout_window) {
        history.pop_front();
      }
      if (history.size() >= options_.timeout_threshold) {
        const auto blockedUntil = now + options_.timeout_cooldown;
        prefill_blocked_until_[entry.prefill_id] = blockedUntil;
        registry_.setAcceptingTasks(entry.prefill_id, false);
        history.clear();
        TT_LOG_WARN(
            "[Dispatcher] prefill='{}' disabled for new tasks after "
            "{} timeouts in {}ms",
            entry.prefill_id, options_.timeout_threshold,
            options_.timeout_window.count());
      }
    }

    failTaskToDecode(taskId, "timeout", &entry);
  }
}

void Dispatcher::failTaskToDecode(uint32_t taskId, const std::string& reason,
                                  const InFlightEntry* entry) {
  TT_LOG_ERROR("[Dispatcher] taskId={} failed: {}", taskId, reason);
  GatewayMetrics::instance().recordRequestFailed(reason);
  if (entry != nullptr && !entry->prefill_id.empty()) {
    GatewayMetrics::instance().recordRequestCompleted(
        entry->prefill_id, reason, Clock::now() - entry->started_at);
  }

  tt::sockets::PrefillResultMessage err(taskId);
  err.error = true;
  err.finished = true;
  err.generated_text = reason;

  if (senders_.sendResultToDecode) {
    senders_.sendResultToDecode(err);
  }
}

}  // namespace tt::gateway
