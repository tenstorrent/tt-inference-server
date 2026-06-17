// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <chrono>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gateway/gateway_metrics.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/prefill_selector.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {

Dispatcher::Dispatcher(PrefillRegistry& registry, Senders senders)
    : Dispatcher(registry, std::move(senders),
                 Options{std::chrono::minutes(5), std::chrono::minutes(1),
                         std::chrono::seconds(30), 3}) {}

Dispatcher::Dispatcher(PrefillRegistry& registry, Senders senders,
                       Options options)
    : registry(registry), senders(std::move(senders)), options(options) {}

void Dispatcher::onPrefillRequest(
    const tt::sockets::PrefillRequestMessage& msg) {
  auto prefills = registry.routingSnapshot(msg.registrationHashes);
  const uint64_t firstRegistrationHash =
      msg.registrationHashes.empty() ? 0 : msg.registrationHashes.front();

  auto selection =
      selectPrefill(prefills, roundRobinCursor, msg.preferredPrefillId);
  GatewayMetrics::instance().recordRoutingDecision(
      routingReasonName(selection.reason));
  if (!selection.serverId.has_value()) {
    const auto summary = summarizePrefillEligibility(prefills);
    TT_LOG_WARN(
        "[Dispatcher] taskId={} no eligible prefill (total={}, healthy={}, "
        "accepting={}, capacity_available={})",
        msg.taskId, summary.total, summary.healthy, summary.accepting,
        summary.capacityAvailable);
    failTaskToDecode(msg.taskId, "no_prefill_available");
    return;
  }

  const std::string& chosen = *selection.serverId;
  if (selection.prefixMatchDepth > 0) {
    GatewayMetrics::instance().observePrefixMatchDepth(
        selection.prefixMatchDepth);
  }
  TT_LOG_INFO(
      "[Dispatcher] taskId={} route prefill='{}' reason={} "
      "prefix_match_depth={} hash={} preferred_prefill='{}'",
      msg.taskId, chosen, routingReasonName(selection.reason),
      selection.prefixMatchDepth, firstRegistrationHash,
      msg.preferredPrefillId.value_or(""));

  registry.incrementInflight(chosen);
  {
    std::lock_guard<std::mutex> lock(inflightMutex);
    inFlight[msg.taskId] = {chosen, Clock::now()};
  }

  bool sent = false;
  if (senders.sendRequestToPrefill) {
    sent = senders.sendRequestToPrefill(chosen, msg);
  }

  if (!sent) {
    TT_LOG_ERROR(
        "[Dispatcher] taskId={} send to prefill='{}' failed, failing task",
        msg.taskId, chosen);
    registry.decrementInflight(chosen);
    InFlightEntry failedEntry;
    {
      std::lock_guard<std::mutex> lock(inflightMutex);
      auto it = inFlight.find(msg.taskId);
      if (it != inFlight.end()) {
        failedEntry = it->second;
      }
      inFlight.erase(msg.taskId);
    }
    failTaskToDecode(msg.taskId, "prefill_send_failed", &failedEntry);
  }
}

void Dispatcher::onPrefillResult(const std::string& fromServerId,
                                 const tt::sockets::PrefillResultMessage& msg) {
  std::optional<InFlightEntry> entry;
  {
    std::lock_guard<std::mutex> lock(inflightMutex);
    auto it = inFlight.find(msg.taskId);
    if (it != inFlight.end()) {
      entry = std::move(it->second);
      inFlight.erase(it);
    }
  }

  if (!entry) {
    TT_LOG_WARN(
        "[Dispatcher] Dropping result for unknown taskId={} from prefill='{}'",
        msg.taskId, fromServerId);
    return;
  }

  // Decrement against the responder, not the original assignee, so a stray
  // result still decrements the right counter.
  registry.decrementInflight(fromServerId);
  const auto latency = Clock::now() - entry->startedAt;

  if (msg.error) {
    TT_LOG_ERROR("[Dispatcher] taskId={} result error from prefill='{}'",
                 msg.taskId, fromServerId);
    GatewayMetrics::instance().recordRequestFailed("prefill_result_error");
    GatewayMetrics::instance().recordRequestCompleted(fromServerId, "error",
                                                      latency);
  } else {
    TT_LOG_INFO("[Dispatcher] taskId={} result ok from prefill='{}'",
                msg.taskId, fromServerId);
    GatewayMetrics::instance().recordRequestCompleted(fromServerId, "success",
                                                      latency);
  }

  if (senders.sendResultToDecode) {
    senders.sendResultToDecode(msg);
  }
}

void Dispatcher::onPrefillCancel(const tt::sockets::CancelPrefillMessage& msg) {
  std::optional<InFlightEntry> entry;
  {
    std::lock_guard<std::mutex> lock(inflightMutex);
    auto it = inFlight.find(msg.taskId);
    if (it != inFlight.end()) {
      entry = std::move(it->second);
      inFlight.erase(it);
    }
  }

  if (!entry) {
    TT_LOG_DEBUG("[Dispatcher] Ignoring cancel for unknown taskId={}",
                 msg.taskId);
    return;
  }

  registry.decrementInflight(entry->prefillId);

  bool sent = false;
  if (senders.sendCancelToPrefill) {
    sent = senders.sendCancelToPrefill(entry->prefillId, msg);
  }
  GatewayMetrics::instance().recordCancel(sent);
  if (sent) {
    TT_LOG_INFO("[Dispatcher] taskId={} cancel -> prefill='{}'", msg.taskId,
                entry->prefillId);
    return;
  }
  TT_LOG_WARN(
      "[Dispatcher] taskId={} cancel send to prefill='{}' failed; "
      "cancellation is best-effort",
      msg.taskId, entry->prefillId);
}

void Dispatcher::onCacheBlocksAdded(
    const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
  registry.addCachedBlocks(msg.serverId, msg.blockHashes);
  GatewayMetrics::instance().recordCacheBlocksAdded(msg.blockHashes.size());
}

void Dispatcher::onPrefillDown(const std::string& serverId) {
  std::vector<std::pair<uint32_t, InFlightEntry>> orphaned;
  {
    std::lock_guard<std::mutex> lock(inflightMutex);
    std::erase_if(inFlight, [&orphaned, &serverId](const auto& task) {
      if (task.second.prefillId != serverId) {
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
    std::lock_guard<std::mutex> lock(timeoutStateMutex);
    std::erase_if(prefillBlockedUntil,
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
    registry.setAcceptingTasks(prefillId, true);
  }

  if (options.requestTimeout.count() <= 0) {
    return;
  }

  std::vector<std::pair<uint32_t, InFlightEntry>> timedOut;
  {
    std::lock_guard<std::mutex> lock(inflightMutex);
    std::erase_if(inFlight, [&timedOut, now, this](auto& task) {
      if (now - task.second.startedAt < options.requestTimeout) {
        return false;
      }
      timedOut.emplace_back(task.first, std::move(task.second));
      return true;
    });
  }

  for (const auto& [taskId, entry] : timedOut) {
    TT_LOG_WARN("[Dispatcher] taskId={} timed out on prefill='{}'", taskId,
                entry.prefillId);
    registry.decrementInflight(entry.prefillId);
    GatewayMetrics::instance().recordTimeout(entry.prefillId);

    tt::sockets::CancelPrefillMessage cancel;
    cancel.taskId = taskId;
    if (senders.sendCancelToPrefill &&
        !senders.sendCancelToPrefill(entry.prefillId, cancel)) {
      TT_LOG_WARN(
          "[Dispatcher] taskId={} timeout cancel send to prefill='{}' "
          "failed",
          taskId, entry.prefillId);
    }

    if (options.timeoutThreshold > 0 && options.timeoutWindow.count() > 0 &&
        options.timeoutCooldown.count() > 0) {
      std::lock_guard<std::mutex> lock(timeoutStateMutex);
      auto& history = prefillTimeoutHistory[entry.prefillId];
      history.push_back(now);
      while (!history.empty() &&
             now - history.front() > options.timeoutWindow) {
        history.pop_front();
      }
      if (history.size() >= options.timeoutThreshold) {
        const auto blockedUntil = now + options.timeoutCooldown;
        prefillBlockedUntil[entry.prefillId] = blockedUntil;
        registry.setAcceptingTasks(entry.prefillId, false);
        history.clear();
        TT_LOG_WARN(
            "[Dispatcher] prefill='{}' disabled for new tasks after "
            "{} timeouts in {}ms",
            entry.prefillId, options.timeoutThreshold,
            options.timeoutWindow.count());
      }
    }

    failTaskToDecode(taskId, "timeout", &entry);
  }
}

void Dispatcher::failTaskToDecode(uint32_t taskId, const std::string& reason,
                                  const InFlightEntry* entry) {
  TT_LOG_ERROR("[Dispatcher] taskId={} failed: {}", taskId, reason);
  GatewayMetrics::instance().recordRequestFailed(reason);
  if (entry != nullptr && !entry->prefillId.empty()) {
    GatewayMetrics::instance().recordRequestCompleted(
        entry->prefillId, reason, Clock::now() - entry->startedAt);
  }

  tt::sockets::PrefillResultMessage err(taskId);
  err.error = true;
  err.generatedText = reason;

  if (senders.sendResultToDecode) {
    senders.sendResultToDecode(err);
  }
}

}  // namespace tt::gateway
