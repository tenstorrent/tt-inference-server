// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "domain/session.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "utils/concurrent_map.hpp"

namespace trantor {
class EventLoop;
}

namespace tt::services {

using tt::domain::INVALID_SLOT_ID;

class SessionManager {
 public:
  SessionManager();
  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  domain::Session createSession(std::optional<uint32_t> slotId = std::nullopt,
                                bool inFlight = false);

  void createSessionAsync(std::function<void(domain::Session)> onSuccess,
                          std::function<void(const std::string&)> onError,
                          trantor::EventLoop* callerLoop,
                          bool inFlight = false);

  bool closeSession(const std::string& sessionId);
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;
  uint32_t acquireSessionSlot(const std::string& sessionId);
  std::optional<domain::Session> getSession(const std::string& sessionId) const;
  size_t getActiveSessionCount() const;

  void setSessionInFlight(const std::string& sessionId, bool inFlight);

 private:
  struct PendingAsyncSession {
    domain::Session session;
    std::string sessionId;
    std::function<void(domain::Session)> onSuccess;
    std::function<void(const std::string&)> onError;
    trantor::EventLoop* callerLoop;
    int attemptsRemaining;
    bool inFlight;

    PendingAsyncSession(domain::Session session, std::string sessionId,
                        std::function<void(domain::Session)> onSuccess,
                        std::function<void(const std::string&)> onError,
                        trantor::EventLoop* callerLoop, int attemptsRemaining,
                        bool inFlight)
        : session(std::move(session)),
          sessionId(std::move(sessionId)),
          onSuccess(std::move(onSuccess)),
          onError(std::move(onError)),
          callerLoop(callerLoop),
          attemptsRemaining(attemptsRemaining),
          inFlight(inFlight) {}
  };

  struct DeferredRetry {
    std::chrono::steady_clock::time_point retryAt;
    std::shared_ptr<PendingAsyncSession> pending;
  };

  void evictOldSessions();
  std::future<uint32_t> requestSlotIdFromMemoryManager(
      const std::string& sessionId);
  void sendDeallocRequest(const std::string& sessionId, uint32_t slotId);
  void drainResultQueue();
  void handleMemoryResult(const domain::ManageMemoryResult& result);
  void sendAsyncAllocRequest(std::shared_ptr<PendingAsyncSession> pending);
  void processRetryQueue();
  void processDeallocQueue();

  mutable ConcurrentMap<std::string, domain::Session> sessions;

  std::unique_ptr<ipc::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::MemoryResultQueue> memoryResultQueue;

  using PromisePtr = std::shared_ptr<std::promise<uint32_t>>;
  ConcurrentMap<uint32_t, PromisePtr> pendingAllocations;

  ConcurrentMap<uint32_t, std::shared_ptr<PendingAsyncSession>>
      pendingAsyncAllocations;
  std::mutex retryMutex;
  std::vector<DeferredRetry> retryQueue;

  struct DeferredDealloc {
    std::string sessionId;
    uint32_t slotId;
  };
  std::mutex deallocMutex;
  std::vector<DeferredDealloc> deallocQueue;

  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;
};

}  // namespace tt::services
