// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "domain/session.hpp"
#include "ipc/boost_ipc_queue.hpp"
#include "utils/concurrent_map.hpp"
#include "messaging/kafka_client.hpp"

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
  bool closeSession(const std::string& sessionId);
  bool assignSlotId(const std::string& sessionId, uint32_t slotId);
  uint32_t getSlotIdBySessionId(const std::string& sessionId) const;
  uint32_t acquireSessionSlot(const std::string& sessionId);
  std::optional<domain::Session> getSession(const std::string& sessionId) const;
  size_t getActiveSessionCount() const;

  // In-flight session management
  void setSessionInFlight(const std::string& sessionId, bool inFlight);

 private:
  void evictOldSessions();
  std::future<uint32_t> requestSlotIdFromMemoryManager(
      const std::string& sessionId);
  void sendDeallocRequest(const std::string& sessionId, uint32_t slotId);
  void drainResultQueue();

  mutable ConcurrentMap<std::string, domain::Session> sessions;

  std::unique_ptr<ipc::MemoryRequestQueue> memoryRequestQueue;
  std::unique_ptr<ipc::MemoryResultQueue> memoryResultQueue;

  using PromisePtr = std::shared_ptr<std::promise<uint32_t>>;
  ConcurrentMap<uint32_t, PromisePtr> pendingAllocations;
  std::atomic<bool> stopped{false};
  std::atomic<bool> evictionInProgress{false};
  std::thread drainThread;
  std::unique_ptr<tt::messaging::KafkaProducer> kafkaProducer;
  size_t maxSessions;

  void checkAndSendOffloadRequest(const std::string& sessionId);
};

}  // namespace tt::services
