// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "domain/slot_types.hpp"

namespace tt::sockets {

/**
 * @brief Prefill request message - sent from decode server to prefill server
 */
struct PrefillRequestMessage {
  uint32_t task_id;
  size_t registration_hash = 0;
  std::vector<int64_t> token_ids;
  std::optional<int> max_tokens;
  std::optional<uint32_t> slot_id;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  bool fast_mode = false;

  explicit PrefillRequestMessage(uint32_t taskId) : task_id(taskId) {}

  template <class Archive>
  void write(Archive& ar) const {
    int mt = max_tokens.has_value() ? max_tokens.value() : -1;
    uint32_t sid = slot_id.value_or(tt::domain::INVALID_SLOT_ID);
    bool hasTemp = temperature.has_value();
    float tempVal = temperature.value_or(0.0f);
    bool hasTopP = top_p.has_value();
    float topPVal = top_p.value_or(0.0f);
    bool hasTopK = top_k.has_value();
    int topKVal = top_k.value_or(0);
    uint64_t hash64 = static_cast<uint64_t>(registration_hash);
    ar(task_id, hash64, token_ids, mt, sid, hasTemp, tempVal, hasTopP, topPVal,
       hasTopK, topKVal, fast_mode);
  }

  template <class Archive>
  static PrefillRequestMessage read(Archive& ar) {
    uint32_t tid;
    uint64_t hash64;
    std::vector<int64_t> tids;
    int mt;
    uint32_t sid;
    bool hasTemp;
    float tempVal;
    bool hasTopP;
    float topPVal;
    bool hasTopK;
    int topKVal;
    bool fastMode;
    ar(tid, hash64, tids, mt, sid, hasTemp, tempVal, hasTopP, topPVal, hasTopK,
       topKVal, fastMode);
    PrefillRequestMessage msg(tid);
    msg.registration_hash = static_cast<size_t>(hash64);
    msg.token_ids = std::move(tids);
    msg.max_tokens = (mt == -1) ? std::nullopt : std::optional<int>(mt);
    msg.slot_id = (sid == tt::domain::INVALID_SLOT_ID)
                      ? std::nullopt
                      : std::optional<uint32_t>(sid);
    if (hasTemp) msg.temperature = tempVal;
    if (hasTopP) msg.top_p = topPVal;
    if (hasTopK) msg.top_k = topKVal;
    msg.fast_mode = fastMode;
    return msg;
  }
};

/**
 * @brief Prefill result message - sent from prefill server back to decode
 * server
 *
 * Contains the first token and updated sequence for decode server to continue
 * generation.
 */
struct PrefillResultMessage {
  uint32_t task_id;
  std::string generated_text;
  bool finished = false;
  bool error = false;
  int tokens_generated = 0;
  double processing_time_ms = 0.0;
  std::vector<int64_t> token_ids;
  std::optional<int> remaining_tokens;
  std::optional<uint32_t> slot_id;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  bool fast_mode = false;

  explicit PrefillResultMessage(uint32_t taskId) : task_id(taskId) {}

  template <class Archive>
  void write(Archive& ar) const {
    int rt = remaining_tokens.has_value() ? remaining_tokens.value() : -1;
    uint32_t sid = slot_id.value_or(tt::domain::INVALID_SLOT_ID);
    bool hasTemp = temperature.has_value();
    float tempVal = temperature.value_or(0.0f);
    bool hasTopP = top_p.has_value();
    float topPVal = top_p.value_or(0.0f);
    bool hasTopK = top_k.has_value();
    int topKVal = top_k.value_or(0);
    ar(task_id, generated_text, finished, tokens_generated, processing_time_ms,
       token_ids, rt, sid, error, hasTemp, tempVal, hasTopP, topPVal, hasTopK,
       topKVal, fast_mode);
  }

  template <class Archive>
  static PrefillResultMessage read(Archive& ar) {
    uint32_t tid;
    std::string genText;
    bool fin;
    int tg;
    double pt;
    std::vector<int64_t> tids;
    int rt;
    uint32_t sid;
    bool err;
    bool hasTemp;
    float tempVal;
    bool hasTopP;
    float topPVal;
    bool hasTopK;
    int topKVal;
    bool fastMode;
    ar(tid, genText, fin, tg, pt, tids, rt, sid, err, hasTemp, tempVal, hasTopP,
       topPVal, hasTopK, topKVal, fastMode);
    PrefillResultMessage msg(tid);
    msg.generated_text = std::move(genText);
    msg.finished = fin;
    msg.tokens_generated = tg;
    msg.processing_time_ms = pt;
    msg.token_ids = std::move(tids);
    msg.remaining_tokens = (rt == -1) ? std::nullopt : std::optional<int>(rt);
    msg.slot_id = (sid == tt::domain::INVALID_SLOT_ID)
                      ? std::nullopt
                      : std::optional<uint32_t>(sid);
    msg.error = err;
    if (hasTemp) msg.temperature = tempVal;
    if (hasTopP) msg.top_p = topPVal;
    if (hasTopK) msg.top_k = topKVal;
    msg.fast_mode = fastMode;
    return msg;
  }
};

/**
 * @brief Health check message
 */
struct HealthCheckMessage {
  std::string server_id;
  double cpu_usage = 0.0;
  double memory_usage = 0.0;
  int active_tasks = 0;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, cpu_usage, memory_usage, active_tasks);
  }

  template <class Archive>
  static HealthCheckMessage read(Archive& ar) {
    HealthCheckMessage msg;
    ar(msg.server_id, msg.cpu_usage, msg.memory_usage, msg.active_tasks);
    return msg;
  }
};

/**
 * @brief Load balancing info message
 */
struct LoadBalanceMessage {
  std::string server_id;
  int queue_size = 0;
  double avg_processing_time = 0.0;
  bool accepting_tasks = false;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, queue_size, avg_processing_time, accepting_tasks);
  }

  template <class Archive>
  static LoadBalanceMessage read(Archive& ar) {
    LoadBalanceMessage msg;
    ar(msg.server_id, msg.queue_size, msg.avg_processing_time,
       msg.accepting_tasks);
    return msg;
  }
};

/**
 * @brief Prefill registers itself with the gateway on connect.
 *
 * Identity is `server_id` (stable across reconnects), not (host, port). The
 * gateway uses this to seed its peer registry and accept routing decisions
 * against the prefill.
 */
struct PrefillRegistrationMessage {
  std::string server_id;
  uint32_t max_in_flight = 0;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, max_in_flight);
  }

  template <class Archive>
  static PrefillRegistrationMessage read(Archive& ar) {
    PrefillRegistrationMessage msg;
    ar(msg.server_id, msg.max_in_flight);
    return msg;
  }
};

/**
 * @brief Gateway notifies decode which prefill server was picked for a task.
 *
 * Informational for v1 (decode logs / metrics). Reusable for KV-transfer
 * routing later.
 */
struct PrefillAssignmentMessage {
  uint32_t task_id = 0;
  std::string server_id;

  template <class Archive>
  void write(Archive& ar) const {
    ar(task_id, server_id);
  }

  template <class Archive>
  static PrefillAssignmentMessage read(Archive& ar) {
    PrefillAssignmentMessage msg;
    ar(msg.task_id, msg.server_id);
    return msg;
  }
};

/**
 * @brief Prefill informs gateway about new cache blocks (after a request).
 *
 * Gateway updates its per-prefill block-cache view used for longest-prefix
 * match routing. Prefill may batch multiple block hashes per message.
 */
struct PrefillCacheBlocksAddedMessage {
  std::string server_id;
  std::vector<uint64_t> block_hashes;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, block_hashes);
  }

  template <class Archive>
  static PrefillCacheBlocksAddedMessage read(Archive& ar) {
    PrefillCacheBlocksAddedMessage msg;
    ar(msg.server_id, msg.block_hashes);
    return msg;
  }
};

/**
 * @brief Prefill informs gateway about evicted cache blocks (LRU pressure).
 *
 * Mirror of PrefillCacheBlocksAddedMessage; gateway removes these from the
 * per-prefill view so they don't influence routing.
 */
struct PrefillCacheBlocksEvictedMessage {
  std::string server_id;
  std::vector<uint64_t> block_hashes;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, block_hashes);
  }

  template <class Archive>
  static PrefillCacheBlocksEvictedMessage read(Archive& ar) {
    PrefillCacheBlocksEvictedMessage msg;
    ar(msg.server_id, msg.block_hashes);
    return msg;
  }
};

/**
 * @brief Wire-protocol message-type tags shared by gateway, decode, prefill.
 *
 * Tags are short ASCII strings prepended by SocketManager::sendObject and
 * matched by SocketManager::registerHandler. Existing tags
 * ("prefill_request", "prefill_result", "health_check") are NOT redeclared
 * here to keep changes additive; new tags below.
 */
namespace tags {
constexpr const char* PREFILL_REGISTRATION = "prefill_registration";
constexpr const char* PREFILL_ASSIGNMENT = "prefill_assignment";
constexpr const char* PREFILL_CACHE_BLOCKS_ADDED = "prefill_cache_added";
constexpr const char* PREFILL_CACHE_BLOCKS_EVICTED = "prefill_cache_evicted";
}  // namespace tags

}  // namespace tt::sockets
