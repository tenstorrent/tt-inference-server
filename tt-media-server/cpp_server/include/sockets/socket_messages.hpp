// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "domain/slot_types.hpp"

namespace tt::sockets {

template <class Derived>
struct SerializableMessage {
  template <class Archive>
  void write(Archive& ar) const {
    static_cast<const Derived&>(*this).fields(
        [&](const auto&... xs) { ar(xs...); });
  }

  template <class Archive>
  static Derived read(Archive& ar) {
    Derived msg;
    msg.fields([&](auto&... xs) { ar(xs...); });
    return msg;
  }
};

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
struct HealthCheckMessage : SerializableMessage<HealthCheckMessage> {
  std::string server_id;
  double cpu_usage = 0.0;
  double memory_usage = 0.0;
  int active_tasks = 0;

  template <class F>
  void fields(F&& f) {
    f(server_id, cpu_usage, memory_usage, active_tasks);
  }
  template <class F>
  void fields(F&& f) const {
    f(server_id, cpu_usage, memory_usage, active_tasks);
  }
};

/**
 * @brief Load balancing info message
 */
struct LoadBalanceMessage : SerializableMessage<LoadBalanceMessage> {
  std::string server_id;
  int queue_size = 0;
  double avg_processing_time = 0.0;
  bool accepting_tasks = false;

  template <class F>
  void fields(F&& f) {
    f(server_id, queue_size, avg_processing_time, accepting_tasks);
  }
  template <class F>
  void fields(F&& f) const {
    f(server_id, queue_size, avg_processing_time, accepting_tasks);
  }
};

// Prefill -> gateway, sent on connect. `server_id` is stable across reconnects.
struct PrefillRegistrationMessage
    : SerializableMessage<PrefillRegistrationMessage> {
  std::string server_id;
  uint32_t max_in_flight = 0;

  template <class F>
  void fields(F&& f) {
    f(server_id, max_in_flight);
  }
  template <class F>
  void fields(F&& f) const {
    f(server_id, max_in_flight);
  }
};

// Decode -> gateway/prefill. Best-effort cancellation for an in-flight prefill
// task.
struct CancelPrefillMessage : SerializableMessage<CancelPrefillMessage> {
  uint32_t task_id = 0;

  template <class F>
  void fields(F&& f) {
    f(task_id);
  }
  template <class F>
  void fields(F&& f) const {
    f(task_id);
  }
};

// Gateway -> prefill. Periodically retried until the gateway gets a
// PrefillRegistrationMessage back. Triggers (re-)registration regardless of
// transport semantics
struct RegistrationProbeMessage
    : SerializableMessage<RegistrationProbeMessage> {
  uint32_t nonce = 0;

  template <class F>
  void fields(F&& f) {
    f(nonce);
  }
  template <class F>
  void fields(F&& f) const {
    f(nonce);
  }
};

// Gateway -> decode. Informs decode which prefill handled a task (for KV
// transfer / logs).
struct PrefillAssignmentMessage
    : SerializableMessage<PrefillAssignmentMessage> {
  uint32_t task_id = 0;
  std::string server_id;

  template <class F>
  void fields(F&& f) {
    f(task_id, server_id);
  }
  template <class F>
  void fields(F&& f) const {
    f(task_id, server_id);
  }
};

// Prefill -> gateway. Updates the gateway's per-prefill block-cache view
// used by longest-prefix-match routing.
struct PrefillCacheBlocksAddedMessage
    : SerializableMessage<PrefillCacheBlocksAddedMessage> {
  std::string server_id;
  std::vector<uint64_t> block_hashes;

  template <class F>
  void fields(F&& f) {
    f(server_id, block_hashes);
  }
  template <class F>
  void fields(F&& f) const {
    f(server_id, block_hashes);
  }
};

// Prefill -> gateway. Mirror of *Added; removes blocks from the routing view.
struct PrefillCacheBlocksEvictedMessage
    : SerializableMessage<PrefillCacheBlocksEvictedMessage> {
  std::string server_id;
  std::vector<uint64_t> block_hashes;

  template <class F>
  void fields(F&& f) {
    f(server_id, block_hashes);
  }
  template <class F>
  void fields(F&& f) const {
    f(server_id, block_hashes);
  }
};

// Wire-protocol tags for the new gateway messages. Existing tags
// ("prefill_request", etc.) remain string literals at their call sites.
namespace tags {
constexpr std::string_view PREFILL_REGISTRATION = "prefill_registration";
constexpr std::string_view PREFILL_ASSIGNMENT = "prefill_assignment";
constexpr std::string_view PREFILL_CACHE_BLOCKS_ADDED = "prefill_cache_added";
constexpr std::string_view PREFILL_CACHE_BLOCKS_EVICTED =
    "prefill_cache_evicted";
constexpr std::string_view REGISTRATION_PROBE = "registration_probe";
constexpr std::string_view CANCEL_PREFILL = "cancel_prefill";
}  // namespace tags

}  // namespace tt::sockets
