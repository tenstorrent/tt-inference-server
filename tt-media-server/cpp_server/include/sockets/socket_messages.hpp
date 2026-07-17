// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "domain/llm/llm_error_reason.hpp"
#include "domain/sentinel_values.hpp"

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
  uint32_t taskId;
  std::vector<uint64_t> registrationHashes;
  std::vector<uint32_t> tokenIds;
  std::optional<int> maxTokens;
  std::optional<uint32_t> slotId;
  std::optional<float> temperature;
  std::optional<float> topP;
  std::optional<int> topK;
  bool fastMode = false;
  int decodePositionId = 0;
  int decodeSkipTokens = 0;

  explicit PrefillRequestMessage(uint32_t taskId) : taskId(taskId) {}

  template <class Archive>
  void write(Archive& ar) const {
    int mt = maxTokens.has_value() ? maxTokens.value() : -1;
    uint32_t sid = slotId.value_or(tt::domain::INVALID_SLOT_ID);
    bool hasTemp = temperature.has_value();
    float tempVal = temperature.value_or(0.0f);
    bool hasTopP = topP.has_value();
    float topPVal = topP.value_or(0.0f);
    bool hasTopK = topK.has_value();
    int topKVal = topK.value_or(0);
    ar(taskId, registrationHashes, tokenIds, mt, sid, hasTemp, tempVal, hasTopP,
       topPVal, hasTopK, topKVal, fastMode, decodePositionId, decodeSkipTokens);
  }

  template <class Archive>
  static PrefillRequestMessage read(Archive& ar) {
    uint32_t tid;
    std::vector<uint64_t> hashes;
    std::vector<uint32_t> tids;
    int mt;
    uint32_t sid;
    bool hasTemp;
    float tempVal;
    bool hasTopP;
    float topPVal;
    bool hasTopK;
    int topKVal;
    bool fastMode;
    int decodePositionId;
    int decodeSkipTokens;
    ar(tid, hashes, tids, mt, sid, hasTemp, tempVal, hasTopP, topPVal, hasTopK,
       topKVal, fastMode, decodePositionId, decodeSkipTokens);
    PrefillRequestMessage msg(tid);
    msg.registrationHashes = std::move(hashes);
    msg.tokenIds = std::move(tids);
    msg.maxTokens = (mt == -1) ? std::nullopt : std::optional<int>(mt);
    msg.slotId = (sid == tt::domain::INVALID_SLOT_ID)
                     ? std::nullopt
                     : std::optional<uint32_t>(sid);
    if (hasTemp) msg.temperature = tempVal;
    if (hasTopP) msg.topP = topPVal;
    if (hasTopK) msg.topK = topKVal;
    msg.fastMode = fastMode;
    msg.decodePositionId = decodePositionId;
    msg.decodeSkipTokens = decodeSkipTokens;
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
  uint32_t taskId;
  std::string generatedText;
  bool error = false;
  std::vector<uint32_t> tokenIds;
  std::optional<int> remainingTokens;
  std::optional<uint32_t> slotId;
  std::optional<float> temperature;
  std::optional<float> topP;
  std::optional<int> topK;
  bool fastMode = false;
  // Number of prompt tokens the prefill server served from its KV cache
  // (prefix-cache reuse). The decode server surfaces this as
  // usage.prompt_tokens_details.cached_tokens.
  int cachedTokens = 0;
  // Unique 64-bit ID correlating this prefill result with the migration
  // (KV transfer) that produced it. Generated on the prefill server.
  uint64_t migrationId = 0;

  explicit PrefillResultMessage(uint32_t taskId) : taskId(taskId) {}

  template <class Archive>
  void write(Archive& ar) const {
    int rt = remainingTokens.has_value() ? remainingTokens.value() : -1;
    uint32_t sid = slotId.value_or(tt::domain::INVALID_SLOT_ID);
    bool hasTemp = temperature.has_value();
    float tempVal = temperature.value_or(0.0f);
    bool hasTopP = topP.has_value();
    float topPVal = topP.value_or(0.0f);
    bool hasTopK = topK.has_value();
    int topKVal = topK.value_or(0);
    ar(taskId, generatedText, tokenIds, rt, sid, error, hasTemp, tempVal,
       hasTopP, topPVal, hasTopK, topKVal, fastMode, cachedTokens, migrationId);
  }

  template <class Archive>
  static PrefillResultMessage read(Archive& ar) {
    uint32_t tid;
    std::string genText;
    std::vector<uint32_t> tids;
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
    int cachedTokens;
    uint64_t migrationId;
    ar(tid, genText, tids, rt, sid, err, hasTemp, tempVal, hasTopP, topPVal,
       hasTopK, topKVal, fastMode, cachedTokens, migrationId);
    PrefillResultMessage msg(tid);
    msg.generatedText = std::move(genText);
    msg.tokenIds = std::move(tids);
    msg.remainingTokens = (rt == -1) ? std::nullopt : std::optional<int>(rt);
    msg.slotId = (sid == tt::domain::INVALID_SLOT_ID)
                     ? std::nullopt
                     : std::optional<uint32_t>(sid);
    msg.error = err;
    if (hasTemp) msg.temperature = tempVal;
    if (hasTopP) msg.topP = topPVal;
    if (hasTopK) msg.topK = topKVal;
    msg.fastMode = fastMode;
    msg.cachedTokens = cachedTokens;
    msg.migrationId = migrationId;
    return msg;
  }
};

inline constexpr std::string_view PREFILL_TIMEOUT_ERROR_TEXT = "timeout";

inline tt::domain::llm::LLMErrorReason errorReasonFromPrefillResult(
    const PrefillResultMessage& message) {
  return message.error && message.generatedText == PREFILL_TIMEOUT_ERROR_TEXT
             ? tt::domain::llm::LLMErrorReason::TIMEOUT
             : tt::domain::llm::LLMErrorReason::GENERIC;
}

inline std::string prefillErrorTextForReason(
    tt::domain::llm::LLMErrorReason reason, std::string genericError) {
  if (reason == tt::domain::llm::LLMErrorReason::TIMEOUT) {
    return std::string(PREFILL_TIMEOUT_ERROR_TEXT);
  }
  return genericError.empty() ? "error" : std::move(genericError);
}

struct PrefillHealthRequestMessage {
  template <class Archive>
  void write(Archive&) const {}

  template <class Archive>
  static PrefillHealthRequestMessage read(Archive&) {
    return {};
  }
};

struct PrefillHealthStatusMessage
    : SerializableMessage<PrefillHealthStatusMessage> {
  bool ready = false;

  template <class F>
  void fields(F&& f) {
    f(ready);
  }
  template <class F>
  void fields(F&& f) const {
    f(ready);
  }
};

// Prefill -> gateway, sent on connect. `serverId` is stable across reconnects.
struct PrefillRegistrationMessage
    : SerializableMessage<PrefillRegistrationMessage> {
  std::string serverId;
  uint32_t maxInFlight = 0;

  template <class F>
  void fields(F&& f) {
    f(serverId, maxInFlight);
  }
  template <class F>
  void fields(F&& f) const {
    f(serverId, maxInFlight);
  }
};

// Decode -> gateway/prefill. Best-effort cancellation for an in-flight prefill
// task.
struct CancelPrefillMessage : SerializableMessage<CancelPrefillMessage> {
  uint32_t taskId = 0;

  template <class F>
  void fields(F&& f) {
    f(taskId);
  }
  template <class F>
  void fields(F&& f) const {
    f(taskId);
  }
};

// Gateway -> prefill. Periodically retried until the gateway gets a
// PrefillRegistrationMessage back. Triggers (re-)registration regardless of
// transport semantics
struct RegistrationProbeMessage {
  template <class Archive>
  void write(Archive&) const {}

  template <class Archive>
  static RegistrationProbeMessage read(Archive&) {
    return {};
  }
};

// Prefill -> gateway. Updates the gateway's per-prefill block-cache view
// used by longest-prefix-match routing.
struct PrefillCacheBlocksAddedMessage
    : SerializableMessage<PrefillCacheBlocksAddedMessage> {
  std::string serverId;
  std::vector<uint64_t> blockHashes;

  template <class F>
  void fields(F&& f) {
    f(serverId, blockHashes);
  }
  template <class F>
  void fields(F&& f) const {
    f(serverId, blockHashes);
  }
};

/**
 * Prefill -> decode: reserve a decode KV slot before running prefill-first
 * disaggregation. Transport is InterServerService (ZMQ); peer selection may
 * still use etcd discovery under DYNAMO_ROUTING.
 */
struct SlotReservationRequestMessage
    : SerializableMessage<SlotReservationRequestMessage> {
  uint32_t taskId = 0;
  std::string prefillServerId;
  std::vector<uint64_t> registrationHashes;
  bool hasPreviousResponseId = false;
  std::string previousResponseId;
  int promptTokenCount = 0;

  template <class F>
  void fields(F&& f) {
    f(taskId, prefillServerId, registrationHashes, hasPreviousResponseId,
      previousResponseId, promptTokenCount);
  }
  template <class F>
  void fields(F&& f) const {
    f(taskId, prefillServerId, registrationHashes, hasPreviousResponseId,
      previousResponseId, promptTokenCount);
  }
};

struct SlotReservationResponseMessage
    : SerializableMessage<SlotReservationResponseMessage> {
  uint32_t taskId = 0;
  bool hasSlot = false;
  uint32_t slotId = tt::domain::INVALID_SLOT_ID;
  int decodePositionId = 0;
  int decodeSkipTokens = 0;
  bool continuation = false;
  int accumulatedThinkTokens = 0;
  bool error = false;
  std::string errorText;

  template <class F>
  void fields(F&& f) {
    f(taskId, hasSlot, slotId, decodePositionId, decodeSkipTokens, continuation,
      accumulatedThinkTokens, error, errorText);
  }
  template <class F>
  void fields(F&& f) const {
    f(taskId, hasSlot, slotId, decodePositionId, decodeSkipTokens, continuation,
      accumulatedThinkTokens, error, errorText);
  }
};

namespace tags {
constexpr std::string_view PREFILL_REQUEST = "prefill_request";
constexpr std::string_view PREFILL_RESULT = "prefill_result";
constexpr std::string_view PREFILL_REGISTRATION = "prefill_registration";
constexpr std::string_view PREFILL_CACHE_BLOCKS_ADDED = "prefill_cache_added";
constexpr std::string_view REGISTRATION_PROBE = "registration_probe";
constexpr std::string_view CANCEL_PREFILL = "cancel_prefill";
constexpr std::string_view SLOT_RESERVATION_REQUEST =
    "slot_reservation_request";
constexpr std::string_view SLOT_RESERVATION_RESPONSE =
    "slot_reservation_response";
constexpr std::string_view PREFILL_HEALTH_REQUEST = "prefill_health_request";
constexpr std::string_view PREFILL_HEALTH_STATUS = "prefill_health_status";
}  // namespace tags

}  // namespace tt::sockets
