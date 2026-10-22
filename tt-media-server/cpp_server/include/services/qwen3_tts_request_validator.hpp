// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"

namespace tt::services::qwen3_tts {

/** What a Qwen3-TTS deployment accepts. */
struct RequestRules {
  config::Qwen3TtsRelease release = config::Qwen3TtsRelease::BASE;
  config::Qwen3TtsModelSize size = config::Qwen3TtsModelSize::SIZE_1B7;
  // Lower-cased names; empty = unknown, so not checked here.
  std::vector<std::string> speakers;
  std::vector<std::string> languages;
  uint32_t maxReferenceSeconds = 0;  // 0 = no limit
};

/** nullopt for a missing or whitespace-only field, as the model reads it. */
std::optional<std::string> nonBlank(const std::optional<std::string>& value);

/** Throw std::invalid_argument for a request the release cannot serve. */
void validateRequest(const domain::tts::TtsRequest& request,
                     const RequestRules& rules);

/** validateRequest() with lists from config.json once it is downloaded. */
class RequestValidator {
 public:
  explicit RequestValidator(const config::TtsConfig& config);

  void validate(const domain::tts::TtsRequest& request) const;

 private:
  static constexpr std::chrono::seconds LOOKUP_INTERVAL{10};

  RequestRules currentRules() const;
  void loadCheckpointLists() const;

  std::string checkpointDir;
  std::string hfModel;
  mutable std::mutex mutex;
  mutable RequestRules rules;
  mutable bool checkpointListsKnown = false;
  mutable std::chrono::steady_clock::time_point lastLookup{};
};

}  // namespace tt::services::qwen3_tts
