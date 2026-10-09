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

/** A request field counts as given only when it holds more than whitespace,
 *  which is how the model's own front-end reads them. */
std::optional<std::string> nonBlank(const std::optional<std::string>& value);

/**
 * Reject, with std::invalid_argument, a request the deployment's release
 * cannot serve. Runs in the parent before the task is queued, so the client
 * gets an HTTP 400 rather than a 200 with an empty WAV:
 *
 *   custom_voice  needs `speaker`; no voice file, no `reference_text`;
 *                 `description` (a delivery instruction) only at 1.7B.
 *   voice_design  needs `description`; no `speaker`, voice file or
 *                 `reference_text`.
 *   base          needs the voice file; no `speaker`; `reference_text`
 *                 optional (present = in-context clone, absent = voice-only
 *                 clone); `description` only with a voice-only clone at 1.7B.
 *
 * `language` is optional everywhere ("Auto" or a language the checkpoint
 * lists).
 */
void validateRequest(const domain::tts::TtsRequest& request,
                     const RequestRules& rules);

/**
 * validateRequest() with the deployment's rules, filling in the speaker and
 * language lists from the checkpoint's config.json once it appears on local
 * disk: a hub checkpoint is downloaded by the worker, after the parent read
 * its config.
 */
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
