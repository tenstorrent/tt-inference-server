// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/qwen3_tts_request_validator.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string_view>

#include "config/qwen3_tts.hpp"
#include "utils/logger.hpp"

namespace tt::services::qwen3_tts {

namespace {

using config::Qwen3TtsModelSize;
using config::Qwen3TtsRelease;

constexpr std::string_view AUTO_LANGUAGE = "auto";

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string join(const std::vector<std::string>& names) {
  std::string out;
  for (const auto& name : names) {
    if (!out.empty()) out += ", ";
    out += name;
  }
  return out;
}

bool contains(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

std::string releaseName(const RequestRules& rules) {
  const char* size =
      rules.size == Qwen3TtsModelSize::SIZE_0B6 ? "0.6B" : "1.7B";
  switch (rules.release) {
    case Qwen3TtsRelease::BASE:
      return std::string("Qwen3-TTS ") + size + " Base";
    case Qwen3TtsRelease::CUSTOM_VOICE:
      return std::string("Qwen3-TTS ") + size + " CustomVoice";
    case Qwen3TtsRelease::VOICE_DESIGN:
      return std::string("Qwen3-TTS ") + size + " VoiceDesign";
  }
  return "Qwen3-TTS";
}

[[noreturn]] void reject(const RequestRules& rules, const std::string& why) {
  throw std::invalid_argument("This server runs " + releaseName(rules) + ": " +
                              why);
}

void validateCustomVoice(const domain::tts::TtsRequest& request,
                         const RequestRules& rules) {
  const auto speaker = nonBlank(request.speaker);
  if (!speaker) {
    reject(rules, "'speaker' is required" +
                      (rules.speakers.empty()
                           ? std::string()
                           : " (one of: " + join(rules.speakers) + ")"));
  }
  if (!rules.speakers.empty() && !contains(rules.speakers, lower(*speaker))) {
    reject(rules, "unknown speaker '" + *speaker +
                      "'; this checkpoint offers: " + join(rules.speakers));
  }
  if (request.voiceSample.has_value()) {
    reject(rules,
           "it speaks as a built-in speaker and cannot clone a voice; drop "
           "the voice file");
  }
  if (nonBlank(request.referenceText)) {
    reject(rules,
           "'reference_text' is the transcript of a voice file, which this "
           "release does not take");
  }
  if (nonBlank(request.description) &&
      rules.size == Qwen3TtsModelSize::SIZE_0B6) {
    reject(rules,
           "the 0.6B checkpoints take no instruction; drop 'description'");
  }
}

void validateVoiceDesign(const domain::tts::TtsRequest& request,
                         const RequestRules& rules) {
  if (!nonBlank(request.description)) {
    reject(rules,
           "'description' is required: it describes the voice to speak in");
  }
  if (nonBlank(request.speaker)) {
    reject(rules,
           "it designs a voice from 'description' and has no built-in "
           "speakers; drop 'speaker'");
  }
  if (request.voiceSample.has_value()) {
    reject(rules,
           "it designs a voice from 'description' and cannot clone one; drop "
           "the voice file");
  }
  if (nonBlank(request.referenceText)) {
    reject(rules,
           "'reference_text' is the transcript of a voice file, which this "
           "release does not take");
  }
}

void validateBase(const domain::tts::TtsRequest& request,
                  const RequestRules& rules) {
  if (!request.voiceSample.has_value()) {
    reject(rules,
           "it clones a voice, so send the voice to clone as a WAV file in a "
           "multipart/form-data request (add 'reference_text', the clip's "
           "transcript, for the closer in-context clone)");
  }
  if (nonBlank(request.speaker)) {
    reject(rules,
           "it has no built-in speakers; drop 'speaker' and clone from the "
           "voice file");
  }
  if (nonBlank(request.description)) {
    if (nonBlank(request.referenceText)) {
      reject(rules,
             "an in-context clone (with 'reference_text') cannot take an "
             "instruction; drop 'description', or drop 'reference_text' to "
             "clone from the voice alone");
    }
    if (rules.size == Qwen3TtsModelSize::SIZE_0B6) {
      reject(rules,
             "the 0.6B checkpoints take no instruction; drop 'description'");
    }
  }

  const auto& sample = *request.voiceSample;
  if (rules.maxReferenceSeconds > 0 && sample.sampleRateHz > 0 &&
      sample.channels > 0) {
    const double seconds = static_cast<double>(sample.wavPcm.size()) /
                           sample.channels / sample.sampleRateHz;
    if (seconds > rules.maxReferenceSeconds) {
      reject(rules, "the voice file is " +
                        std::to_string(static_cast<int>(seconds)) +
                        " s long; send at most " +
                        std::to_string(rules.maxReferenceSeconds) +
                        " s (3 to 10 s of clean speech is the useful range)");
    }
  }
}

}  // namespace

std::optional<std::string> nonBlank(const std::optional<std::string>& value) {
  if (!value.has_value() || trim(*value).empty()) {
    return std::nullopt;
  }
  return value;
}

void validateRequest(const domain::tts::TtsRequest& request,
                     const RequestRules& rules) {
  if (trim(request.text).empty()) {
    throw std::invalid_argument("'text' must not be empty");
  }

  switch (rules.release) {
    case Qwen3TtsRelease::CUSTOM_VOICE:
      validateCustomVoice(request, rules);
      break;
    case Qwen3TtsRelease::VOICE_DESIGN:
      validateVoiceDesign(request, rules);
      break;
    case Qwen3TtsRelease::BASE:
      validateBase(request, rules);
      break;
  }

  if (const auto language = nonBlank(request.language)) {
    const std::string key = lower(trim(*language));
    if (key != AUTO_LANGUAGE && !rules.languages.empty() &&
        !contains(rules.languages, key)) {
      reject(rules, "unsupported language '" + *language +
                        "'; use Auto or one of: " + join(rules.languages));
    }
  }
}

RequestValidator::RequestValidator(const config::TtsConfig& config) {
  rules.release = config.qwen3Release;
  rules.size = config.qwen3ModelSize;
  rules.speakers = config.qwen3Speakers;
  rules.languages = config.qwen3Languages;
  rules.maxReferenceSeconds = config.qwen3MaxReferenceSeconds;
  checkpointListsKnown = !config.qwen3Languages.empty();
  checkpointDir = config.qwen3CheckpointDir;
  hfModel = config.qwen3HfModel;
}

void RequestValidator::validate(const domain::tts::TtsRequest& request) const {
  validateRequest(request, currentRules());
}

RequestRules RequestValidator::currentRules() const {
  std::lock_guard<std::mutex> lock(mutex);
  if (!checkpointListsKnown) {
    // Throttled: until the worker has downloaded the checkpoint, each attempt
    // is a few failed stat() calls, but there is no reason to repeat them per
    // request.
    const auto now = std::chrono::steady_clock::now();
    if (now - lastLookup >= LOOKUP_INTERVAL) {
      lastLookup = now;
      loadCheckpointLists();
    }
  }
  return rules;
}

void RequestValidator::loadCheckpointLists() const {
  namespace qwen3 = config::qwen3_tts;
  const auto checkpoint =
      qwen3::readCheckpointConfig(qwen3::findCheckpointConfig(
          checkpointDir, hfModel, qwen3::huggingFaceHubCacheDir()));
  if (!checkpoint || checkpoint->languages.empty()) {
    return;
  }
  checkpointListsKnown = true;
  if (checkpoint->release && *checkpoint->release != rules.release) {
    // The worker refuses to warm up on a mismatch, so this deployment serves
    // nothing; keep the configured rules rather than adopt the file's.
    TT_LOG_ERROR(
        "[Qwen3TtsRequestValidator] {} is a {} checkpoint but this deployment "
        "is configured as {}",
        checkpoint->configPath, config::toString(*checkpoint->release),
        config::toString(rules.release));
    return;
  }
  rules.speakers = checkpoint->speakers;
  rules.languages = checkpoint->languages;
  TT_LOG_INFO(
      "[Qwen3TtsRequestValidator] Validating speakers ({}) and languages ({}) "
      "against {}",
      rules.speakers.size(), rules.languages.size(), checkpoint->configPath);
}

}  // namespace tt::services::qwen3_tts
