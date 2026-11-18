// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "config/types.hpp"

/** Which Qwen3-TTS release a deployment serves, resolved in the parent. */
namespace tt::config::qwen3_tts {

/** Parse "base" | "custom_voice" | "voice_design" (case-insensitive). */
std::optional<Qwen3TtsRelease> releaseFromString(std::string_view value);

/** Parse "1b7" | "0b6" (also "1.7b" | "0.6b"; case-insensitive). */
std::optional<Qwen3TtsModelSize> modelSizeFromString(std::string_view value);

/** What a checkpoint's config.json says about itself. */
struct CheckpointInfo {
  std::optional<Qwen3TtsRelease> release;
  std::optional<Qwen3TtsModelSize> size;
  // Lower-cased `talker_config.spk_id` keys (empty outside CustomVoice).
  std::vector<std::string> speakers;
  // Lower-cased `talker_config.codec_language_id` keys.
  std::vector<std::string> languages;
  std::string configPath;
};

/** Read a checkpoint's config.json; nullopt when missing or unparsable. */
std::optional<CheckpointInfo> readCheckpointConfig(const std::string& path);

/** The HF hub cache root, resolved like huggingface_hub does. */
std::string huggingFaceHubCacheDir();

/** Path of the config.json the worker will load, or "" if not on disk yet. */
std::string findCheckpointConfig(const std::string& ckptDir,
                                 const std::string& hfModel,
                                 const std::string& hubCacheDir);

/** Guess the release and size from a hub id's name. */
std::optional<Qwen3TtsRelease> releaseFromModelName(std::string_view name);
std::optional<Qwen3TtsModelSize> modelSizeFromModelName(std::string_view name);

struct ReleaseInputs {
  // TTS_QWEN3_RELEASE / TTS_QWEN3_MODEL_SIZE; empty = derive.
  std::string explicitRelease{};
  std::string explicitSize{};
  // QWEN3_TTS_CKPT / HF_MODEL as the deployment set them.
  std::string ckptDir{};
  std::string hfModel{};
  // The checkpoint's own config.json, when it could be found.
  std::optional<CheckpointInfo> checkpoint{};
};

struct ResolvedRelease {
  Qwen3TtsRelease release = Qwen3TtsRelease::BASE;
  Qwen3TtsModelSize size = Qwen3TtsModelSize::SIZE_1B7;
};

/** Release and size: explicit values, then config.json, then the hub id. */
ResolvedRelease resolveRelease(const ReleaseInputs& inputs);

}  // namespace tt::config::qwen3_tts
