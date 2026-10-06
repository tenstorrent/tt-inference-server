// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "config/types.hpp"

/**
 * Which Qwen3-TTS release a deployment serves, resolved in the parent process
 * from the same sources tt-metal's `weights.checkpoint_dir()` reads
 * (QWEN3_TTS_CKPT, then HF_MODEL, then the default Base repo), so requests can
 * be validated before any audio streams. The worker re-checks the answer
 * against the loaded checkpoint at warmup.
 */
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

/** The HF hub cache root, resolved like huggingface_hub does: HF_HUB_CACHE,
 *  else HF_HOME/hub, else XDG_CACHE_HOME/huggingface/hub, else
 *  ~/.cache/huggingface/hub. */
std::string huggingFaceHubCacheDir();

/**
 * Path of the config.json the worker will load, or "" when it is not on local
 * disk yet (a hub checkpoint that has not been downloaded). Mirrors
 * `weights.checkpoint_dir()`: `ckptDir` first, then `hfModel` as a directory,
 * then a snapshot of `hfModel` (default repo when empty) in `hubCacheDir`.
 */
std::string findCheckpointConfig(const std::string& ckptDir,
                                 const std::string& hfModel,
                                 const std::string& hubCacheDir);

/** Guess the release and size from a hub id such as
 *  "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice". */
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

/**
 * Release and size for the deployment. Explicit values win, then the
 * checkpoint's config.json, then the hub id's name; with neither QWEN3_TTS_CKPT
 * nor HF_MODEL set it is tt-metal's default, 1.7B Base. Throws
 * std::runtime_error when it cannot be derived, when an explicit value
 * contradicts the checkpoint, or for the release that does not exist (0.6B
 * VoiceDesign).
 */
ResolvedRelease resolveRelease(const ReleaseInputs& inputs);

}  // namespace tt::config::qwen3_tts
