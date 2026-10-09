// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "config/qwen3_tts.hpp"

#include <json/json.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include "config/defaults.hpp"

namespace tt::config::qwen3_tts {

namespace {

namespace fs = std::filesystem;

std::string lower(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

std::string trim(std::string_view value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return std::string(value.substr(first, last - first + 1));
}

std::string envOrEmpty(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool isDirectory(const std::string& path) {
  std::error_code ec;
  return !path.empty() && fs::is_directory(path, ec);
}

bool isFile(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec);
}

std::vector<std::string> lowerKeys(const Json::Value& object) {
  std::vector<std::string> keys;
  if (!object.isObject()) return keys;
  for (const auto& name : object.getMemberNames()) {
    keys.push_back(lower(name));
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

/** The snapshot of `repo` in the hub cache holding a config.json; the most
 *  recently written one when several revisions are cached. */
std::string hubSnapshotConfig(const std::string& repo,
                              const std::string& hubCacheDir) {
  if (repo.empty() || hubCacheDir.empty()) return {};
  std::string folder = "models--" + repo;
  for (size_t pos = folder.find('/'); pos != std::string::npos;
       pos = folder.find('/', pos)) {
    folder.replace(pos, 1, "--");
  }
  const fs::path snapshots = fs::path(hubCacheDir) / folder / "snapshots";
  std::error_code ec;
  if (!fs::is_directory(snapshots, ec)) return {};

  fs::path best;
  fs::file_time_type bestTime{};
  for (const auto& entry : fs::directory_iterator(snapshots, ec)) {
    const fs::path candidate = entry.path() / "config.json";
    if (!isFile(candidate)) continue;
    std::error_code timeEc;
    const auto written = fs::last_write_time(candidate, timeEc);
    if (best.empty() || (!timeEc && written > bestTime)) {
      best = candidate;
      bestTime = timeEc ? bestTime : written;
    }
  }
  return best.string();
}

}  // namespace

std::optional<Qwen3TtsRelease> releaseFromString(std::string_view value) {
  const std::string key = lower(trim(value));
  if (key == "base") return Qwen3TtsRelease::BASE;
  if (key == "custom_voice") return Qwen3TtsRelease::CUSTOM_VOICE;
  if (key == "voice_design") return Qwen3TtsRelease::VOICE_DESIGN;
  return std::nullopt;
}

std::optional<Qwen3TtsModelSize> modelSizeFromString(std::string_view value) {
  const std::string key = lower(trim(value));
  if (key == "1b7" || key == "1.7b") return Qwen3TtsModelSize::SIZE_1B7;
  if (key == "0b6" || key == "0.6b") return Qwen3TtsModelSize::SIZE_0B6;
  return std::nullopt;
}

std::optional<CheckpointInfo> readCheckpointConfig(const std::string& path) {
  if (path.empty() || !isFile(path)) return std::nullopt;
  std::ifstream in(path);
  if (!in) return std::nullopt;

  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errors;
  if (!Json::parseFromStream(builder, in, &root, &errors) || !root.isObject()) {
    return std::nullopt;
  }

  CheckpointInfo info;
  info.configPath = path;
  if (root["tts_model_type"].isString()) {
    info.release = releaseFromString(root["tts_model_type"].asString());
  }
  if (root["tts_model_size"].isString()) {
    info.size = modelSizeFromString(root["tts_model_size"].asString());
  }
  const Json::Value& talker = root["talker_config"];
  if (talker.isObject()) {
    info.speakers = lowerKeys(talker["spk_id"]);
    info.languages = lowerKeys(talker["codec_language_id"]);
  }
  return info;
}

std::string huggingFaceHubCacheDir() {
  if (auto hubCache = envOrEmpty("HF_HUB_CACHE"); !hubCache.empty()) {
    return hubCache;
  }
  if (auto hfHome = envOrEmpty("HF_HOME"); !hfHome.empty()) {
    return (fs::path(hfHome) / "hub").string();
  }
  if (auto xdg = envOrEmpty("XDG_CACHE_HOME"); !xdg.empty()) {
    return (fs::path(xdg) / "huggingface" / "hub").string();
  }
  if (auto home = envOrEmpty("HOME"); !home.empty()) {
    return (fs::path(home) / ".cache" / "huggingface" / "hub").string();
  }
  return {};
}

std::string findCheckpointConfig(const std::string& ckptDir,
                                 const std::string& hfModel,
                                 const std::string& hubCacheDir) {
  const std::string ckpt = trim(ckptDir);
  if (!ckpt.empty()) {
    // weights.checkpoint_dir() stops here whether or not the file exists.
    const fs::path config = fs::path(ckpt) / "config.json";
    return isFile(config) ? config.string() : std::string();
  }
  std::string repo = trim(hfModel);
  if (isDirectory(repo)) {
    const fs::path config = fs::path(repo) / "config.json";
    return isFile(config) ? config.string() : std::string();
  }
  if (repo.empty()) repo = defaults::TTS_QWEN3_DEFAULT_HF_MODEL;
  return hubSnapshotConfig(repo, hubCacheDir);
}

std::optional<Qwen3TtsRelease> releaseFromModelName(std::string_view name) {
  const std::string key = lower(name);
  const auto has = [&key](std::string_view part) {
    return key.find(part) != std::string::npos;
  };
  if (has("customvoice") || has("custom_voice") || has("custom-voice")) {
    return Qwen3TtsRelease::CUSTOM_VOICE;
  }
  if (has("voicedesign") || has("voice_design") || has("voice-design")) {
    return Qwen3TtsRelease::VOICE_DESIGN;
  }
  if (has("-base") || has("_base")) {
    return Qwen3TtsRelease::BASE;
  }
  return std::nullopt;
}

std::optional<Qwen3TtsModelSize> modelSizeFromModelName(std::string_view name) {
  const std::string key = lower(name);
  if (key.find("0.6b") != std::string::npos ||
      key.find("0b6") != std::string::npos) {
    return Qwen3TtsModelSize::SIZE_0B6;
  }
  if (key.find("1.7b") != std::string::npos ||
      key.find("1b7") != std::string::npos) {
    return Qwen3TtsModelSize::SIZE_1B7;
  }
  return std::nullopt;
}

ResolvedRelease resolveRelease(const ReleaseInputs& inputs) {
  const std::string ckpt = trim(inputs.ckptDir);
  const std::string hfModel = trim(inputs.hfModel);
  const std::string source =
      !ckpt.empty() ? "QWEN3_TTS_CKPT=" + ckpt
      : !hfModel.empty()
          ? "HF_MODEL=" + hfModel
          : std::string("the default ") + defaults::TTS_QWEN3_DEFAULT_HF_MODEL;
  // Only a hub id carries the release in its name; a local directory may be
  // called anything.
  const std::string nameHint =
      !ckpt.empty()     ? std::string()
      : hfModel.empty() ? std::string(defaults::TTS_QWEN3_DEFAULT_HF_MODEL)
                        : (isDirectory(hfModel) ? std::string() : hfModel);

  std::optional<Qwen3TtsRelease> explicitRelease;
  if (!trim(inputs.explicitRelease).empty()) {
    explicitRelease = releaseFromString(inputs.explicitRelease);
    if (!explicitRelease) {
      throw std::runtime_error(
          "[Config] TTS_QWEN3_RELEASE='" + inputs.explicitRelease +
          "' is not one of: base, custom_voice, voice_design");
    }
  }
  std::optional<Qwen3TtsModelSize> explicitSize;
  if (!trim(inputs.explicitSize).empty()) {
    explicitSize = modelSizeFromString(inputs.explicitSize);
    if (!explicitSize) {
      throw std::runtime_error("[Config] TTS_QWEN3_MODEL_SIZE='" +
                               inputs.explicitSize +
                               "' is not one of: 1b7, 0b6");
    }
  }

  const auto fromCheckpointRelease =
      inputs.checkpoint ? inputs.checkpoint->release : std::nullopt;
  const auto fromCheckpointSize =
      inputs.checkpoint ? inputs.checkpoint->size : std::nullopt;

  if (explicitRelease && fromCheckpointRelease &&
      *explicitRelease != *fromCheckpointRelease) {
    throw std::runtime_error(
        "[Config] TTS_QWEN3_RELEASE=" + toString(*explicitRelease) +
        " contradicts the checkpoint (" + inputs.checkpoint->configPath +
        " says tts_model_type=" + toString(*fromCheckpointRelease) + ")");
  }
  if (explicitSize && fromCheckpointSize &&
      *explicitSize != *fromCheckpointSize) {
    throw std::runtime_error(
        "[Config] TTS_QWEN3_MODEL_SIZE=" + toString(*explicitSize) +
        " contradicts the checkpoint (" + inputs.checkpoint->configPath +
        " says tts_model_size=" + toString(*fromCheckpointSize) + ")");
  }

  ResolvedRelease resolved;
  if (auto release = explicitRelease         ? explicitRelease
                     : fromCheckpointRelease ? fromCheckpointRelease
                                             : releaseFromModelName(nameHint)) {
    resolved.release = *release;
  } else {
    throw std::runtime_error(
        "[Config] Cannot tell which Qwen3-TTS release " + source +
        " is; set TTS_QWEN3_RELEASE to base, custom_voice or voice_design");
  }
  if (auto size = explicitSize         ? explicitSize
                  : fromCheckpointSize ? fromCheckpointSize
                                       : modelSizeFromModelName(nameHint)) {
    resolved.size = *size;
  } else {
    throw std::runtime_error(
        "[Config] Cannot tell the Qwen3-TTS model size of " + source +
        "; set TTS_QWEN3_MODEL_SIZE to 1b7 or 0b6");
  }

  if (resolved.release == Qwen3TtsRelease::VOICE_DESIGN &&
      resolved.size == Qwen3TtsModelSize::SIZE_0B6) {
    throw std::runtime_error(
        "[Config] There is no 0.6B Qwen3-TTS VoiceDesign release (" + source +
        ")");
  }
  return resolved;
}

}  // namespace tt::config::qwen3_tts
