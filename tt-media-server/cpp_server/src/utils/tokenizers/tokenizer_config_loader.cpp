// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <json/json.h>

#include <fstream>
#include <sstream>

#include "config/settings.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tokenizers {

namespace {

std::string extractToken(const Json::Value& v) {
  if (v.isNull() || !v) return {};
  if (v.isString()) return v.asString();
  if (v.isObject() && v.isMember("content")) return v["content"].asString();
  return {};
}

bool loadFromPath(const std::string& path, TokenizerConfig& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;

  std::stringstream ss;
  ss << f.rdbuf();
  f.close();
  std::string content = ss.str();

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream iss(content);
  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    return false;
  }

  if (root.isMember("bos_token"))
    out.bos_token = extractToken(root["bos_token"]);
  if (root.isMember("eos_token"))
    out.eos_token = extractToken(root["eos_token"]);
  if (root.isMember("pad_token"))
    out.pad_token = extractToken(root["pad_token"]);
  if (root.isMember("unk_token"))
    out.unk_token = extractToken(root["unk_token"]);
  if (root.isMember("chat_template") && root["chat_template"].isString()) {
    out.chat_template = root["chat_template"].asString();
  }
  if (root.isMember("add_bos_token") && root["add_bos_token"].isBool()) {
    out.add_bos_token = root["add_bos_token"].asBool();
  }
  if (root.isMember("add_eos_token") && root["add_eos_token"].isBool()) {
    out.add_eos_token = root["add_eos_token"].asBool();
  }
  return true;
}

}  // namespace

static TokenizerConfig loadAndValidate(const std::string& path) {
  if (path.empty()) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer config not found (tokenizer_config.json "
        "missing)");
  }
  TokenizerConfig cfg;
  if (!loadFromPath(path, cfg)) {
    throw std::runtime_error(
        "[TokenizerUtil] Failed to load tokenizer config: " + path);
  }
  // A tokenizer with no bos/eos token has nothing to add, even if the flag is
  // set (which it is by default for add_bos_token). Some models legitimately
  // ship no bos_token — e.g. GLM opens turns with [gMASK]<sop> in its chat
  // template rather than a BOS token. HF treats a missing token as "don't add
  // one"; mirror that (disable the flag with a warning) instead of aborting, so
  // such models load. The tokenizers already no-op on an empty token string.
  if (cfg.add_bos_token && cfg.bos_token.empty()) {
    TT_LOG_WARN(
        "[TokenizerUtil] add_bos_token set but no bos_token in {}; disabling "
        "bos prepend",
        path);
    cfg.add_bos_token = false;
  }
  if (cfg.add_eos_token && cfg.eos_token.empty()) {
    TT_LOG_WARN(
        "[TokenizerUtil] add_eos_token set but no eos_token in {}; disabling "
        "eos append",
        path);
    cfg.add_eos_token = false;
  }
  return cfg;
}

TokenizerConfig getTokenizerConfig() {
  static TokenizerConfig cached =
      loadAndValidate(tt::config::tokenizerConfigPath());
  return cached;
}

TokenizerConfig getTokenizerConfig(const std::string& configPath) {
  return loadAndValidate(configPath);
}

}  // namespace tt::utils::tokenizers
