// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "config/settings.hpp"
#include "utils/deepseek_tokenizer.hpp"
#include "utils/llama_tokenizer.hpp"
#include "utils/logger.hpp"

namespace tt::utils {

// ---------------------------------------------------------------------------
// Tokenizer base class
// ---------------------------------------------------------------------------

namespace {

std::unordered_set<int> parseSpecialTokenIds(const std::string& jsonBlob) {
  std::unordered_set<int> ids;
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream iss(jsonBlob);
  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    return ids;
  }
  const Json::Value& added = root["added_tokens"];
  if (!added.isArray()) return ids;
  for (const auto& tok : added) {
    if (tok.isMember("special") && tok["special"].asBool() &&
        tok.isMember("id")) {
      ids.insert(tok["id"].asInt());
    }
  }
  return ids;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  if (path.empty()) {
    throw std::runtime_error(
        "[TokenizerUtil] Cannot initialize with empty path");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("[TokenizerUtil] Failed to open: " + path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string blob = ss.str();
  f.close();

  if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".json") == 0) {
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    special_token_ids_ = parseSpecialTokenIds(blob);
  } else if (path.size() >= 7 &&
             path.compare(path.size() - 7, 7, ".model") == 0) {
    tok_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
  } else {
    throw std::runtime_error(
        "[TokenizerUtil] Unknown extension; use .json or .model: " + path);
  }

  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Failed to create tokenizer from: " + path);
  }

  std::filesystem::path configPath =
      std::filesystem::path(path).parent_path() / "tokenizer_config.json";
  if (std::filesystem::exists(configPath)) {
    cfg_ = getTokenizerConfig(configPath.string());
  }

  TT_LOG_INFO("[TokenizerUtil] Loaded tokenizer from: {} ({} special tokens)",
              path, special_token_ids_.size());
}

bool Tokenizer::isLoaded() const { return tok_ != nullptr; }

std::vector<int> Tokenizer::encode(const std::string& text) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot encode");
  }
  return tok_->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokenIds) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot decode");
  }
  if (tokenIds.empty()) return "";

  if (special_token_ids_.empty()) {
    return tok_->Decode(tokenIds);
  }

  std::vector<int> filtered;
  filtered.reserve(tokenIds.size());
  for (int id : tokenIds) {
    if (special_token_ids_.find(id) == special_token_ids_.end()) {
      filtered.push_back(id);
    }
  }
  if (filtered.empty()) return "";
  return tok_->Decode(filtered);
}

// ---------------------------------------------------------------------------
// Factory + standalone helpers
// ---------------------------------------------------------------------------

std::string tokenizerDirForModel(config::ModelType model) {
  switch (model) {
    case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      return "meta-llama/Llama-3.1-8B-Instruct";
    case config::ModelType::DEEPSEEK_R1_0528:
    default:
      return "deepseek-ai/DeepSeek-R1-0528";
  }
}

std::unique_ptr<Tokenizer> createTokenizer(config::ModelType model,
                                           const std::string& path) {
  switch (model) {
    case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      return std::make_unique<LlamaTokenizer>(path);
    case config::ModelType::DEEPSEEK_R1_0528:
    default:
      return std::make_unique<DeepseekTokenizer>(path);
  }
}

const Tokenizer& activeTokenizer() {
  static auto tok =
      createTokenizer(config::modelType(), config::tokenizerPath());
  return *tok;
}

}  // namespace tt::utils
