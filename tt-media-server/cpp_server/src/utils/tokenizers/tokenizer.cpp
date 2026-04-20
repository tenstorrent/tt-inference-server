// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/tokenizer.hpp"

#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "config/settings.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/deepseek_tokenizer.hpp"
#include "utils/tokenizers/llama_tokenizer.hpp"

namespace tt::utils::tokenizers {

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
    tok_ = ::tokenizers::Tokenizer::FromBlobJSON(blob);
    specialTokenIds_ = parseSpecialTokenIds(blob);
  } else if (path.size() >= 7 &&
             path.compare(path.size() - 7, 7, ".model") == 0) {
    tok_ = ::tokenizers::Tokenizer::FromBlobSentencePiece(blob);
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
              path, specialTokenIds_.size());
}

bool Tokenizer::isLoaded() const { return tok_ != nullptr; }

std::vector<int> Tokenizer::encode(const std::string& text) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot encode");
  }
  return tok_->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokenIds,
                              bool skipSpecialTokens) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot decode");
  }
  if (tokenIds.empty()) return "";

  // Fast path: no special tokens to filter
  if (!skipSpecialTokens || specialTokenIds_.empty()) {
    return tok_->Decode(tokenIds);
  }

  // Fast path: single token, check if special
  if (tokenIds.size() == 1) {
    if (specialTokenIds_.find(tokenIds[0]) != specialTokenIds_.end()) {
      return "";  // Special token, skip it
    }
    return tok_->Decode(tokenIds);
  }

  // Slow path: multiple tokens, filter special tokens
  std::vector<int> filtered;
  filtered.reserve(tokenIds.size());
  for (int id : tokenIds) {
    if (specialTokenIds_.find(id) == specialTokenIds_.end()) {
      filtered.push_back(id);
    }
  }
  if (filtered.empty()) return "";
  return tok_->Decode(filtered);
}

std::vector<std::string> Tokenizer::getEncodedVocab() const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot get vocabulary");
  }
  size_t size = tok_->GetVocabSize();
  std::vector<std::string> vocab(size);
  for (size_t i = 0; i < size; ++i) {
    vocab[i] = tok_->IdToToken(static_cast<int32_t>(i));
  }
  return vocab;
}

// ---------------------------------------------------------------------------
// StreamDecoder
// ---------------------------------------------------------------------------

namespace {

// U+FFFD in UTF-8 is the 3-byte sequence EF BF BD
bool endsWithReplacementChar(const std::string& s) {
  return s.size() >= 3 && static_cast<unsigned char>(s[s.size() - 3]) == 0xEF &&
         static_cast<unsigned char>(s[s.size() - 2]) == 0xBF &&
         static_cast<unsigned char>(s[s.size() - 1]) == 0xBD;
}

}  // namespace

Tokenizer::StreamDecoder::StreamDecoder(const Tokenizer& tokenizer,
                                        bool skipSpecialTokens)
    : tokenizer_(tokenizer), skipSpecialTokens_(skipSpecialTokens) {}

std::string Tokenizer::StreamDecoder::step(int tokenId) {
  if (pending_.empty()) {
    std::string decoded = tokenizer_.decode({tokenId}, skipSpecialTokens_);
    if (!endsWithReplacementChar(decoded)) {
      return decoded;
    }
    pending_.push_back(tokenId);
    return "";
  }

  pending_.push_back(tokenId);
  std::string decoded = tokenizer_.decode(pending_, skipSpecialTokens_);
  if (!endsWithReplacementChar(decoded)) {
    pending_.clear();
    return decoded;
  }
  return "";
}

std::string Tokenizer::StreamDecoder::flush() {
  if (pending_.empty()) return "";
  std::string decoded = tokenizer_.decode(pending_, skipSpecialTokens_);
  pending_.clear();
  return decoded;
}

std::unique_ptr<Tokenizer::StreamDecoder> Tokenizer::createStreamDecoder(
    bool skipSpecialTokens) const {
  return std::make_unique<StreamDecoder>(*this, skipSpecialTokens);
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

}  // namespace tt::utils::tokenizers
