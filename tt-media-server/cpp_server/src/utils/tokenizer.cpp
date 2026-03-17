// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

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
    tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  } else if (path.size() >= 7 &&
             path.compare(path.size() - 7, 7, ".model") == 0) {
    tok = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
  } else {
    throw std::runtime_error(
        "[TokenizerUtil] Unknown extension; use .json or .model: " + path);
  }

  if (!tok) {
    throw std::runtime_error(
        "[TokenizerUtil] Failed to create tokenizer from: " + path);
  }

  std::filesystem::path configPath =
      std::filesystem::path(path).parent_path() / "tokenizer_config.json";
  if (std::filesystem::exists(configPath)) {
    cfg = getTokenizerConfig(configPath.string());
  }

  TT_LOG_INFO("[TokenizerUtil] Loaded tokenizer from: {}", path);
}

bool Tokenizer::isLoaded() const { return tok != nullptr; }

std::vector<int> Tokenizer::encode(const std::string& text) const {
  if (!tok) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot encode");
  }
  return tok->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokenIds) const {
  if (!tok) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot decode");
  }
  if (tokenIds.empty()) return "";

  if (cachedSpecialTokenThreshold == -2) {
    cachedSpecialTokenThreshold = specialTokenDecodeThreshold();
  }
  int threshold = cachedSpecialTokenThreshold;
  if (threshold > 0) {
    std::vector<int> filtered;
    filtered.reserve(tokenIds.size());
    for (int id : tokenIds) {
      if (id < threshold) filtered.push_back(id);
    }
    if (filtered.empty()) return "";
    return tok->Decode(filtered);
  }
  return tok->Decode(tokenIds);
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
  static auto tokenizer =
      createTokenizer(config::modelType(), config::tokenizerPath());
  return *tokenizer;
}

}  // namespace tt::utils
