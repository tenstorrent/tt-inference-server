// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/tokenizer.hpp"

#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "config/settings.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/deepseek_tokenizer.hpp"
#include "utils/tokenizers/llama_tokenizer.hpp"

namespace tt::utils::tokenizers {

// ---------------------------------------------------------------------------
// Tokenizer base class
// ---------------------------------------------------------------------------

namespace {

std::unordered_set<uint32_t> parseSpecialTokenIds(const std::string& jsonBlob) {
  std::unordered_set<uint32_t> ids;
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
      ids.insert(static_cast<uint32_t>(tok["id"].asInt()));
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
  } else if (path.size() >= 6 &&
             path.compare(path.size() - 6, 6, ".model") == 0) {
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

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot encode");
  }
  // tokenizers-cpp emits int32_t ids; token ids are non-negative and well
  // within uint32_t range, so this widening conversion is lossless.
  const std::vector<int32_t> ids = tok_->Encode(text);
  return std::vector<uint32_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokenIds,
                              bool skipSpecialTokens) const {
  if (!tok_) {
    throw std::runtime_error(
        "[TokenizerUtil] Tokenizer not loaded, cannot decode");
  }
  if (tokenIds.empty()) return "";

  // tokenizers-cpp Decode takes int32_t ids; convert without any value change
  // (token ids fit in both types).
  auto toI32 = [](const std::vector<uint32_t>& v) {
    return std::vector<int32_t>(v.begin(), v.end());
  };

  // Fast path: no special tokens to filter
  if (!skipSpecialTokens || specialTokenIds_.empty()) {
    return tok_->Decode(toI32(tokenIds));
  }

  // Fast path: single token, check if special
  if (tokenIds.size() == 1) {
    if (specialTokenIds_.find(tokenIds[0]) != specialTokenIds_.end()) {
      return "";  // Special token, skip it
    }
    return tok_->Decode(toI32(tokenIds));
  }

  // Slow path: multiple tokens, filter special tokens
  std::vector<int32_t> filtered;
  filtered.reserve(tokenIds.size());
  for (uint32_t id : tokenIds) {
    if (specialTokenIds_.find(id) == specialTokenIds_.end()) {
      filtered.push_back(static_cast<int32_t>(id));
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

std::string Tokenizer::StreamDecoder::step(uint32_t tokenId) {
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
    case config::ModelType::KIMI_K2_6:
      return "moonshotai/Kimi-K2.6";
    case config::ModelType::KIMI_K2_7_CODE:
      return "moonshotai/Kimi-K2.7-Code";
    case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      return "meta-llama/Llama-3.1-8B-Instruct";
    case config::ModelType::GPT_OSS_120B:
      return "openai/gpt-oss-120b";
    case config::ModelType::MINIMAX_M2_7:
      return "MiniMaxAI/MiniMax-M2.7";
    case config::ModelType::MINIMAX_M3:
      return "MiniMaxAI/MiniMax-M3";
    case config::ModelType::GLM_5_1:
      return "zai-org/GLM-5.1";
    case config::ModelType::GLM_5_2:
      return "zai-org/GLM-5.2";
    case config::ModelType::DEEPSEEK_V4_PRO:
      return "deepseek-ai/DeepSeek-V4-Pro";
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
    case config::ModelType::KIMI_K2_6:
    case config::ModelType::KIMI_K2_7_CODE:
      // Kimi K2.6 / K2.7-Code use model-specific files, but currently share the
      // same chat-template/tool-call behavior as DeepSeek until a dedicated
      // Kimi tokenizer implementation is added.
      return std::make_unique<DeepseekTokenizer>(path);
    case config::ModelType::GPT_OSS_120B:
    case config::ModelType::MINIMAX_M2_7:
    case config::ModelType::MINIMAX_M3:
    case config::ModelType::GLM_5_1:
    case config::ModelType::GLM_5_2:
      // These load their own model-specific files but currently reuse the
      // DeepSeek chat-template/tool-call behavior until a dedicated tokenizer
      // implementation is added.
      return std::make_unique<DeepseekTokenizer>(path);
    case config::ModelType::DEEPSEEK_R1_0528:
    case config::ModelType::DEEPSEEK_V4_PRO:
    default:
      return std::make_unique<DeepseekTokenizer>(path);
  }
}

const Tokenizer& activeTokenizer() {
  // Per-thread instance: tokenizers-cpp's encode/decode mutate hidden state
  // inside the wrapper, so a single instance is not safe to share across
  // threads. See tenstorrent/tt-inference-server#3179.
  thread_local auto tok =
      createTokenizer(config::modelType(), config::tokenizerPath());
  return *tok;
}

// Mirrors what each Tokenizer subclass returns from modelName() /
// stopTokenIds() / assistantHeaderSequence(). Add an entry here whenever
// a new ModelType is added; staticInfoFor() throws otherwise.

namespace {

const StaticTokenizerInfo& deepseekR1Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"deepseek-ai/DeepSeek-R1-0528",
      /*stopTokenIds=*/{1},
      /*eosTokenId=*/1,
      /*assistantHeaderSequence=*/{128804},
      /*thinkStartTokenId=*/128798,
      /*thinkEndTokenId=*/128799,
  };
  return kInfo;
}

const StaticTokenizerInfo& llama31Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"meta-llama/Llama-3.1-8B-Instruct",
      /*stopTokenIds=*/{128001, 128008, 128009},
      /*eosTokenId=*/128001,
      /*assistantHeaderSequence=*/{128006, 78191, 128007, 271},
  };
  return kInfo;
}

const StaticTokenizerInfo& kimiK26Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"moonshotai/Kimi-K2.6",
      /*stopTokenIds=*/{163586},
      /*eosTokenId=*/163585,
      /*assistantHeaderSequence=*/{163588},
      /*thinkStartTokenId=*/163606,  // <think>
      /*thinkEndTokenId=*/163607,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched Kimi-K2.7-Code tokenizer; identical layout
// to Kimi-K2.6 (config.json eos_token_id 163586 <|im_end|>, [EOS] 163585,
// <|im_assistant|> 163588, <think>/</think> 163606/163607). model_type is
// "kimi_k25", so discovery reuses the existing kimi_k25 parser branch.
const StaticTokenizerInfo& kimiK27CodeInfo() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"moonshotai/Kimi-K2.7-Code",
      /*stopTokenIds=*/{163586},
      /*eosTokenId=*/163585,
      /*assistantHeaderSequence=*/{163588},
      /*thinkStartTokenId=*/163606,  // <think>
      /*thinkEndTokenId=*/163607,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched o200k_harmony tokenizer. gpt-oss uses the
// Harmony channel format rather than <think> tags, so no think tokens; the
// assistant turn ends on <|return|> (200002) and a tool call on <|call|>
// (200012). assistantHeaderSequence is left empty (the Harmony header is
// multi-token and handled by the chat template, not a fixed prefix here).
const StaticTokenizerInfo& gptOss120bInfo() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"openai/gpt-oss-120b",
      // generation_config.json eos_token_id: [200002, 199999, 200012] =
      // <|return|> (final turn), <|endoftext|>, <|call|> (tool call).
      /*stopTokenIds=*/{199999, 200012},
      /*eosTokenId=*/200002,  // <|return|> (primary, per config.json)
      /*assistantHeaderSequence=*/{},
  };
  return kInfo;
}

// IDs verified against the fetched MiniMax-M2.7 tokenizer.
const StaticTokenizerInfo& minimaxM27Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"MiniMaxAI/MiniMax-M2.7",
      /*stopTokenIds=*/{},
      /*eosTokenId=*/200020,  // [e~[
      /*assistantHeaderSequence=*/{},
      /*thinkStartTokenId=*/200050,  // <think>
      /*thinkEndTokenId=*/200051,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched MiniMax-M3 tokenizer. Same special-token
// layout as M2.7 (eos 200020, <think>/</think> 200050/200051); the top-level
// config.json carries no eos_token_id, so discovery.cpp publishes the
// generation_config.json that contains it.
const StaticTokenizerInfo& minimaxM3Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"MiniMaxAI/MiniMax-M3",
      /*stopTokenIds=*/{},
      /*eosTokenId=*/200020,  // [e~[
      /*assistantHeaderSequence=*/{},
      /*thinkStartTokenId=*/200050,  // <think>
      /*thinkEndTokenId=*/200051,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched GLM-5.1 tokenizer (added_tokens in
// tokenizer.json). Identical special-token layout to GLM-5.2: same eos set
// [154820, 154827, 154829] and <think>/</think> 154841/154842, and the same
// glm_moe_dsa model_type (so discovery reuses the glm45/glm47 parser branch).
const StaticTokenizerInfo& glm51Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"zai-org/GLM-5.1",
      // config.json / generation_config.json eos_token_id:
      // [154820, 154827, 154829] = <|endoftext|>, <|user|>, <|observation|>.
      /*stopTokenIds=*/{154827, 154829},
      /*eosTokenId=*/154820,  // <|endoftext|> (primary; also pad + tokenizer
                              // eos)
      /*assistantHeaderSequence=*/{},
      /*thinkStartTokenId=*/154841,  // <think>
      /*thinkEndTokenId=*/154842,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched GLM-5.2 tokenizer (added_tokens in
// tokenizer.json). GLM uses <think>...</think> reasoning and <tool_call>/
// <arg_key>/<arg_value> tool calls.
const StaticTokenizerInfo& glm52Info() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"zai-org/GLM-5.2",
      // config.json / generation_config.json eos_token_id:
      // [154820, 154827, 154829] = <|endoftext|>, <|user|>, <|observation|>.
      /*stopTokenIds=*/{154827, 154829},
      /*eosTokenId=*/154820,  // <|endoftext|> (primary; also pad + tokenizer
                              // eos)
      /*assistantHeaderSequence=*/{},
      /*thinkStartTokenId=*/154841,  // <think>
      /*thinkEndTokenId=*/154842,    // </think>
  };
  return kInfo;
}

// IDs verified against the fetched DeepSeek-V4-Pro tokenizer (added_tokens in
// tokenizer.json). Same DeepSeek-R1 special-token layout (eos 1, assistant
// header 128804) but the <think>/</think> ids differ from R1-0528
// (128821/128822 vs 128798/128799), so it needs its own static info.
const StaticTokenizerInfo& deepseekV4ProInfo() {
  static const StaticTokenizerInfo kInfo{
      /*modelName=*/"deepseek-ai/DeepSeek-V4-Pro",
      /*stopTokenIds=*/{1},
      /*eosTokenId=*/1,  // <｜end▁of▁sentence｜> (config + generation_config)
      /*assistantHeaderSequence=*/{128804},  // <｜Assistant｜>
      /*thinkStartTokenId=*/128821,          // <think>
      /*thinkEndTokenId=*/128822,            // </think>
  };
  return kInfo;
}

}  // namespace

const StaticTokenizerInfo& staticInfoFor(config::ModelType model) {
  switch (model) {
    case config::ModelType::DEEPSEEK_R1_0528:
      return deepseekR1Info();
    case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
      return llama31Info();
    case config::ModelType::KIMI_K2_6:
      return kimiK26Info();
    case config::ModelType::KIMI_K2_7_CODE:
      return kimiK27CodeInfo();
    case config::ModelType::GPT_OSS_120B:
      return gptOss120bInfo();
    case config::ModelType::MINIMAX_M2_7:
      return minimaxM27Info();
    case config::ModelType::MINIMAX_M3:
      return minimaxM3Info();
    case config::ModelType::GLM_5_1:
      return glm51Info();
    case config::ModelType::GLM_5_2:
      return glm52Info();
    case config::ModelType::DEEPSEEK_V4_PRO:
      return deepseekV4ProInfo();
  }
  throw std::invalid_argument(
      "tokenizers::staticInfoFor: no static info registered for ModelType " +
      std::to_string(static_cast<int>(model)));
}

const StaticTokenizerInfo& staticInfo() {
  return staticInfoFor(config::modelType());
}

std::pair<uint32_t, uint32_t> thinkTokenIdsFor(config::ModelType model) {
  const auto& info = staticInfoFor(model);
  return {info.thinkStartTokenId, info.thinkEndTokenId};
}

std::pair<uint32_t, uint32_t> thinkTokenIds() {
  return thinkTokenIdsFor(config::modelType());
}

}  // namespace tt::utils::tokenizers
