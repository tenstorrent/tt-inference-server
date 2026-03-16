// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "utils/deepseek_tokenizer.hpp"
#include "utils/llama_tokenizer.hpp"
#include "config/settings.hpp"
#include "utils/logger.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace tt::utils {

// ---------------------------------------------------------------------------
// Tokenizer base class
// ---------------------------------------------------------------------------

Tokenizer::Tokenizer(const std::string& path) {
    if (path.empty()) {
        throw std::runtime_error("[TokenizerUtil] Cannot initialize with empty path");
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
    } else if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".model") == 0) {
        tok_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
    } else {
        throw std::runtime_error("[TokenizerUtil] Unknown extension; use .json or .model: " + path);
    }

    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Failed to create tokenizer from: " + path);
    }

    std::filesystem::path config_path = std::filesystem::path(path).parent_path() / "tokenizer_config.json";
    if (std::filesystem::exists(config_path)) {
        cfg_ = get_tokenizer_config(config_path.string());
    }

    TT_LOG_INFO("[TokenizerUtil] Loaded tokenizer from: {}", path);
}

bool Tokenizer::is_loaded() const {
    return tok_ != nullptr;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot encode");
    }
    return tok_->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    if (!tok_) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer not loaded, cannot decode");
    }
    if (token_ids.empty()) return "";

    if (cached_special_token_threshold_ == -2) {
        cached_special_token_threshold_ = special_token_decode_threshold();
    }
    int threshold = cached_special_token_threshold_;
    if (threshold > 0) {
        std::vector<int> filtered;
        filtered.reserve(token_ids.size());
        for (int id : token_ids) {
            if (id < threshold) filtered.push_back(id);
        }
        if (filtered.empty()) return "";
        return tok_->Decode(filtered);
    }
    return tok_->Decode(token_ids);
}

// ---------------------------------------------------------------------------
// Factory + standalone helpers
// ---------------------------------------------------------------------------

std::string tokenizer_dir_for_model(config::ModelType model) {
    switch (model) {
        case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
            return "meta-llama/Llama-3.1-8B-Instruct";
        case config::ModelType::DEEPSEEK_R1_0528:
        default:
            return "deepseek-ai/DeepSeek-R1-0528";
    }
}

std::unique_ptr<Tokenizer> create_tokenizer(config::ModelType model, const std::string& path) {
    switch (model) {
        case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
            return std::make_unique<LlamaTokenizer>(path);
        case config::ModelType::DEEPSEEK_R1_0528:
        default:
            return std::make_unique<DeepseekTokenizer>(path);
    }
}

const Tokenizer& active_tokenizer() {
    static auto tok = create_tokenizer(config::model_type(), config::tokenizer_path());
    return *tok;
}

}  // namespace tt::utils
