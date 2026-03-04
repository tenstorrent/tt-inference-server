// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

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

    std::cout << "[TokenizerUtil] Loaded tokenizer from: " << path << std::endl;
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

    int threshold = special_token_decode_threshold();
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
// DeepseekTokenizer
// ---------------------------------------------------------------------------

static const char* DS_USER_TAG = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
static const char* DS_ASSISTANT_TAG = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";

class DeepseekTokenizer final : public Tokenizer {
public:
    using Tokenizer::Tokenizer;

    std::string model_name() const override { return "deepseek-ai/DeepSeek-V3"; }
    int special_token_decode_threshold() const override { return -1; }
    std::vector<int64_t> stop_token_ids() const override { return {1}; }

    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt) const override {

        static TokenizerConfig cfg = get_tokenizer_config();
        std::ostringstream out;

        if (cfg.add_bos_token) out << cfg.bos_token;

        for (const auto& m : messages) {
            if (m.role == "system") out << m.content;
        }

        for (const auto& m : messages) {
            if (m.role == "system") continue;
            if (m.role == "user") {
                out << DS_USER_TAG << m.content;
            } else if (m.role == "assistant") {
                out << DS_ASSISTANT_TAG << m.content;
                if (cfg.add_eos_token) out << cfg.eos_token;
            }
        }

        if (add_generation_prompt) {
            out << DS_ASSISTANT_TAG;
        }
        return out.str();
    }
};

// ---------------------------------------------------------------------------
// LlamaTokenizer
// ---------------------------------------------------------------------------

static const char* LLAMA_HEADER_START = "<|start_header_id|>";
static const char* LLAMA_HEADER_END = "<|end_header_id|>";
static const char* LLAMA_EOT = "<|eot_id|>";
static const char* LLAMA_SYSTEM_PREAMBLE =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

class LlamaTokenizer final : public Tokenizer {
public:
    using Tokenizer::Tokenizer;

    std::string model_name() const override { return "meta-llama/Llama-3.1-8B-Instruct"; }
    int special_token_decode_threshold() const override { return 128000; }
    std::vector<int64_t> stop_token_ids() const override { return {128001, 128008, 128009}; }

    std::string apply_chat_template(
        const std::vector<domain::ChatMessage>& messages,
        bool add_generation_prompt) const override {

        static TokenizerConfig cfg = get_tokenizer_config();
        std::ostringstream out;

        std::string system_content;
        for (const auto& m : messages) {
            if (m.role == "system") {
                if (!system_content.empty()) system_content += "\n\n";
                system_content += m.content;
            }
        }

        if (cfg.add_bos_token) out << cfg.bos_token;

        out << LLAMA_HEADER_START << "system" << LLAMA_HEADER_END << "\n\n"
            << LLAMA_SYSTEM_PREAMBLE
            << system_content
            << LLAMA_EOT;

        for (const auto& m : messages) {
            if (m.role == "system") continue;
            std::string role = m.role.empty() ? "user" : m.role;
            out << LLAMA_HEADER_START << role << LLAMA_HEADER_END << "\n\n"
                << m.content << LLAMA_EOT;
        }

        if (add_generation_prompt) {
            out << LLAMA_HEADER_START << "assistant" << LLAMA_HEADER_END << "\n\n";
        }
        return out.str();
    }
};

// ---------------------------------------------------------------------------
// Factory + standalone helpers
// ---------------------------------------------------------------------------

std::string tokenizer_dir_for_model(config::ModelType model) {
    switch (model) {
        case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
            return "meta-llama/Llama-3.1-8B-Instruct";
        case config::ModelType::DEEPSEEK_V3:
        default:
            return "deepseek-ai/DeepSeek-V3";
    }
}

std::unique_ptr<Tokenizer> create_tokenizer(config::ModelType model, const std::string& path) {
    switch (model) {
        case config::ModelType::LLAMA_3_1_8B_INSTRUCT:
            return std::make_unique<LlamaTokenizer>(path);
        case config::ModelType::DEEPSEEK_V3:
        default:
            return std::make_unique<DeepseekTokenizer>(path);
    }
}

const Tokenizer& active_tokenizer() {
    static auto tok = create_tokenizer(config::model_type(), config::tokenizer_path());
    return *tok;
}

}  // namespace tt::utils
