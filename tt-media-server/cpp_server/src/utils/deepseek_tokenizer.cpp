// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/deepseek_tokenizer.hpp"

#include <sstream>

namespace tt::utils {

static const char* DS_USER_TAG = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
static const char* DS_ASSISTANT_TAG = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";

std::string DeepseekTokenizer::apply_chat_template(
    const std::vector<domain::ChatMessage>& messages,
    bool add_generation_prompt) const {

    std::ostringstream out;

    if (cfg_.add_bos_token) out << cfg_.bos_token;

    for (const auto& m : messages) {
        if (m.role == "system") out << m.content;
    }

    for (const auto& m : messages) {
        if (m.role == "system") continue;
        if (m.role == "user") {
            out << DS_USER_TAG << m.content;
        } else if (m.role == "assistant") {
            out << DS_ASSISTANT_TAG << m.content;
            if (cfg_.add_eos_token) out << cfg_.eos_token;
        }
    }

    if (add_generation_prompt) {
        out << DS_ASSISTANT_TAG;
    }
    return out.str();
}

}  // namespace tt::utils
