// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>
#include <sstream>

#include "domain/chat_message.hpp"

namespace tt::chat_templates::deepseek {

/** DeepSeek V3 ChatML special tokens (match tokenizer added_tokens). */
inline const char* im_start() { return "<|im_start|>"; }
inline const char* im_end() { return "<|im_end|>"; }
inline const char* bos() { return "\n"; }

/**
 * Format messages as DeepSeek V3 chat template (ChatML-style).
 * Layout: BOS + system(s) + </think>content</think> (user) + \n</think>content</think> (assistant) + ... + \n</think> (generation).
 * Tool messages are not supported; only system/user/assistant with content.
 */
inline std::string messages_to_prompt(const std::vector<tt::domain::ChatMessage>& messages) {
    std::ostringstream out;
    std::string system_prompt;
    bool first_system = true;
    for (const auto& m : messages) {
        if (m.role != "system") continue;
        if (!first_system) system_prompt += "\n\n";
        system_prompt += m.content;
        first_system = false;
    }
    out << bos() << system_prompt;
    bool need_leading_newline = !system_prompt.empty();
    for (const auto& m : messages) {
        std::string role = m.role.empty() ? "user" : m.role;
        if (role == "system") continue;
        if (role == "user") {
            if (need_leading_newline) out << "\n";
            need_leading_newline = false;
            out << im_start() << m.content << im_end();
        } else if (role == "assistant") {
            out << "\n" << im_start() << m.content << im_end();
        }
    }
    out << "\n" << im_start();
    return out.str();
}

}  // namespace tt::chat_templates::deepseek
