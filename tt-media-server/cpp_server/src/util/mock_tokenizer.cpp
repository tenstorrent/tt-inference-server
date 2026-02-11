// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "util/mock_tokenizer.hpp"

namespace tt::util {

std::vector<int> MockTokenizer::tokenize(const std::string& text) {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int>(c));
    }
    return tokens;
}

std::string MockTokenizer::detokenize(const std::vector<int>& token_ids) {
    std::string result;
    result.reserve(token_ids.size());
    for (int id : token_ids) {
        result += static_cast<char>(id & 0xFF);
    }
    return result;
}

}  // namespace tt::util
