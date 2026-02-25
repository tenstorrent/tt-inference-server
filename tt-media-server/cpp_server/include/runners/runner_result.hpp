// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <variant>

namespace tt::runners {

struct TokenPayload {
    uint64_t token_id;
    bool finished;
};

struct EmbeddingPayload {};

using ResultPayload = std::variant<TokenPayload, EmbeddingPayload>;

struct RunnerResult {
    std::string task_id;
    ResultPayload payload;
};

using ResultCallback = std::function<void(const RunnerResult&)>;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace tt::runners
