// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <string>
#include <variant>

#include "ipc/shared_memory.hpp"

namespace tt::runners {

using ResultPayload = std::variant<ipc::SharedToken, ipc::SharedEmbedding>;

struct RunnerResult {
    std::string task_id;
    ResultPayload payload;
};

using ResultCallback = std::function<void(const RunnerResult&)>;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace tt::runners
