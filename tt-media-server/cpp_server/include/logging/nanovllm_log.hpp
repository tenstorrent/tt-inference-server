// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <sstream>
#include <iostream>

namespace tt::logging {

struct NanovllmLogStream {
    std::ostringstream ss;
    const char* component;
    explicit NanovllmLogStream(const char* c) : component(c) {}
    ~NanovllmLogStream() {
        std::cout << "[DEBUG] [nanovllm:" << component << "] " << ss.str() << std::endl;
    }
    template<typename T>
    NanovllmLogStream& operator<<(const T& v) { ss << v; return *this; }
};

}  // namespace tt::logging

#define NANOVLLM_LOG(component) ::tt::logging::NanovllmLogStream(component)
