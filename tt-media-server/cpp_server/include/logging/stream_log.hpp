// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <sstream>
#include <iostream>

namespace tt::logging {

/**
 * Shared stream-style logger used by runners and engine code.
 * Output: [level] [component] msg (component omitted if nullptr).
 */
struct StreamLog {
    std::ostringstream ss;
    const char* level;
    const char* component;
    StreamLog(const char* level_, const char* component_ = nullptr)
        : level(level_), component(component_) {}
    ~StreamLog() {
        std::cout << "[" << level << "]";
        if (component) std::cout << " [" << component << "]";
        std::cout << " " << ss.str() << std::endl;
    }
    template<typename T>
    StreamLog& operator<<(const T& v) { ss << v; return *this; }
    StreamLog& operator<<(std::ostream& (*)(std::ostream&)) { ss << '\n'; return *this; }
};

}  // namespace tt::logging
