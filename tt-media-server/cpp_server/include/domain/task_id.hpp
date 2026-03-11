// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace tt::domain {

struct TaskID {
    static constexpr size_t kSerializedSize = 36;

    explicit TaskID(std::string task_id) : id(std::move(task_id)) {}

    std::string id;

    bool operator==(const TaskID& other) const { return id == other.id; }

    std::vector<char> ipc_serialize() const {
        std::vector<char> buf(kSerializedSize, '\0');
        std::copy_n(id.begin(), std::min(id.size(), kSerializedSize), buf.begin());
        return buf;
    }

    static TaskID ipc_deserialize(const char* data, size_t len) {
        size_t actual_len = strnlen(data, len);
        return TaskID(std::string(data, actual_len));
    }

    static std::string generate() {
        return boost::uuids::to_string(boost::uuids::random_generator()());
    }
};

inline std::ostream& operator<<(std::ostream& os, const TaskID& tid) {
    return os << tid.id;
}

}  // namespace tt::domain

namespace std {
template <>
struct hash<tt::domain::TaskID> {
    size_t operator()(const tt::domain::TaskID& s) const {
        return hash<string>{}(s.id);
    }
};
}  // namespace std
