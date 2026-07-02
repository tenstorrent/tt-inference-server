// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#pragma once
#include <optional>
#include <string>

#include "utils/concurrent_map.hpp"
namespace tt::domain::prefix_cache {
class ResponseIdIndex {
 public:
  std::optional<std::string> lookup(const std::string& id);
  void init(const std::string& id, const std::string& sessionId);
  std::optional<std::string> rekey(const std::string& prevId,
                                   const std::string& newId);
  void removeIf(const std::string& sessionId, const std::string& id);

 private:
  utils::ConcurrentMap<std::string, std::string> responseIdIndex;
};
}  // namespace tt::domain