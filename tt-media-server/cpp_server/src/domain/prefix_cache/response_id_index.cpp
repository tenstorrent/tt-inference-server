// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/prefix_cache/response_id_index.hpp"

#include <optional>
#include <string>

#include "domain/prefix_cache/helpers.hpp"
namespace tt::domain::prefix_cache {

std::optional<std::string> ResponseIdIndex::lookup(const std::string& id) {
  if (id.empty()) return std::nullopt;
  std::string sessionId;
  const bool present = responseIdIndex.modify(
      id, [&sessionId](std::string& sid) { sessionId = sid; });
  if (!present) return std::nullopt;
  return sessionId;
}

void ResponseIdIndex::init(const std::string& id,
                           const std::string& sessionId) {
  if (id.empty()) return;
  const bool existed = responseIdIndex.modify(
      id, [&sessionId](std::string& sid) { sid = sessionId; });
  if (!existed) {
    responseIdIndex.insert(id, sessionId);
  }
}

std::optional<std::string> ResponseIdIndex::rekey(const std::string& prevId,
                                                  const std::string& newId) {
  if (prevId.empty() || newId.empty() || prevId == newId) {
    return std::nullopt;
  }
  std::string sessionId;
  const bool found = responseIdIndex.modify(
      prevId, [&sessionId](std::string& sid) { sessionId = sid; });
  if (!found) return std::nullopt;
  responseIdIndex.erase(prevId);
  const bool existed = responseIdIndex.modify(
      newId, [&sessionId](std::string& sid) { sid = sessionId; });
  if (!existed) {
    responseIdIndex.insert(newId, sessionId);
  }
  return sessionId;
}

void ResponseIdIndex::removeIf(const std::string& sessionId,
                               const std::string& id) {
  if (id.empty()) return;
  bool matches = false;
  const bool found =
      responseIdIndex.modify(id, [&sessionId, &matches](std::string& sid) {
        matches = (sid == sessionId);
      });
  if (found && matches) {
    responseIdIndex.erase(id);
  }
}
}  // namespace tt::domain::prefix_cache
