// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/prefix_cache/response_id_index.hpp"

#include <optional>
#include <string>

#include "domain/prefix_cache/helpers.hpp"
namespace tt::domain::prefix_cache {

std::optional<std::string> ResponseIdIndex::lookup(
    const std::string& id) const {
  if (id.empty()) {
    return std::nullopt;
  }
  return responseIdIndex.get(id);
}

void ResponseIdIndex::registerId(const std::string& id,
                                 const std::string& sessionId) {
  if (id.empty()) {
    return;
  }
  responseIdIndex.insert(id, sessionId);
}

std::optional<std::string> ResponseIdIndex::updateId(const std::string& prevId,
                                                     const std::string& newId) {
  if (prevId.empty() || newId.empty() || prevId == newId) {
    return std::nullopt;
  }

  auto sessionId = responseIdIndex.take(prevId);
  if (!sessionId.has_value()) {
    return std::nullopt;
  }

  responseIdIndex.insert(newId, *sessionId);
  return sessionId;
}

void ResponseIdIndex::removeIf(const std::string& sessionId,
                               const std::string& id) {
  if (id.empty()) {
    return;
  }

  const auto indexedSessionId = responseIdIndex.get(id);
  if (indexedSessionId.has_value() && *indexedSessionId == sessionId) {
    responseIdIndex.erase(id);
  }
}
}  // namespace tt::domain::prefix_cache
