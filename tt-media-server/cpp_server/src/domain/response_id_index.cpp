// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/response_id_index.hpp"

namespace tt::domain {

std::optional<std::string> ResponseIdIndex::lookup(const std::string& id) const {
  if (id.empty()) {
    return std::nullopt;
  }
  return responseIdIndex.get(id);
}

void ResponseIdIndex::init(const std::string& id,
                           const std::string& sessionId) {
  if (id.empty()) {
    return;
  }
  responseIdIndex.insert(id, sessionId);
}

std::optional<std::string> ResponseIdIndex::rekey(const std::string& prevId,
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

}  // namespace tt::domain
