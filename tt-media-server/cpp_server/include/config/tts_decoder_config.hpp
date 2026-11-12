// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cctype>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tt::config {

struct TtsDecoderSocketPair {
  std::string h2dSocketId;
  std::string d2hSocketId;
};

// Explicit IDs, not filenames: tt-metal adds direction and descriptor suffix.
inline std::vector<TtsDecoderSocketPair> parseTtsDecoderSocketPairs(
    const std::string& value) {
  const auto invalid = []() {
    return std::runtime_error(
        "TTS_DECODER_SOCKET_PAIRS must be a nonempty comma-separated list of "
        "h2d_id:d2h_id pairs, with unique IDs per direction and no paths or "
        "whitespace");
  };
  const auto validId = [](const std::string& id) {
    if (id.empty()) return false;
    for (unsigned char c : id) {
      if (c == '/' || c == ':' || std::isspace(c) || std::iscntrl(c))
        return false;
    }
    return true;
  };
  std::vector<TtsDecoderSocketPair> pairs;
  std::set<std::string> h2dIds;
  std::set<std::string> d2hIds;
  size_t start = 0;
  do {
    const auto end = value.find(',', start);
    const auto entry =
        value.substr(start, end == std::string::npos ? end : end - start);
    const auto colon = entry.find(':');
    if (colon == std::string::npos) throw invalid();
    TtsDecoderSocketPair pair{entry.substr(0, colon), entry.substr(colon + 1)};
    if (!validId(pair.h2dSocketId) || !validId(pair.d2hSocketId) ||
        !h2dIds.insert(pair.h2dSocketId).second ||
        !d2hIds.insert(pair.d2hSocketId).second) {
      throw invalid();
    }
    pairs.push_back(std::move(pair));
    if (end == std::string::npos) break;
    start = end + 1;
  } while (true);
  return pairs;
}

}  // namespace tt::config
