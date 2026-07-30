// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Build a prompt whose tokenized length is roughly `targetTokens`.
// Uses short single-token words so word count ≈ token count (chat-template
// overhead still applies on the wire).

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace tt::test {

inline std::string generatePromptWithApproxTokens(std::size_t targetTokens) {
  static constexpr std::string_view kWords[] = {"hello", "world", "test",
                                                "data", "check"};
  static constexpr std::size_t kNumWords = sizeof(kWords) / sizeof(kWords[0]);
  // Longest entry is "hello "/"check " (6 chars). *7 is reserve headroom only;
  // generation length is driven by the word loop, not this multiplier.
  std::string out;
  out.reserve(targetTokens * 7);
  for (std::size_t i = 0; i < targetTokens; ++i) {
    out += kWords[i % kNumWords];
    out += ' ';
  }
  return out;
}

}  // namespace tt::test
