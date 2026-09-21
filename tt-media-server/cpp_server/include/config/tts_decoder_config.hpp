// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt::config {

// One audio-decoder instance: the /dev/shm descriptor pair its process
// exported, plus the chunk size its captured trace was built for.
//
// The trace is a single fixed op DAG, so ONE decoder process serves exactly
// ONE wire size. The v2 chunk ramp (17 -> 32 -> 48) therefore runs several
// instances, each on its own chip, and chunks route to the one whose trace
// matches. chunkTokens == 0 means "the server's steady size", which keeps a
// bare "h2d:d2h" entry behaving as it did before sizes existed.
struct TtsDecoderSocketPair {
  std::string h2dSocketId;
  std::string d2hSocketId;
  uint32_t chunkTokens = 0;
};

// Blackhole's host-memory (PCIe) page alignment. Every H2D page size must be a
// multiple of it, and a decoder page is `tokens * 4` bytes, so a wire size must
// be a multiple of 16 tokens.
inline constexpr uint32_t TTS_PCIE_ALIGNMENT = 64;
inline constexpr uint32_t TTS_TOKENS_PER_PAGE_UNIT = TTS_PCIE_ALIGNMENT / 4;

// Smallest legal wire size that can carry `chunkTokens` codes.
//
// 17 is NOT a legal wire size: 17 * 4 = 68 bytes is not a multiple of 64 and
// set_page_size would TT_FATAL. A 17-token first chunk therefore rides a
// 32-token page, padded, with the audio trimmed back to 17 -- the same
// pad-up/slice-down path a short final chunk already uses.
inline constexpr uint32_t ttsWireTokensFor(uint32_t chunkTokens) {
  const uint32_t units =
      (chunkTokens + TTS_TOKENS_PER_PAGE_UNIT - 1) / TTS_TOKENS_PER_PAGE_UNIT;
  return (units == 0 ? 1 : units) * TTS_TOKENS_PER_PAGE_UNIT;
}

inline void ttsValidateWireTokens(uint32_t tokens) {
  if (tokens == 0 || (tokens * 4) % TTS_PCIE_ALIGNMENT != 0) {
    throw std::runtime_error(
        "[Config] TTS decoder wire size " + std::to_string(tokens) +
        " gives a " + std::to_string(tokens * 4) +
        "-byte H2D page, which is not a multiple of the " +
        std::to_string(TTS_PCIE_ALIGNMENT) +
        "-byte PCIe alignment. Use a multiple of " +
        std::to_string(TTS_TOKENS_PER_PAGE_UNIT) +
        " tokens and pad shorter chunks onto it.");
  }
}

// Parse TTS_DECODER_SOCKET_PAIRS: a comma-separated list of
// "h2d_id:d2h_id" or "h2d_id:d2h_id:tokens". Ids must be unique per direction
// and carry no path separators or whitespace -- they name /dev/shm
// descriptors, not files.
inline std::vector<TtsDecoderSocketPair> parseTtsDecoderSocketPairs(
    const std::string& spec) {
  const auto invalid = [](const std::string& why) {
    throw std::runtime_error(
        "[Config] TTS_DECODER_SOCKET_PAIRS must be a nonempty comma-separated "
        "list of h2d_id:d2h_id[:tokens] entries, with unique IDs per direction "
        "and no paths or whitespace (" + why + ")");
  };
  const auto clean = [&](const std::string& id) {
    if (id.empty()) invalid("empty id");
    if (id.find_first_of(" \t\n/\\") != std::string::npos) invalid("id '" + id + "'");
    return id;
  };

  std::vector<TtsDecoderSocketPair> pairs;
  std::vector<std::string> seenH2d, seenD2h;
  size_t start = 0;
  while (start <= spec.size()) {
    const size_t comma = spec.find(',', start);
    const std::string entry =
        spec.substr(start, comma == std::string::npos ? std::string::npos
                                                      : comma - start);
    if (entry.empty()) invalid("empty entry");

    const size_t first = entry.find(':');
    if (first == std::string::npos) invalid("entry '" + entry + "' has no ':'");
    const size_t second = entry.find(':', first + 1);

    TtsDecoderSocketPair pair;
    pair.h2dSocketId = clean(entry.substr(0, first));
    pair.d2hSocketId = clean(second == std::string::npos
                                 ? entry.substr(first + 1)
                                 : entry.substr(first + 1, second - first - 1));
    if (second != std::string::npos) {
      const std::string tok = entry.substr(second + 1);
      if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) {
        invalid("token count '" + tok + "' is not a number");
      }
      pair.chunkTokens = static_cast<uint32_t>(std::stoul(tok));
      ttsValidateWireTokens(pair.chunkTokens);
    }

    for (const auto& s : seenH2d)
      if (s == pair.h2dSocketId) invalid("duplicate h2d id '" + s + "'");
    for (const auto& s : seenD2h)
      if (s == pair.d2hSocketId) invalid("duplicate d2h id '" + s + "'");
    seenH2d.push_back(pair.h2dSocketId);
    seenD2h.push_back(pair.d2hSocketId);
    pairs.push_back(std::move(pair));

    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (pairs.empty()) invalid("no entries");
  return pairs;
}

}  // namespace tt::config
