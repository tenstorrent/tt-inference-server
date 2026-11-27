// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "domain/video/video_deployment.hpp"
#include "domain/video_generate_request.hpp"

namespace tt::domain::video {

// Python open_ai_api/video.py and domain/video_i2v_generate_request.py.
constexpr size_t kMaxImageBytes = 30'000'000;
constexpr size_t base64Length(size_t bytes) { return 4 * ((bytes + 2) / 3); }
constexpr size_t kMaxBase64ImageLen = base64Length(kMaxImageBytes);
/** The request body cap for inline media (URL assets are not in the body). */
constexpr size_t kMaxInlineMediaLen = 64'000'000;
/** `/generations/i2v/upload`: the uploaded file. */
constexpr size_t kMaxUploadBytes = kMaxImageBytes;

/** A request the media rules refuse, with the HTTP status to answer. */
class MediaRequestError : public std::runtime_error {
 public:
  MediaRequestError(int status, const std::string& message)
      : std::runtime_error(message), status_(status) {}
  int status() const { return status_; }

 private:
  int status_;
};

inline bool isMediaUrl(const std::string& value) {
  const std::string head = lowerCase(value.substr(0, 8));
  return head.rfind("http://", 0) == 0 || head.rfind("https://", 0) == 0;
}

/** Standard base64 like the Python path (`ImageManager.base64_to_pil_image`): a
 *  leading `data:...,` prefix is skipped and missing `=` padding is accepted.
 *  Line breaks and spaces are skipped, anything else outside the alphabet is an
 *  error (nullopt), as is padding that does not fit the length. */
inline std::optional<std::vector<uint8_t>> base64Decode(
    const std::string& text) {
  static const std::string alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<uint8_t> out;
  out.reserve(text.size() / 4 * 3);
  uint32_t buffer = 0;
  int bits = 0;
  size_t padding = 0;
  size_t symbols = 0;
  size_t start = 0;
  if (text.rfind("data:", 0) == 0) {
    const size_t comma = text.find(',');
    if (comma == std::string::npos) return std::nullopt;
    start = comma + 1;
  }
  for (size_t i = start; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    if (c == '=') {
      ++padding;
      continue;
    }
    if (padding != 0) return std::nullopt;  // data after the padding
    const size_t pos = alphabet.find(c);
    if (pos == std::string::npos) return std::nullopt;
    buffer = (buffer << 6) | static_cast<uint32_t>(pos);
    bits += 6;
    ++symbols;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
    }
  }
  // One symbol more than a multiple of 4 cannot be the end of any input;
  // padding, when present, has to complete the last group.
  if (padding > 2 || symbols == 0 || symbols % 4 == 1 ||
      (padding != 0 && (symbols + padding) % 4 != 0)) {
    return std::nullopt;
  }
  return out;
}

inline uint32_t be32(const std::vector<uint8_t>& b, size_t at) {
  return (uint32_t(b[at]) << 24) | (uint32_t(b[at + 1]) << 16) |
         (uint32_t(b[at + 2]) << 8) | uint32_t(b[at + 3]);
}
inline uint32_t le16(const std::vector<uint8_t>& b, size_t at) {
  return uint32_t(b[at]) | (uint32_t(b[at + 1]) << 8);
}
inline uint32_t le24(const std::vector<uint8_t>& b, size_t at) {
  return uint32_t(b[at]) | (uint32_t(b[at + 1]) << 8) |
         (uint32_t(b[at + 2]) << 16);
}
inline uint32_t le32(const std::vector<uint8_t>& b, size_t at) {
  return le24(b, at) | (uint32_t(b[at + 3]) << 24);
}

/** (width, height) of a PNG, JPEG, WebP, GIF or BMP, read from its header the
 *  way PIL's `Image.open(...).size` does (no pixel decoding). nullopt when the
 *  bytes are none of these or are truncated. */
inline std::optional<std::pair<int, int>> imageSize(
    const std::vector<uint8_t>& b) {
  auto has = [&](size_t n) { return b.size() >= n; };
  auto tag = [&](size_t at, const char* t) {
    for (size_t i = 0; t[i] != '\0'; ++i) {
      if (b[at + i] != static_cast<uint8_t>(t[i])) return false;
    }
    return true;
  };
  if (has(24) && b[0] == 0x89 && tag(1, "PNG") && tag(12, "IHDR")) {
    return std::make_pair(int(be32(b, 16)), int(be32(b, 20)));
  }
  if (has(4) && b[0] == 0xFF && b[1] == 0xD8) {
    size_t at = 2;
    while (at + 9 < b.size()) {
      if (b[at] != 0xFF) {
        ++at;
        continue;
      }
      const uint8_t marker = b[at + 1];
      if (marker == 0xFF) {
        ++at;
        continue;
      }
      const bool sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 &&
                       marker != 0xC8 && marker != 0xCC;
      if (sof) {
        const int height = int((b[at + 5] << 8) | b[at + 6]);
        const int width = int((b[at + 7] << 8) | b[at + 8]);
        return std::make_pair(width, height);
      }
      if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) ||
          marker == 0x01) {
        at += 2;
        continue;
      }
      at += 2 + size_t((b[at + 2] << 8) | b[at + 3]);
    }
    return std::nullopt;
  }
  if (has(30) && tag(0, "RIFF") && tag(8, "WEBP")) {
    if (tag(12, "VP8X")) {
      return std::make_pair(int(le24(b, 24)) + 1, int(le24(b, 27)) + 1);
    }
    if (tag(12, "VP8L") && b[20] == 0x2F) {
      const int width = 1 + int(b[21] | ((b[22] & 0x3F) << 8));
      const int height =
          1 + int((b[22] >> 6) | (b[23] << 2) | ((b[24] & 0x0F) << 10));
      return std::make_pair(width, height);
    }
    if (tag(12, "VP8 ")) {
      return std::make_pair(int(le16(b, 26) & 0x3FFF),
                            int(le16(b, 28) & 0x3FFF));
    }
    return std::nullopt;
  }
  if (has(10) && (tag(0, "GIF87a") || tag(0, "GIF89a"))) {
    return std::make_pair(int(le16(b, 6)), int(le16(b, 8)));
  }
  if (has(26) && b[0] == 'B' && b[1] == 'M') {
    // A negative height is a top-down bitmap. INT32_MIN has no absolute value
    // in 32 bits (and no real image has it), so it is refused.
    const auto height = static_cast<int32_t>(le32(b, 22));
    if (height == std::numeric_limits<int32_t>::min()) return std::nullopt;
    return std::make_pair(int(le32(b, 18)), std::abs(height));
  }
  return std::nullopt;
}

namespace detail {

inline std::vector<uint8_t> decodeImage(const std::string& b64,
                                        const std::string& name,
                                        std::pair<int, int>* size) {
  const auto bytes = base64Decode(b64);
  const auto dims = bytes ? imageSize(*bytes) : std::nullopt;
  if (!dims || dims->first <= 0 || dims->second <= 0) {
    throw MediaRequestError(
        422, name +
                 " is not a decodable image (supported formats: PNG, JPEG, "
                 "WebP, etc.)");
  }
  *size = *dims;
  return *bytes;
}

inline void checkCap(const std::string& name, const std::string& b64,
                     size_t cap) {
  if (b64.size() > cap) {
    throw MediaRequestError(
        422, name + " base64 length " + std::to_string(b64.size()) +
                 " exceeds the " + std::to_string(cap) + "-char cap");
  }
}

}  // namespace detail

/** Python's submit-time media rules for a parsed request: the inline size cap
 *  and that every image is decodable, so an unreadable asset is a 4xx now and
 *  not a failed job later. Throws MediaRequestError (status and message to
 *  answer). */
inline void checkRequestMedia(const VideoGenerateRequest& request,
                              VideoDeployment /*deployment*/) {
  if (!request.imagePrompts) return;
  const Json::Value& prompts = *request.imagePrompts;

  // Total of the inline payloads (413), before anything is decoded.
  size_t total = 0;
  for (const auto& entry : prompts) total += entry["image"].asString().size();
  if (total > kMaxInlineMediaLen) {
    throw MediaRequestError(413, "Inline media totals " +
                                     std::to_string(total) +
                                     " base64 chars, over the " +
                                     std::to_string(kMaxInlineMediaLen) +
                                     "-char cap; send large assets by URL");
  }

  int index = 0;
  for (const auto& entry : prompts) {
    const std::string name = "image_prompts[" + std::to_string(index) + "]";
    detail::checkCap(name + ".image", entry["image"].asString(),
                     kMaxBase64ImageLen);
    std::pair<int, int> size{0, 0};
    detail::decodeImage(entry["image"].asString(), name, &size);
    ++index;
  }
}

}  // namespace tt::domain::video
