// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/image_codec.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include <stb_image_write.h>

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tt::utils::image_codec {

namespace {

constexpr const char BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void stbWriteCallback(void* context, void* data, int size) {
  auto* sink = static_cast<std::vector<uint8_t>*>(context);
  const auto* bytes = static_cast<const uint8_t*>(data);
  sink->insert(sink->end(), bytes, bytes + size);
}

uint8_t denormalize(float v) {
  float scaled = (v * 0.5F + 0.5F) * 255.0F;
  if (std::isnan(scaled)) scaled = 0.0F;
  if (scaled < 0.0F) scaled = 0.0F;
  if (scaled > 255.0F) scaled = 255.0F;
  return static_cast<uint8_t>(scaled + 0.5F);
}

std::vector<uint8_t> chwFloatToHwcUint8(const float* chw, int channels,
                                        int height, int width) {
  const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
  std::vector<uint8_t> out(hw * static_cast<size_t>(channels));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pixelIdx =
          (static_cast<size_t>(y) * static_cast<size_t>(width) +
           static_cast<size_t>(x));
      for (int c = 0; c < channels; ++c) {
        const size_t srcIdx = static_cast<size_t>(c) * hw + pixelIdx;
        out[pixelIdx * static_cast<size_t>(channels) + static_cast<size_t>(c)] =
            denormalize(chw[srcIdx]);
      }
    }
  }
  return out;
}

}  // namespace

Format parseFormat(const std::string& s) {
  std::string lower;
  lower.reserve(s.size());
  for (char c : s) {
    lower.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "png") return Format::PNG;
  return Format::JPEG;
}

std::string base64Encode(const uint8_t* data, size_t size) {
  std::string out;
  out.reserve(((size + 2) / 3) * 4);
  size_t i = 0;
  while (i + 3 <= size) {
    uint32_t triple = (uint32_t{data[i]} << 16) | (uint32_t{data[i + 1]} << 8) |
                      uint32_t{data[i + 2]};
    out.push_back(BASE64_ALPHABET[(triple >> 18) & 0x3F]);
    out.push_back(BASE64_ALPHABET[(triple >> 12) & 0x3F]);
    out.push_back(BASE64_ALPHABET[(triple >> 6) & 0x3F]);
    out.push_back(BASE64_ALPHABET[triple & 0x3F]);
    i += 3;
  }
  if (i < size) {
    uint32_t triple = uint32_t{data[i]} << 16;
    if (i + 1 < size) triple |= uint32_t{data[i + 1]} << 8;
    out.push_back(BASE64_ALPHABET[(triple >> 18) & 0x3F]);
    out.push_back(BASE64_ALPHABET[(triple >> 12) & 0x3F]);
    if (i + 1 < size) {
      out.push_back(BASE64_ALPHABET[(triple >> 6) & 0x3F]);
      out.push_back('=');
    } else {
      out.append("==");
    }
  }
  return out;
}

std::string encodeFloatChwToBase64(const float* chw, int channels, int height,
                                   int width, Format format, int jpegQuality) {
  if (channels != 3 && channels != 4) {
    throw std::runtime_error(
        "encodeFloatChwToBase64: unsupported channel count " +
        std::to_string(channels));
  }
  if (height <= 0 || width <= 0 || chw == nullptr) {
    throw std::runtime_error("encodeFloatChwToBase64: invalid input");
  }

  const auto hwc = chwFloatToHwcUint8(chw, channels, height, width);

  std::vector<uint8_t> encoded;
  encoded.reserve(static_cast<size_t>(height) * static_cast<size_t>(width) * 4);

  int rc = 0;
  if (format == Format::PNG) {
    rc = stbi_write_png_to_func(&stbWriteCallback, &encoded, width, height,
                                channels, hwc.data(), width * channels);
  } else {
    int q = jpegQuality;
    if (q < 1) q = 1;
    if (q > 100) q = 100;
    rc = stbi_write_jpg_to_func(&stbWriteCallback, &encoded, width, height,
                                channels, hwc.data(), q);
  }
  if (rc == 0 || encoded.empty()) {
    throw std::runtime_error("encodeFloatChwToBase64: stb_image_write failed");
  }

  return base64Encode(encoded.data(), encoded.size());
}

}  // namespace tt::utils::image_codec
