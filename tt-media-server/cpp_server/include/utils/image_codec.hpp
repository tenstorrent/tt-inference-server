// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tt::utils::image_codec {

/**
 * Output formats supported by encodeFloatChw.
 * Mirrors the values accepted by the OpenAI-compatible
 * `image_return_format` field on the request.
 */
enum class Format { PNG, JPEG };

/** Parse "PNG" / "JPEG" / "JPG" (case-insensitive). Defaults to JPEG to match
 * the Python ImageManager fast path. */
Format parseFormat(const std::string& s);

/**
 * Encode a CHW float32 tensor produced by SDXL VAE-decode (range [-1, 1])
 * into a base64-encoded PNG/JPEG string.
 *
 * The float buffer is contiguous CHW with C ∈ {3, 4}; we apply diffusers'
 * `image_processor.postprocess` denormalization in C++:
 *
 *   uint8 = clamp((x / 2 + 0.5) * 255, 0, 255)
 *
 * For C=4 we treat it as RGBA, otherwise RGB. The base64 payload contains
 * no line breaks so it is wire-compatible with the Python
 * `ImageManager._convert_image_to_base64` output.
 *
 * Throws std::runtime_error on unsupported channel counts or encode failure.
 */
std::string encodeFloatChwToBase64(const float* chw, int channels, int height,
                                   int width, Format format = Format::JPEG,
                                   int jpegQuality = 85);

/** Helper: base64-encode an arbitrary byte buffer (no line breaks, '='
 * padding). Exposed for tests + reuse outside the float-CHW path. */
std::string base64Encode(const uint8_t* data, size_t size);

}  // namespace tt::utils::image_codec
