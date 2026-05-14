// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tt::utils::image_codec {

enum class Format { PNG, JPEG };

/** Case-insensitive "PNG"/"JPEG"/"JPG". Defaults to JPEG. */
Format parseFormat(const std::string& s);

/**
 * Encode an SDXL VAE-decode CHW float32 tensor (range [-1, 1]) into a
 * base64 PNG/JPEG. Applies diffusers' `image_processor.postprocess`
 * denormalization:  uint8 = clamp((x / 2 + 0.5) * 255, 0, 255).
 * C must be 3 (RGB) or 4 (RGBA); no line breaks in the base64 output.
 */
std::string encodeFloatChwToBase64(const float* chw, int channels, int height,
                                   int width, Format format = Format::JPEG,
                                   int jpegQuality = 85);

std::string base64Encode(const uint8_t* data, size_t size);

}  // namespace tt::utils::image_codec
