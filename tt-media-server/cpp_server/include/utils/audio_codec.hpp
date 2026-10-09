// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "domain/tts/tts_types.hpp"

namespace tt::utils::audio_codec {

/** Decode a RIFF/WAVE PCM16 voice sample into raw signed 16-bit samples. */
tt::domain::tts::VoiceSample decodePcm16Wav(std::string_view bytes);

/** Build a streaming WAV header with unknown final data size. */
std::string makeStreamingPcm16WavHeader(uint32_t sampleRateHz,
                                        uint16_t channels);

/** Convert decoder BF16 sample bit patterns into little-endian PCM16 bytes. */
std::string bf16SamplesToPcm16Bytes(const std::vector<uint16_t>& samplesBf16);

/** Serialize PCM16 samples as little-endian bytes, unconverted. */
std::string pcm16SamplesToBytes(const std::vector<int16_t>& samplesPcm16);

/** Client bytes for one chunk: its PCM16 samples as they are when it carries
 *  any, otherwise its BF16 samples converted. */
std::string audioChunkToPcm16Bytes(const tt::domain::tts::TtsAudioChunk& chunk);

}  // namespace tt::utils::audio_codec
