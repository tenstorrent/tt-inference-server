// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/audio_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tt::utils::audio_codec {

namespace {

constexpr uint16_t PCM_FORMAT = 1;
constexpr uint16_t PCM_BITS_PER_SAMPLE = 16;
constexpr uint32_t STREAMING_WAV_SIZE = 0xFFFFFFFFu;

void writeU16Le(std::string& out, size_t offset, uint16_t value) {
  out[offset] = static_cast<char>(value & 0xFFu);
  out[offset + 1] = static_cast<char>((value >> 8) & 0xFFu);
}

void writeU32Le(std::string& out, size_t offset, uint32_t value) {
  out[offset] = static_cast<char>(value & 0xFFu);
  out[offset + 1] = static_cast<char>((value >> 8) & 0xFFu);
  out[offset + 2] = static_cast<char>((value >> 16) & 0xFFu);
  out[offset + 3] = static_cast<char>((value >> 24) & 0xFFu);
}

uint16_t readU16Le(std::string_view bytes, size_t offset) {
  if (offset + 2 > bytes.size()) {
    throw std::invalid_argument("truncated WAV file");
  }
  return static_cast<uint16_t>(
      static_cast<unsigned char>(bytes[offset]) |
      (static_cast<unsigned char>(bytes[offset + 1]) << 8));
}

uint32_t readU32Le(std::string_view bytes, size_t offset) {
  if (offset + 4 > bytes.size()) {
    throw std::invalid_argument("truncated WAV file");
  }
  return static_cast<uint32_t>(
      static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
      (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1]))
       << 8) |
      (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2]))
       << 16) |
      (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3]))
       << 24));
}

bool hasTag(std::string_view bytes, size_t offset, std::string_view tag) {
  return offset + tag.size() <= bytes.size() &&
         bytes.substr(offset, tag.size()) == tag;
}

}  // namespace

std::string makeStreamingPcm16WavHeader(uint32_t sampleRateHz,
                                        uint16_t channels) {
  const uint32_t byteRate =
      sampleRateHz * channels * PCM_BITS_PER_SAMPLE / 8;
  const uint16_t blockAlign = channels * PCM_BITS_PER_SAMPLE / 8;

  std::string out(44, '\0');
  std::memcpy(out.data(), "RIFF", 4);
  writeU32Le(out, 4, STREAMING_WAV_SIZE);
  std::memcpy(out.data() + 8, "WAVEfmt ", 8);
  writeU32Le(out, 16, 16);
  writeU16Le(out, 20, PCM_FORMAT);
  writeU16Le(out, 22, channels);
  writeU32Le(out, 24, sampleRateHz);
  writeU32Le(out, 28, byteRate);
  writeU16Le(out, 32, blockAlign);
  writeU16Le(out, 34, PCM_BITS_PER_SAMPLE);
  std::memcpy(out.data() + 36, "data", 4);
  writeU32Le(out, 40, STREAMING_WAV_SIZE);
  return out;
}

std::string bf16SamplesToPcm16Bytes(const std::vector<uint16_t>& samplesBf16) {
  auto bf16ToFloat = [](uint16_t raw) {
    const uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  auto appendPcm16 = [](std::string& out, float sample) {
    const float clamped = std::clamp(sample, -1.0F, 1.0F);
    const auto pcm =
        static_cast<int16_t>(clamped * std::numeric_limits<int16_t>::max());
    const auto raw = static_cast<uint16_t>(pcm);
    out.push_back(static_cast<char>(raw & 0xFFu));
    out.push_back(static_cast<char>((raw >> 8) & 0xFFu));
  };

  std::string out;
  out.reserve(samplesBf16.size() * sizeof(int16_t));
  for (uint16_t sample : samplesBf16) {
    appendPcm16(out, bf16ToFloat(sample));
  }
  return out;
}

std::string audioChunkToPcm16Bytes(
    const tt::domain::tts::TtsAudioChunk& chunk) {
  return bf16SamplesToPcm16Bytes(chunk.samplesBf16);
}

tt::domain::tts::VoiceSample decodePcm16Wav(std::string_view bytes) {
  if (bytes.size() < 44 || !hasTag(bytes, 0, "RIFF") ||
      !hasTag(bytes, 8, "WAVE")) {
    throw std::invalid_argument("voice sample must be a RIFF/WAVE file");
  }

  bool foundFmt = false;
  bool foundData = false;
  uint16_t audioFormat = 0;
  uint16_t channels = 0;
  uint32_t sampleRateHz = 0;
  uint16_t bitsPerSample = 0;
  size_t dataOffset = 0;
  size_t dataSize = 0;

  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const std::string_view chunkId = bytes.substr(offset, 4);
    const uint32_t chunkSize = readU32Le(bytes, offset + 4);
    const size_t chunkDataOffset = offset + 8;
    const size_t nextOffset = chunkDataOffset + chunkSize + (chunkSize % 2);
    if (chunkDataOffset + chunkSize > bytes.size()) {
      throw std::invalid_argument("WAV chunk extends past end of file");
    }

    if (chunkId == "fmt ") {
      if (chunkSize < 16) {
        throw std::invalid_argument("WAV fmt chunk is too small");
      }
      audioFormat = readU16Le(bytes, chunkDataOffset);
      channels = readU16Le(bytes, chunkDataOffset + 2);
      sampleRateHz = readU32Le(bytes, chunkDataOffset + 4);
      bitsPerSample = readU16Le(bytes, chunkDataOffset + 14);
      foundFmt = true;
    } else if (chunkId == "data") {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
      foundData = true;
    }

    offset = nextOffset;
  }

  if (!foundFmt || !foundData) {
    throw std::invalid_argument("WAV file must contain fmt and data chunks");
  }
  if (audioFormat != PCM_FORMAT || bitsPerSample != PCM_BITS_PER_SAMPLE) {
    throw std::invalid_argument("voice sample WAV must be PCM16");
  }
  if (channels == 0 || sampleRateHz == 0) {
    throw std::invalid_argument("voice sample WAV has invalid audio metadata");
  }
  if (dataSize % sizeof(int16_t) != 0) {
    throw std::invalid_argument(
        "voice sample WAV data size must be 16-bit aligned");
  }

  tt::domain::tts::VoiceSample sample;
  sample.sampleRateHz = sampleRateHz;
  sample.channels = channels;
  sample.wavPcm.reserve(dataSize / sizeof(int16_t));
  for (size_t i = dataOffset; i < dataOffset + dataSize; i += 2) {
    const uint16_t raw = readU16Le(bytes, i);
    const auto value =
        raw <= static_cast<uint16_t>(std::numeric_limits<int16_t>::max())
            ? static_cast<int32_t>(raw)
            : static_cast<int32_t>(raw) - 65536;
    sample.wavPcm.push_back(static_cast<int16_t>(value));
  }
  return sample;
}

}  // namespace tt::utils::audio_codec
