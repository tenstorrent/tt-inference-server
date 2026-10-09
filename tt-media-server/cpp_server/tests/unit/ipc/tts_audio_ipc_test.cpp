// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// The TTS IPC wire after the Qwen3-TTS additions: the task's optional voice
// fields and the audio message's PCM16 samples survive serialization, messages
// without them (TTS-2) still round-trip unchanged, the runner's chunks fit the
// audio queue, and the bytes the WAV writer streams for a PCM16 chunk are its
// samples, unconverted.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "config/defaults.hpp"
#include "domain/tts/tts_types.hpp"
#include "ipc/tts_ipc.hpp"
#include "runtime/runners/qwen3_tts/qwen3_tts_runner_support.hpp"
#include "utils/audio_codec.hpp"

namespace {

using tt::ipc::tts::TtsAudioChunkMessage;
using tt::ipc::tts::TtsAudioChunkQueue;
using tt::ipc::tts::TtsIpcTask;
namespace qwen3 = tt::runners::qwen3_tts;
namespace codec = tt::utils::audio_codec;

template <typename Message>
Message roundTrip(const Message& message) {
  std::stringstream stream;
  message.serialize(stream);
  return Message::deserialize(stream);
}

template <typename Message>
size_t wireSize(const Message& message) {
  std::stringstream stream;
  message.serialize(stream);
  return stream.str().size();
}

TEST(TtsIpcTaskTest, RoundTripsTheQwen3VoiceFields) {
  TtsIpcTask task;
  task.task_id = 11;
  task.text = "Hello.";
  task.description = "Whisper.";
  task.voiceWavPcm = {1, -2, 3};
  task.speaker = "ryan";
  task.language = "English";
  task.referenceText = "what the clip says";

  const auto back = roundTrip(task);
  EXPECT_EQ(back.task_id, 11u);
  EXPECT_EQ(back.text, "Hello.");
  EXPECT_EQ(back.description, "Whisper.");
  EXPECT_EQ(back.voiceWavPcm, (std::vector<int16_t>{1, -2, 3}));
  EXPECT_EQ(back.speaker, "ryan");
  EXPECT_EQ(back.language, "English");
  EXPECT_EQ(back.referenceText, "what the clip says");
}

TEST(TtsIpcTaskTest, RoundTripsATtsTwoTaskUnchanged) {
  TtsIpcTask task;
  task.task_id = 12;
  task.text = "Hello.";
  task.promptTokens = {5, 6, 7};
  task.generation.ignoreEos = true;
  task.generation.stopTokenIds = {9};

  const auto back = roundTrip(task);
  EXPECT_EQ(back.promptTokens, (std::vector<uint32_t>{5, 6, 7}));
  EXPECT_TRUE(back.generation.ignoreEos);
  EXPECT_EQ(back.generation.stopTokenIds, (std::vector<uint32_t>{9}));
  EXPECT_FALSE(back.description.has_value());
  EXPECT_FALSE(back.speaker.has_value());
  EXPECT_FALSE(back.language.has_value());
  EXPECT_FALSE(back.referenceText.has_value());
}

TEST(TtsIpcTaskTest, DomainConversionCarriesTheVoiceFields) {
  tt::domain::tts::TtsTask task;
  task.task_id = 3;
  task.speaker = "vivian";
  task.language = "Chinese";
  task.referenceText = "transcript";
  const auto back = TtsIpcTask::fromDomainTask(task).toDomainTask();
  EXPECT_EQ(back.speaker, "vivian");
  EXPECT_EQ(back.language, "Chinese");
  EXPECT_EQ(back.referenceText, "transcript");
}

TEST(TtsAudioChunkMessageTest, RoundTripsPcm16Samples) {
  TtsAudioChunkMessage message;
  message.task_id = 21;
  message.chunkIndex = 2;
  message.sampleRateHz = 24000;
  message.channels = 1;
  message.samplesPcm16 = {std::numeric_limits<int16_t>::min(), -1, 0, 1,
                          std::numeric_limits<int16_t>::max()};

  const auto back = roundTrip(message);
  EXPECT_EQ(back.task_id, 21u);
  EXPECT_EQ(back.chunkIndex, 2u);
  EXPECT_EQ(back.sampleRateHz, 24000u);
  EXPECT_EQ(back.channels, 1);
  EXPECT_EQ(back.samplesPcm16, message.samplesPcm16);
  EXPECT_TRUE(back.samplesBf16.empty());
  EXPECT_FALSE(back.isFinal());

  const auto chunk = back.toDomainChunk();
  EXPECT_EQ(chunk.samplesPcm16, message.samplesPcm16);
  EXPECT_EQ(TtsAudioChunkMessage::fromDomainChunk(21, chunk).samplesPcm16,
            message.samplesPcm16);
}

TEST(TtsAudioChunkMessageTest, RoundTripsABf16OnlyMessageUnchanged) {
  TtsAudioChunkMessage message;
  message.task_id = 22;
  message.chunkIndex = 0;
  message.sampleRateHz = 48000;
  message.channels = 1;
  message.samplesBf16 = {0x3F80, 0xBF80, 0x0000};

  const auto back = roundTrip(message);
  EXPECT_EQ(back.samplesBf16, message.samplesBf16);
  EXPECT_TRUE(back.samplesPcm16.empty());
  EXPECT_EQ(back.sampleRateHz, 48000u);
}

TEST(TtsAudioChunkMessageTest, RoundTripsTheTerminalMessage) {
  auto message = TtsAudioChunkMessage::finish(
      23, tt::domain::tts::TtsFinishReason::Error, "boom");
  message.voiceEncodeUs = 1234;
  const auto back = roundTrip(message);
  EXPECT_TRUE(back.isFinal());
  EXPECT_TRUE(back.isError());
  EXPECT_EQ(back.error, "boom");
  EXPECT_EQ(back.voiceEncodeUs, 1234u);
  EXPECT_TRUE(back.samplesPcm16.empty());
}

TEST(Qwen3TtsChunkingTest, SplitsAnUtteranceIntoQueueSizedChunks) {
  std::vector<int16_t> samples(250000);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = static_cast<int16_t>(i % 30000);
  }
  const auto chunks = qwen3::splitIntoChunks(
      5, samples, tt::config::defaults::TTS_QWEN3_CHUNK_SAMPLES, 24000);
  ASSERT_EQ(chunks.size(), 3u);

  std::vector<int16_t> joined;
  for (size_t i = 0; i < chunks.size(); ++i) {
    EXPECT_EQ(chunks[i].task_id, 5u);
    EXPECT_EQ(chunks[i].chunkIndex, i);
    EXPECT_EQ(chunks[i].sampleRateHz, 24000u);
    EXPECT_EQ(chunks[i].channels, 1);
    EXPECT_EQ(chunks[i].flags, 0u);
    EXPECT_TRUE(chunks[i].samplesBf16.empty());
    EXPECT_LE(chunks[i].samplesPcm16.size(),
              tt::config::defaults::TTS_QWEN3_CHUNK_SAMPLES);
    // The audio queue's message limit (TtsAudioChunkQueue::Queue).
    EXPECT_LT(wireSize(chunks[i]), TtsAudioChunkQueue::Queue::MAX_MSG_SIZE);
    joined.insert(joined.end(), chunks[i].samplesPcm16.begin(),
                  chunks[i].samplesPcm16.end());
  }
  EXPECT_EQ(joined, samples);
}

TEST(Qwen3TtsChunkingTest, EmptyUtteranceHasNoChunks) {
  EXPECT_TRUE(qwen3::splitIntoChunks(5, {}, 96000, 24000).empty());
  EXPECT_THROW(qwen3::splitIntoChunks(5, {1}, 0, 24000), std::invalid_argument);
}

TEST(Qwen3TtsCancelledTaskSetTest, RemembersABoundedNumberOfCancels) {
  qwen3::CancelledTaskSet cancelled(2);
  cancelled.add(1);
  cancelled.add(2);
  cancelled.add(2);
  EXPECT_EQ(cancelled.size(), 2u);
  cancelled.add(3);  // 1 falls out
  EXPECT_FALSE(cancelled.contains(1));
  EXPECT_TRUE(cancelled.contains(2));
  EXPECT_TRUE(cancelled.contains(3));
  EXPECT_TRUE(cancelled.take(2));
  EXPECT_FALSE(cancelled.take(2));
  EXPECT_EQ(cancelled.size(), 1u);
  cancelled.add(4);
  cancelled.add(5);  // 3 falls out, not 4
  EXPECT_FALSE(cancelled.contains(3));
  EXPECT_TRUE(cancelled.contains(4));
}

TEST(TtsWavBytesTest, Pcm16ChunkIsWrittenAsItsSamples) {
  tt::domain::tts::TtsAudioChunk chunk;
  chunk.samplesPcm16 = {0x0102, -2, std::numeric_limits<int16_t>::min()};
  // A PCM16 chunk ignores any bf16 payload (there never is one).
  chunk.samplesBf16 = {0x3F80};
  const std::string bytes = codec::audioChunkToPcm16Bytes(chunk);
  const std::string expected("\x02\x01\xFE\xFF\x00\x80", 6);
  EXPECT_EQ(bytes, expected);
}

TEST(TtsWavBytesTest, Bf16ChunkIsStillConverted) {
  tt::domain::tts::TtsAudioChunk chunk;
  chunk.samplesBf16 = {0x3F80, 0xBF80, 0x0000};  // 1.0, -1.0, 0.0
  EXPECT_EQ(codec::audioChunkToPcm16Bytes(chunk),
            codec::bf16SamplesToPcm16Bytes(chunk.samplesBf16));
  const std::string expected("\xFF\x7F\x01\x80\x00\x00", 6);
  EXPECT_EQ(codec::audioChunkToPcm16Bytes(chunk), expected);
}

TEST(TtsWavBytesTest, HeaderAdvertisesTheQwen3Rate) {
  const std::string header = codec::makeStreamingPcm16WavHeader(24000, 1);
  ASSERT_EQ(header.size(), 44u);
  const auto u32 = [&header](size_t offset) {
    return static_cast<uint32_t>(static_cast<unsigned char>(header[offset])) |
           static_cast<uint32_t>(static_cast<unsigned char>(header[offset + 1]))
               << 8 |
           static_cast<uint32_t>(static_cast<unsigned char>(header[offset + 2]))
               << 16 |
           static_cast<uint32_t>(static_cast<unsigned char>(header[offset + 3]))
               << 24;
  };
  EXPECT_EQ(u32(24), 24000u);      // sample rate
  EXPECT_EQ(u32(28), 24000u * 2);  // byte rate, mono PCM16
}

}  // namespace
