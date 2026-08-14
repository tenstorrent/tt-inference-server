// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_scheduler_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/types.hpp"
#include "utils/logger.hpp"
#include "utils/tts_tokenizer.hpp"

#if __has_include(<tt_llm_engine/scheduler/tts/tts_scheduler.hpp>) && \
    __has_include(<tt_llm_engine/pipeline/mock_pipeline.hpp>) &&        \
    __has_include(<tt_llm_engine/pipeline/encoder_pipeline.hpp>) &&     \
    __has_include(<tt_llm_engine/pipeline/decoder_pipeline.hpp>)
#define TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER 1
#include <tt_llm_engine/pipeline/decoder_pipeline.hpp>
#include <tt_llm_engine/pipeline/encoder_pipeline.hpp>
#include <tt_llm_engine/pipeline/mock_pipeline.hpp>
#include <tt_llm_engine/scheduler/scheduler_types.hpp>
#include <tt_llm_engine/scheduler/tts/tts_scheduler.hpp>
#endif

#if defined(TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER) && \
    defined(TT_MEDIA_SERVER_TTS_FULL_PIPELINES) &&     \
    __has_include(<tt_llm_engine/pipeline/decoder_socket_pipeline.hpp>) && \
    __has_include(<tt_llm_engine/pipeline/encoder_socket_pipeline.hpp>) && \
    __has_include(<tt_llm_engine/pipeline/socket_pipeline.hpp>) &&         \
    __has_include(<tt_llm_engine/pipeline/speechlm_wire_codec.hpp>)
#define TT_MEDIA_SERVER_HAS_TTS_SOCKET_PIPELINES 1
#include <tt_llm_engine/pipeline/decoder_socket_pipeline.hpp>
#include <tt_llm_engine/pipeline/encoder_socket_pipeline.hpp>
#include <tt_llm_engine/pipeline/socket_pipeline.hpp>
#include <tt_llm_engine/pipeline/speechlm_wire_codec.hpp>
#endif

namespace tt::runners::blaze {

namespace {

uint16_t floatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

class MockTtsScheduler final : public tts_scheduler::ITtsScheduler {
 public:
  MockTtsScheduler(uint32_t audioSampleRateHz, uint16_t audioChannels,
                   size_t maxUsers)
      : audioSampleRateHz(audioSampleRateHz),
        audioChannels(audioChannels),
        slotBusy(std::max<size_t>(maxUsers, 1), false) {}

  void start() override {}
  void stop() override {}

  bool pushRequest(const tts_scheduler::SchedulerRequest& request) override {
    tts_scheduler::SchedulerResponse response;
    response.type = request.type;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId = request.slotId;

    switch (request.type) {
      case tts_scheduler::RequestType::ALLOCATE:
        response.slotId = allocateSlot();
        if (response.slotId == tts_scheduler::INVALID_SLOT) {
          response.errorCode = -1;
          response.error = "Mock TTS scheduler has no free slots";
        }
        break;
      case tts_scheduler::RequestType::STOP:
      case tts_scheduler::RequestType::EVICT:
        releaseSlot(request.slotId);
        break;
      case tts_scheduler::RequestType::CONTINUE:
      case tts_scheduler::RequestType::SUBMIT:
        break;
    }

    responses.push_back(std::move(response));
    return true;
  }

  bool submit(const tts_scheduler::TtsSubmit& request) override {
    tts_scheduler::SchedulerResponse response;
    response.type = tts_scheduler::RequestType::SUBMIT;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId = request.slotId;
    if (!isValidSlot(request.slotId) || !slotBusy[request.slotId]) {
      response.errorCode = -1;
      response.error = "Mock TTS scheduler received SUBMIT for an invalid slot";
      responses.push_back(std::move(response));
      return true;
    }

    responses.push_back(std::move(response));
    enqueueAudio(request);
    return true;
  }

  bool tryPopResponse(tts_scheduler::SchedulerResponse& response) override {
    if (responses.empty()) {
      return false;
    }
    response = std::move(responses.front());
    responses.erase(responses.begin());
    return true;
  }

  bool tryPopToken(tts_scheduler::TokenOutput& output) override {
    if (tokens.empty()) {
      return false;
    }
    output = std::move(tokens.front());
    tokens.erase(tokens.begin());
    return true;
  }

  bool tryPopAudio(tts_scheduler::AudioOutput& output) override {
    if (audio.empty()) {
      return false;
    }
    output = std::move(audio.front());
    audio.erase(audio.begin());
    return true;
  }

  bool enqueueVoiceEncode(tts_scheduler::VoiceEncodeRequest request) override {
    tts_scheduler::VoiceEncodeResult result;
    result.requestId = request.requestId;
    result.status = tts_scheduler::VoiceEncodeStatus::Completed;
    result.speechIds = {12, 34, 56};
    voiceResults.push_back(std::move(result));
    return true;
  }

  bool tryPopVoiceEncodeResult(
      tts_scheduler::VoiceEncodeResult& result) override {
    if (voiceResults.empty()) {
      return false;
    }
    result = std::move(voiceResults.front());
    voiceResults.erase(voiceResults.begin());
    return true;
  }

 private:
  bool isValidSlot(uint32_t slotId) const { return slotId < slotBusy.size(); }

  uint32_t allocateSlot() {
    for (size_t slotId = 0; slotId < slotBusy.size(); ++slotId) {
      if (!slotBusy[slotId]) {
        slotBusy[slotId] = true;
        return static_cast<uint32_t>(slotId);
      }
    }
    return tts_scheduler::INVALID_SLOT;
  }

  void releaseSlot(uint32_t slotId) {
    if (!isValidSlot(slotId)) {
      return;
    }
    slotBusy[slotId] = false;
  }

  void enqueueAudio(const tts_scheduler::TtsSubmit& request) {
    constexpr size_t kSamplesPerChunk = 960;  // 20 ms at 48 kHz.
    constexpr size_t kChunkCount = 3;
    // Codec tokens the mock "decodes" into each audio chunk. The real engine
    // emits one TokenOutput per acoustic token; the mock mirrors that shape so
    // codec-token throughput (tt_tts_codec_tokens_total) is exercisable
    // without hardware.
    constexpr size_t kTokensPerChunk = 8;

    for (size_t i = 0; i < kChunkCount * kTokensPerChunk; ++i) {
      tts_scheduler::TokenOutput token;
      token.requestId = request.requestId;
      token.taskId = request.taskId;
      token.slotId = request.slotId;
      token.tokenId = static_cast<uint32_t>(i);
      tokens.push_back(std::move(token));
    }

    tts_scheduler::TokenOutput terminalToken;
    terminalToken.requestId = request.requestId;
    terminalToken.taskId = request.taskId;
    terminalToken.slotId = request.slotId;
    terminalToken.final = true;
    tokens.push_back(std::move(terminalToken));

    for (size_t chunkIndex = 0; chunkIndex < kChunkCount; ++chunkIndex) {
      tts_scheduler::AudioOutput output;
      output.requestId = request.requestId;
      output.taskId = request.taskId;
      output.slotId = request.slotId;
      output.chunkIndex = static_cast<uint32_t>(chunkIndex);
      output.sampleRateHz = audioSampleRateHz;
      output.channels = audioChannels;
      output.last = chunkIndex == kChunkCount - 1;
      output.finishReason = tt::domain::tts::TtsFinishReason::Completed;
      output.samplesBf16.reserve(kSamplesPerChunk);
      for (size_t sampleIndex = 0; sampleIndex < kSamplesPerChunk;
           ++sampleIndex) {
        const bool high = ((sampleIndex / 80) + chunkIndex) % 2 == 0;
        output.samplesBf16.push_back(floatToBf16(high ? 0.10f : -0.10f));
      }
      audio.push_back(std::move(output));
    }
  }

  uint32_t audioSampleRateHz = 0;
  uint16_t audioChannels = 0;
  std::vector<bool> slotBusy;
  std::vector<tts_scheduler::SchedulerResponse> responses;
  std::vector<tts_scheduler::TokenOutput> tokens;
  std::vector<tts_scheduler::AudioOutput> audio;
  std::vector<tts_scheduler::VoiceEncodeResult> voiceResults;
};

#if defined(TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER)
namespace engine_tts = tt_llm_engine::scheduler::tts;
namespace engine_sched = tt_llm_engine::scheduler;
namespace engine_pipeline = tt_llm_engine::pipeline;

tts_scheduler::RequestType fromEngineRequestType(
    engine_sched::RequestType type) {
  switch (type) {
    case engine_sched::RequestType::ALLOCATE:
      return tts_scheduler::RequestType::ALLOCATE;
    case engine_sched::RequestType::SUBMIT:
      return tts_scheduler::RequestType::SUBMIT;
    case engine_sched::RequestType::CONTINUE:
      return tts_scheduler::RequestType::CONTINUE;
    case engine_sched::RequestType::STOP:
      return tts_scheduler::RequestType::STOP;
    case engine_sched::RequestType::EVICT:
      return tts_scheduler::RequestType::EVICT;
  }
  return tts_scheduler::RequestType::ALLOCATE;
}

engine_sched::RequestType toEngineRequestType(tts_scheduler::RequestType type) {
  switch (type) {
    case tts_scheduler::RequestType::ALLOCATE:
      return engine_sched::RequestType::ALLOCATE;
    case tts_scheduler::RequestType::SUBMIT:
      return engine_sched::RequestType::SUBMIT;
    case tts_scheduler::RequestType::CONTINUE:
      return engine_sched::RequestType::CONTINUE;
    case tts_scheduler::RequestType::STOP:
      return engine_sched::RequestType::STOP;
    case tts_scheduler::RequestType::EVICT:
      return engine_sched::RequestType::EVICT;
  }
  return engine_sched::RequestType::ALLOCATE;
}

engine_sched::GenerationParams toEngineGeneration(
    const tts_scheduler::GenerationParams& generation) {
  engine_sched::GenerationParams out;
  out.max_new_tokens = generation.maxNewTokens;
  out.ignore_eos = generation.ignoreEos;
  out.stop_tokens = generation.stopTokens;
  return out;
}

tts_scheduler::VoiceEncodeStatus fromEngineVoiceStatus(
    engine_tts::VoiceEncodeStatus status) {
  switch (status) {
    case engine_tts::VoiceEncodeStatus::COMPLETED:
      return tts_scheduler::VoiceEncodeStatus::Completed;
    case engine_tts::VoiceEncodeStatus::CANCELLED:
      return tts_scheduler::VoiceEncodeStatus::Cancelled;
    case engine_tts::VoiceEncodeStatus::ERROR:
      return tts_scheduler::VoiceEncodeStatus::Error;
  }
  return tts_scheduler::VoiceEncodeStatus::Error;
}

template <typename AudioOut>
bool isLastAudioOutput(const AudioOut& audio) {
  if constexpr (requires { audio.last; }) {
    return audio.last;
  }
  return false;
}

engine_tts::TtsSchedulerParams makeEngineTtsParams(
    const tt::config::TtsConfig& config) {
  constexpr uint32_t CODEBOOK_SIZE = 65536;
  const auto& tokenizer =
      tt::utils::tts_tokenizer::tokenizerForPath(config.tokenizerPath);

  engine_tts::TtsSchedulerParams params;
  params.max_users = static_cast<uint32_t>(config.maxUsers);
  params.chunk_tokens = config.chunkTokens;
  params.first_chunk_tokens = config.chunkTokens;
  params.max_batch_size = static_cast<uint32_t>(config.maxBatchSize);
  params.speech_end_token = tt::utils::tts_tokenizer::tokenIdFor(
      tokenizer, tt::utils::tts_tokenizer::SPEECH_END_TOKEN);
  params.speech_token_base = tt::utils::tts_tokenizer::tokenIdFor(
      tokenizer, tt::utils::tts_tokenizer::SPEECH_TOKEN_BASE);
  params.speech_vocab_size = CODEBOOK_SIZE;
  return params;
}

class RealTtsScheduler final : public tts_scheduler::ITtsScheduler {
 public:
  RealTtsScheduler(std::unique_ptr<engine_tts::TtsScheduler> scheduler,
                   uint32_t audioSampleRateHz, uint16_t audioChannels)
      : impl(std::move(scheduler)),
        audioSampleRateHz(audioSampleRateHz),
        audioChannels(audioChannels) {}

  void start() override { impl->start(); }
  void stop() override { impl->stop(); }

  bool pushRequest(const tts_scheduler::SchedulerRequest& request) override {
    rememberRequest(request.requestId, request.taskId);
    engine_sched::ISRequest engineRequest;
    engineRequest.type = toEngineRequestType(request.type);
    engineRequest.request_id = static_cast<uint32_t>(request.requestId);
    engineRequest.slot_id = request.slotId;
    return impl->push_request(engineRequest);
  }

  bool submit(const tts_scheduler::TtsSubmit& request) override {
    rememberRequest(request.requestId, request.taskId);
    rememberSlot(request.slotId, request.requestId, request.taskId);
    engine_tts::TtsSubmit submitRequest;
    submitRequest.request_id = static_cast<uint32_t>(request.requestId);
    submitRequest.slot_id = request.slotId;
    submitRequest.prompt_tokens = request.promptTokens;
    submitRequest.gen = toEngineGeneration(request.generation);
    return impl->submit(submitRequest);
  }

  bool tryPopResponse(tts_scheduler::SchedulerResponse& response) override {
    engine_sched::SchedulerResponse engineResponse;
    if (!impl->try_pop_response(engineResponse)) {
      return false;
    }

    response.type = fromEngineRequestType(engineResponse.request_type);
    response.requestId = engineResponse.request_id;
    response.taskId = taskForRequest(engineResponse.request_id);
    response.slotId = engineResponse.slot_id;
    response.errorCode = engineResponse.error_code;
    if (engineResponse.error_code != engine_sched::request_error::kOk) {
      response.error = "TTS scheduler request failed with error_code=" +
                       std::to_string(engineResponse.error_code);
    }

    if (response.type == tts_scheduler::RequestType::ALLOCATE &&
        response.errorCode == engine_sched::request_error::kOk &&
        response.slotId != tts_scheduler::INVALID_SLOT) {
      rememberSlot(response.slotId, response.requestId, response.taskId);
    } else if (response.type == tts_scheduler::RequestType::EVICT) {
      slotContexts.erase(response.slotId);
    }

    return true;
  }

  bool tryPopToken(tts_scheduler::TokenOutput& output) override {
    engine_sched::OutputMessage engineOutput;
    if (!impl->try_pop_output(engineOutput)) {
      return false;
    }

    output.slotId = engineOutput.slot_id;
    output.requestId = requestForSlot(engineOutput.slot_id);
    output.taskId = taskForSlot(engineOutput.slot_id);
    output.tokenId = engineOutput.token_id;
    output.final = engineOutput.is_complete;
    return true;
  }

  bool tryPopAudio(tts_scheduler::AudioOutput& output) override {
    engine_tts::AudioOut audio;
    if (!impl->try_pop_audio(audio)) {
      return false;
    }

    output.slotId = audio.uid;
    output.requestId = requestForSlot(audio.uid);
    output.taskId = taskForSlot(audio.uid);
    output.chunkIndex = audio.chunk_index;
    output.samplesBf16 = std::move(audio.samples_bf16);
    output.sampleRateHz = audioSampleRateHz;
    output.channels = audioChannels;
    output.last = isLastAudioOutput(audio);
    return true;
  }

  bool enqueueVoiceEncode(tts_scheduler::VoiceEncodeRequest request) override {
    engine_tts::VoiceEncodeRequest engineRequest;
    engineRequest.requestId = static_cast<uint64_t>(request.requestId);
    engineRequest.wavPcm = std::move(request.wavPcm);
    return impl->enqueueVoiceEncode(std::move(engineRequest));
  }

  bool tryPopVoiceEncodeResult(
      tts_scheduler::VoiceEncodeResult& result) override {
    engine_tts::VoiceEncodeResult engineResult;
    if (!impl->tryPopVoiceEncodeResult(engineResult)) {
      return false;
    }
    result.requestId = static_cast<uint32_t>(engineResult.requestId);
    result.speechIds = std::move(engineResult.speechIds);
    result.status = fromEngineVoiceStatus(engineResult.status);
    return true;
  }

 private:
  struct SlotContext {
    uint32_t requestId = 0;
    uint32_t taskId = 0;
  };

  void rememberRequest(uint32_t requestId, uint32_t taskId) {
    if (requestId != 0 && taskId != 0) {
      requestToTask[requestId] = taskId;
    }
  }

  void rememberSlot(uint32_t slotId, uint32_t requestId, uint32_t taskId) {
    if (slotId != tts_scheduler::INVALID_SLOT) {
      slotContexts[slotId] = SlotContext{requestId, taskId};
    }
  }

  uint32_t taskForRequest(uint32_t requestId) const {
    auto it = requestToTask.find(requestId);
    return it == requestToTask.end() ? requestId : it->second;
  }

  uint32_t requestForSlot(uint32_t slotId) const {
    auto it = slotContexts.find(slotId);
    return it == slotContexts.end() ? 0 : it->second.requestId;
  }

  uint32_t taskForSlot(uint32_t slotId) const {
    auto it = slotContexts.find(slotId);
    return it == slotContexts.end() ? 0 : it->second.taskId;
  }

  std::unique_ptr<engine_tts::TtsScheduler> impl;
  uint32_t audioSampleRateHz = 0;
  uint16_t audioChannels = 0;
  std::unordered_map<uint32_t, uint32_t> requestToTask;
  std::unordered_map<uint32_t, SlotContext> slotContexts;
};

#if defined(TT_MEDIA_SERVER_HAS_TTS_SOCKET_PIPELINES)
std::unique_ptr<tts_scheduler::ITtsScheduler> makeRealTtsScheduler(
    const tt::config::TtsConfig& config) {
  TT_LOG_INFO("makeTtsScheduler: constructing real TtsScheduler");
  auto speechlm = std::make_unique<engine_pipeline::SocketPipeline>(
      config.speechlmSocketDescriptorPrefix,
      config.speechlmSocketDescriptorPrefix, config.connectTimeoutMs,
      std::make_unique<engine_pipeline::SpeechlmWireCodec>());
  auto encoder = std::make_unique<engine_pipeline::EncoderSocketPipeline>(
      config.encoderSocketDescriptorPrefix,
      config.encoderSocketDescriptorPrefix, config.connectTimeoutMs);
  auto decoder = std::make_unique<engine_pipeline::DecoderSocketPipeline>(
      config.decoderSocketDescriptorPrefix,
      config.decoderSocketDescriptorPrefix, config.connectTimeoutMs);

  auto scheduler = std::make_unique<engine_tts::TtsScheduler>(
      std::move(speechlm), std::move(encoder), std::move(decoder),
      makeEngineTtsParams(config));
  return std::make_unique<RealTtsScheduler>(
      std::move(scheduler), config.audioSampleRateHz, config.audioChannels);
}
#endif

#endif

}  // namespace

std::unique_ptr<tts_scheduler::ITtsScheduler> makeTtsScheduler(
    const tt::config::TtsConfig& config) {
  (void)config;
#if defined(TT_MEDIA_SERVER_HAS_TTS_SOCKET_PIPELINES)
  return makeRealTtsScheduler(config);
#elif defined(TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER)
  TT_LOG_WARN(
      "makeTtsScheduler: TtsScheduler headers are available, but "
      "socket-capable TtLlmEngine::Full is not linked");
  throw std::runtime_error(
      "TTS scheduler is not linked with socket-capable TtLlmEngine::Full");
#else
  TT_LOG_WARN(
      "makeTtsScheduler: tt-llm-engine TtsScheduler headers are not available; "
      "cannot create real TT_TTS scheduler");
  throw std::runtime_error(
      "tt-llm-engine TtsScheduler headers are not available; cannot create "
      "real TT_TTS scheduler");
#endif
}

std::unique_ptr<tts_scheduler::ITtsScheduler> makeMockTtsScheduler(
    const tt::config::TtsConfig& config) {
  TT_LOG_INFO("makeMockTtsScheduler: constructing mock TTS scheduler");
  return std::make_unique<MockTtsScheduler>(
      config.audioSampleRateHz, config.audioChannels, config.maxUsers);
}

}  // namespace tt::runners::blaze
