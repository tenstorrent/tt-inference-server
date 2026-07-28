// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_tts_scheduler_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/types.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

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

#if defined(TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER) &&          \
    defined(TT_MEDIA_SERVER_TTS_FULL_PIPELINES) &&              \
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

engine_tts::TtsSchedulerParams makeEngineTtsParams(
    const tt::config::TtsConfig& config) {
  constexpr uint32_t CODEBOOK_SIZE = 65536;
  auto tokenIdFor = [](const std::vector<std::string>& vocab,
                       const std::string& token) -> uint32_t {
    auto it = std::find(vocab.begin(), vocab.end(), token);
    if (it == vocab.end()) {
      throw std::runtime_error("TTS tokenizer is missing required token: " +
                               token);
    }
    return static_cast<uint32_t>(std::distance(vocab.begin(), it));
  };

  const auto vocab = tt::utils::tokenizers::activeTokenizer().getEncodedVocab();

  engine_tts::TtsSchedulerParams params;
  params.max_users = static_cast<uint32_t>(config.maxUsers);
  params.chunk_tokens = config.chunkTokens;
  params.first_chunk_tokens = config.firstChunkTokens;
  params.max_batch_size = static_cast<uint32_t>(config.maxBatchSize);
  params.speech_end_token = tokenIdFor(vocab, "<|speech_end|>");
  params.speech_token_base = tokenIdFor(vocab, "<|s_0|>");
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
    return true;
  }

  bool enqueueVoiceEncode(
      tts_scheduler::VoiceEncodeRequest request) override {
    engine_tts::VoiceEncodeRequest engineRequest;
    engineRequest.requestId = request.requestId;
    engineRequest.wavPcm = std::move(request.wavPcm);
    return impl->enqueueVoiceEncode(std::move(engineRequest));
  }

  bool tryPopVoiceEncodeResult(
      tts_scheduler::VoiceEncodeResult& result) override {
    engine_tts::VoiceEncodeResult engineResult;
    if (!impl->tryPopVoiceEncodeResult(engineResult)) {
      return false;
    }
    result.requestId = engineResult.requestId;
    result.speechIds = std::move(engineResult.speechIds);
    result.status = fromEngineVoiceStatus(engineResult.status);
    return true;
  }

  bool isComplete(uint32_t slotId) const override {
    return impl->get_user_state(slotId) == engine_tts::UserState::COMPLETE;
  }

  uint32_t getInFlightCount(uint32_t slotId) const override {
    return impl->get_in_flight_count(slotId);
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

class UnavailableTtsScheduler final : public tts_scheduler::ITtsScheduler {
 public:
  void start() override {}
  void stop() override {}

  bool pushRequest(const tts_scheduler::SchedulerRequest& request) override {
    tts_scheduler::SchedulerResponse response;
    response.type = request.type;
    response.requestId = request.requestId;
    response.taskId = request.taskId;
    response.slotId = request.slotId;
    if (request.type == tts_scheduler::RequestType::ALLOCATE) {
      response.slotId = tts_scheduler::INVALID_SLOT;
      response.errorCode = -1;
      response.error = "TTS scheduler is not linked into this build";
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
    response.errorCode = -1;
    response.error = "TTS scheduler is not linked into this build";
    responses.push_back(std::move(response));
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

  bool tryPopToken(tts_scheduler::TokenOutput& /*output*/) override {
    return false;
  }

  bool tryPopAudio(tts_scheduler::AudioOutput& /*output*/) override {
    return false;
  }

  bool enqueueVoiceEncode(
      tts_scheduler::VoiceEncodeRequest /*request*/) override {
    return false;
  }

  bool tryPopVoiceEncodeResult(
      tts_scheduler::VoiceEncodeResult& /*result*/) override {
    return false;
  }

  bool isComplete(uint32_t /*slotId*/) const override { return false; }

  uint32_t getInFlightCount(uint32_t /*slotId*/) const override { return 0; }

 private:
  std::vector<tts_scheduler::SchedulerResponse> responses;
};

}  // namespace

std::unique_ptr<tts_scheduler::ITtsScheduler> makeTtsScheduler(
    const tt::config::TtsConfig& config) {
#if defined(TT_MEDIA_SERVER_HAS_TTS_SOCKET_PIPELINES)
  return makeRealTtsScheduler(config);
#elif defined(TT_MEDIA_SERVER_HAS_REAL_TTS_SCHEDULER)
  TT_LOG_WARN(
      "makeTtsScheduler: TtsScheduler headers are available, but "
      "socket-capable TtLlmEngine::Full is not linked; using "
      "UnavailableTtsScheduler");
#else
  TT_LOG_WARN(
      "makeTtsScheduler: tt-llm-engine TtsScheduler headers are not available; "
      "using UnavailableTtsScheduler");
#endif
  return std::make_unique<UnavailableTtsScheduler>();
}

}  // namespace tt::runners::blaze
