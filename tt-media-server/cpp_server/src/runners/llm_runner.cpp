#include "runners/llm_runner.hpp"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "ipc/token_push.hpp"
#include "profiling/tracy.hpp"
#include "services/guided_decoder_manager.hpp"
#include "services/memory_services/paged_memory_manager.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::runners {
using namespace tt::runners::llm_engine;
using Config = tt::config::LLMConfig;

LLMRunner::LLMRunner(const Config& config, ipc::IResultQueue* resultQueue,
                     ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)
    : config_(config), result_queue_(resultQueue), cancel_queue_(cancelQueue) {
  scheduler_ =
      makeScheduler(config_, taskQueue, tt::config::maxInFlightCount());

  if (tt::config::llmMode() != config::LLMMode::PREFILL_ONLY) {
    memoryManager = std::make_unique<services::PagedMemoryManager>(
        scheduler_->blockManager());
    memoryThread = std::thread([this] { memoryLoop(); });
  }

  try {
    const auto& tok = tt::utils::tokenizers::activeTokenizer();
    auto encodedVocab = tok.getEncodedVocab();
    int vocabSize = static_cast<int>(encodedVocab.size());
    guidedDecoder = std::make_unique<services::GuidedDecoderManager>(
        encodedVocab, vocabSize);
    TT_LOG_INFO("[LLMRunner] Guided decoder initialized (vocab_size={})",
                vocabSize);
  } catch (const std::exception& e) {
    TT_LOG_WARN(
        "[LLMRunner] Failed to init guided decoder, structured outputs"
        "disabled: {}",
        e.what());
  }

  auto decodeCb = [this](const TokenResult& result) {
    ZoneScopedN("LLMRunner::process_token_result");
    Sequence* seq = scheduler_->findSequence(result.taskId);

    if (!seq || seq->isAborted()) return;

    if (result.isError) {
      if (guidedDecoder) guidedDecoder->removeRequest(result.taskId);
      scheduler_->removeSequence(result.taskId);
      ipc::pushErrorToken(*result_queue_, result.taskId);
      return;
    }

    bool grammarFinished = false;
    if (guidedDecoder && guidedDecoder->hasGuidedDecoding(result.taskId)) {
      auto grammarResult = guidedDecoder->acceptToken(
          result.taskId, static_cast<int32_t>(result.tokenId));
      if (!grammarResult.accepted) {
        TT_LOG_WARN(
            "[LLMRunner] Grammar rejected token {} for task {} - "
            "finishing sequence",
            result.tokenId, result.taskId);
        guidedDecoder->removeRequest(result.taskId);
        seq->setStatus(SequenceStatus::FINISHED);
        ipc::pushToken(*result_queue_, result.taskId, result.tokenId, true);
        scheduler_->removeSequence(result.taskId);
        return;
      }
      grammarFinished = grammarResult.completed;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> tokenIds = {static_cast<int64_t>(result.tokenId)};
    scheduler_->postprocess(seqs, tokenIds);

    bool finished = seq->isFinished();
    if (!finished && grammarFinished) {
      finished = true;
      seq->setStatus(SequenceStatus::FINISHED);
    }

    ipc::pushToken(*result_queue_, result.taskId, result.tokenId, finished);

    if (finished) {
      if (guidedDecoder) guidedDecoder->removeRequest(result.taskId);
      scheduler_->removeSequence(result.taskId);
    }
  };

  model_runner_ = makeModelRunner(config_, std::move(decodeCb));
}

LLMRunner::~LLMRunner() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
  exit();
}

void LLMRunner::exit() {
  if (model_runner_) {
    model_runner_->exit();
  }
}

void LLMRunner::run() {
  while (!stopped_.load(std::memory_order_relaxed)) {
    step();
  }
}

void LLMRunner::stop() { stopped_.store(true, std::memory_order_relaxed); }

void LLMRunner::memoryLoop() {
  while (!stopped_.load(std::memory_order_relaxed)) {
    auto task = memoryManager->getRequest();
    if (task) {
      memoryManager->handleRequest(*task);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void LLMRunner::applyGuidedDecodingMasks(const std::vector<Sequence*>& seqs,
                                         bool isPrefill) {
  if (!guidedDecoder) return;

  for (Sequence* seq : seqs) {
    if (isPrefill && seq->getSamplingParams().hasGuidedDecoding()) {
      guidedDecoder->initRequest(seq->taskId, seq->getSamplingParams());
    }
  }

  std::vector<uint32_t> taskIds;
  std::vector<Sequence*> guidedSeqs;
  for (Sequence* seq : seqs) {
    if (guidedDecoder->hasGuidedDecoding(seq->taskId)) {
      taskIds.push_back(seq->taskId);
      guidedSeqs.push_back(seq);
    }
  }

  if (taskIds.empty()) return;

  auto batch = guidedDecoder->getNextAllowedTokenIdsBatch(taskIds);
  for (size_t i = 0; i < batch.size(); ++i) {
    std::vector<int> allowedInt(batch[i].allowedTokenIds.begin(),
                                batch[i].allowedTokenIds.end());
    guidedSeqs[i]->getMutableSamplingParams().allowed_token_ids =
        std::move(allowedInt);
  }
}

void LLMRunner::step() {
  if (cancel_queue_) {
    std::vector<uint32_t> cancelled;
    cancel_queue_->tryPopAll(cancelled);
    for (const auto& taskId : cancelled) {
      if (guidedDecoder) guidedDecoder->removeRequest(taskId);
      scheduler_->abortRequest(taskId);
    }
  }

  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) return;
  ZoneScopedN("LLMRunner::step");

  applyGuidedDecodingMasks(seqs, is_prefill);
  model_runner_->run(seqs, is_prefill);
}
}  // namespace tt::runners
