// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/llama_model_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <string>

#include "domain/sequence.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::runners::llm_engine {
using Sequence = tt::domain::Sequence;
using TokenResult = tt::domain::TokenResult;
using Config = tt::config::LLMConfig;

namespace {
py::object gRunner;
py::object gStepSeqClass;
}  // namespace

bool LlamaModelRunner::initialize() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }

  bool success = false;
  try {
    py::module_ sysMod = py::module_::import("sys");
    py::list sysPath = sysMod.attr("path");

    const char* pythonPath = std::getenv("TT_PYTHON_PATH");
    if (pythonPath && *pythonPath) {
      sysPath.attr("insert")(0, pythonPath);
    }
    const char* metalHome = std::getenv("TT_METAL_HOME");
    if (metalHome && *metalHome) {
      sysPath.attr("insert")(0, metalHome);
    }

    py::module_ llamaMod = py::module_::import("tt_model_runners.llama_runner");
    gStepSeqClass = llamaMod.attr("StepSequence");
    py::object runnerClass = llamaMod.attr("Llama31_8BRunner");

    const char* envDev = std::getenv("TT_VISIBLE_DEVICES");
    std::string deviceId = (envDev && *envDev) ? envDev : "0";
    gRunner = runnerClass(deviceId);

    py::module_ asyncio = py::module_::import("asyncio");
    bool warmupOk = asyncio.attr("run")(gRunner.attr("warmup")()).cast<bool>();
    if (!warmupOk) {
      TT_LOG_ERROR("[LlamaModelRunner] Warmup failed");
    } else {
      TT_LOG_INFO("[LlamaModelRunner] Llama runner ready (in-process)");
      initialized = true;
      success = true;
    }
  } catch (const py::error_already_set& e) {
    TT_LOG_ERROR("[LlamaModelRunner] Python init error: {}", e.what());
  }

  PyEval_SaveThread();
  return success;
}

void LlamaModelRunner::failSequences(const std::vector<Sequence*>& seqs) {
  for (Sequence* seq : seqs) {
    TokenResult dr(seq->taskId, 0, {}, true);
    decodeCallback(dr);
  }
}

LlamaModelRunner::LlamaModelRunner(const Config& config,
                                   DecodeCallback callback)
    : config(config), decodeCallback(std::move(callback)) {
  initialize();
}

LlamaModelRunner::~LlamaModelRunner() { exit(); }

void LlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool isPrefill) {
  if (stop.load() || !initialized) return;

  bool hadError = false;
  {
    py::gil_scoped_acquire acquire;
    try {
      py::list pySeqs;
      for (Sequence* seq : seqs) {
        py::list tokenIds;
        if (isPrefill) {
          for (int64_t t : seq->getTokenIds()) tokenIds.append(t);
        } else {
          tokenIds.append(seq->getTokenIds().back());
        }

        py::list blockTable;
        for (int bid : seq->getBlockTable()) {
          blockTable.append(bid);
        }

        int currentPos =
            isPrefill ? 0 : static_cast<int>(seq->getTokenIds().size() - 1);
        int promptLen = static_cast<int>(seq->getNumPromptTokens());

        const tt::domain::SamplingParams* sp = &seq->getSamplingParams();
        double temperature = sp ? static_cast<double>(sp->temperature) : 1.0;
        bool ignoreEos = sp ? sp->ignore_eos : false;

        py::object seed = py::none();
        if (sp && sp->seed.has_value()) {
          seed = py::int_(*sp->seed);
        }

        py::object topP = py::none();
        if (sp && sp->top_p.has_value()) {
          topP = py::float_(*sp->top_p);
        }

        py::object topK = py::none();
        if (sp && sp->top_k.has_value()) {
          topK = py::int_(*sp->top_k);
        }

        py::object minP = py::none();
        if (sp && sp->min_p.has_value()) {
          minP = py::float_(*sp->min_p);
        }

        py::object repetitionPenalty = py::none();
        if (sp && sp->repetition_penalty.has_value()) {
          repetitionPenalty = py::float_(*sp->repetition_penalty);
        }

        double presencePenalty =
            sp ? static_cast<double>(sp->presence_penalty) : 0.0;
        double frequencyPenalty =
            sp ? static_cast<double>(sp->frequency_penalty) : 0.0;

        py::object allowedTokenIds = py::none();
        if (sp && sp->token_bitmask.has_value()) {
          py::list pyAllowed;
          const auto& bitmask = *sp->token_bitmask;
          int vocabSize = sp->bitmask_vocab_size;
          for (size_t w = 0; w < bitmask.size(); ++w) {
            auto word = static_cast<uint32_t>(bitmask[w]);
            while (word != 0) {
              int bit = __builtin_ctz(word);
              int tokenId = static_cast<int>(w) * 32 + bit;
              if (tokenId < vocabSize) pyAllowed.append(tokenId);
              word &= word - 1;
            }
          }
          allowedTokenIds = std::move(pyAllowed);
        } else if (sp && sp->allowed_token_ids.has_value()) {
          py::list pyAllowed;
          for (int tid : *sp->allowed_token_ids) {
            pyAllowed.append(tid);
          }
          allowedTokenIds = std::move(pyAllowed);
        }

        pySeqs.append(gStepSeqClass(
            seq->taskId, tokenIds, temperature, ignoreEos, blockTable,
            currentPos, promptLen, seed, topP, topK, minP, repetitionPenalty,
            presencePenalty, frequencyPenalty, allowedTokenIds));
      }

      // First decode step after prefill must set reset_batch=true so on-device
      // sampling state is initialized correctly.
      bool resetBatch = !isPrefill && lastStepWasPrefill;
      lastStepWasPrefill = isPrefill;

      py::object results = gRunner.attr("run")(isPrefill, pySeqs, resetBatch);

      for (size_t i = 0; i < seqs.size(); ++i) {
        py::object item = results[py::int_(i)];
        uint32_t drTaskId = item.attr("task_id").cast<uint32_t>();
        uint64_t drTokenId =
            static_cast<uint64_t>(item.attr("token_id").cast<int64_t>());
        std::string error = item.attr("error").cast<std::string>();
        bool drIsError = !error.empty();
        if (drIsError) {
          TT_LOG_ERROR("[LlamaModelRunner] sequence {} error: {}", drTaskId,
                       error);
        }
        TokenResult dr(drTaskId, drTokenId, {}, drIsError);
        decodeCallback(dr);
      }
    } catch (const py::error_already_set& e) {
      TT_LOG_ERROR("[LlamaModelRunner] Python error in run_step: {}", e.what());
      hadError = true;
    }
  }
  if (hadError) {
    failSequences(seqs);
    stop.store(true);
  }
}

void LlamaModelRunner::exit() {
  if (stop.exchange(true)) return;
  if (!initialized) return;
  {
    py::gil_scoped_acquire acquire;
    gRunner = py::object();
    gStepSeqClass = py::object();
  }
  initialized = false;
  TT_LOG_INFO("[LlamaModelRunner] Runner exited");
}

std::unique_ptr<IModelRunner> makeLlamaModelRunner(const Config& config,
                                                   DecodeCallback callback) {
  auto runner = std::make_unique<LlamaModelRunner>(config, std::move(callback));
  if (!runner->isReady()) {
    return nullptr;
  }
  return runner;
}

}  // namespace tt::runners::llm_engine
