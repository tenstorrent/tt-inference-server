// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llama_model_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <string>

#include "runners/llm_runner/sequence.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;

namespace llm_engine {

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
    py::list sysPath = sys_mod.attr("path");

    const char* pythonPath = std::getenv("TT_PYTHON_PATH");
    if (pythonPath && *pythonPath) {
      sysPath.attr("insert")(0, pythonPath);
    }
    const char* metalHome = std::getenv("TT_METAL_HOME");
    if (metalHome && *metalHome) {
      sysPath.attr("insert")(0, metalHome);
    }

    py::module_ llamaMod = py::module_::import("tt_model_runners.llama_runner");
    gStepSeqClass = llama_mod.attr("StepSequence");
    py::object runnerClass = llama_mod.attr("Llama31_8BRunner");

    const char* envDev = std::getenv("TT_VISIBLE_DEVICES");
    std::string deviceId = (envDev && *envDev) ? envDev : "0";
    gRunner = runner_class(deviceId);

    py::module_ asyncio = py::module_::import("asyncio");
    bool warmupOk = asyncio.attr("run")(g_runner.attr("warmup")()).cast<bool>();
    if (!warmupOk) {
      TT_LOG_ERROR("[LlamaModelRunner] Warmup failed");
    } else {
      TT_LOG_INFO("[LlamaModelRunner] Llama runner ready (in-process)");
      initialized_ = true;
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
    TokenResult dr(seq->task_id, 0, {}, true);
    decode_callback_(dr);
  }
}

LlamaModelRunner::LlamaModelRunner(const Config& config,
                                   DecodeCallback callback)
    : config_(config), decode_callback_(std::move(callback)) {
  initialize();
}

LlamaModelRunner::~LlamaModelRunner() { exit(); }

void LlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool isPrefill) {
  if (stop_.load() || !initialized_) return;

  bool hadError = false;
  {
    py::gil_scoped_acquire acquire;
    try {
      py::list pySeqs;
      for (Sequence* seq : seqs) {
        py::list token_ids;
        if (is_prefill) {
          for (int64_t t : seq->token_ids_) token_ids.append(t);
        } else {
          token_ids.append(seq->token_ids_.back());
        }

        py::list block_table;
        for (int bid : seq->block_table_) {
          block_table.append(bid);
        }

        int current_pos =
            is_prefill ? 0 : static_cast<int>(seq->token_ids_.size() - 1);
        int prompt_len = static_cast<int>(seq->numPromptTokens_);

        const SamplingParams* sp = seq->sampling_params.get();
        double temperature = sp ? static_cast<double>(sp->temperature) : 1.0;
        bool ignore_eos = sp ? sp->ignore_eos : false;

        py::object seed = py::none();
        if (sp && sp->seed.has_value()) {
          seed = py::int_(*sp->seed);
        }

        py::object top_p = py::none();
        if (sp && sp->top_p.has_value()) {
          top_p = py::float_(*sp->top_p);
        }

        py::object top_k = py::none();
        if (sp && sp->top_k.has_value()) {
          top_k = py::int_(*sp->top_k);
        }

        py::object min_p = py::none();
        if (sp && sp->min_p.has_value()) {
          min_p = py::float_(*sp->min_p);
        }

        py::object repetition_penalty = py::none();
        if (sp && sp->repetition_penalty.has_value()) {
          repetition_penalty = py::float_(*sp->repetition_penalty);
        }

        double presence_penalty =
            sp ? static_cast<double>(sp->presence_penalty) : 0.0;
        double frequency_penalty =
            sp ? static_cast<double>(sp->frequency_penalty) : 0.0;

        py_seqs.append(g_step_seq_class(
            seq->task_id.id, token_ids, temperature, ignore_eos, block_table,
            current_pos, prompt_len, seed, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, frequency_penalty));
      }

      py::object results = g_runner.attr("run")(isPrefill, py_seqs);

      for (size_t i = 0; i < seqs.size(); ++i) {
        py::object item = results[py::int_(i)];
        TaskID drTaskId(item.attr("task_id").cast<std::string>());
        uint64_t drTokenId =
            static_cast<uint64_t>(item.attr("token_id").cast<int64_t>());
        std::string error = item.attr("error").cast<std::string>();
        bool drIsError = !error.empty();
        if (drIsError) {
          TT_LOG_ERROR("[LlamaModelRunner] sequence {} error: {}", drTaskId.id,
                       error);
        }
        TokenResult dr(drTaskId, drTokenId, {}, drIsError);
        decode_callback_(dr);
      }
    } catch (const py::error_already_set& e) {
      TT_LOG_ERROR("[LlamaModelRunner] Python error in run_step: {}", e.what());
      hadError = true;
    }
  }
  if (hadError) {
    fail_sequences(seqs);
    stop_.store(true);
  }
}

void LlamaModelRunner::exit() {
  if (stop_.exchange(true)) return;
  if (!initialized_) return;
  {
    py::gil_scoped_acquire acquire;
    g_runner = py::object();
    g_step_seq_class = py::object();
  }
  initialized_ = false;
  TT_LOG_INFO("[LlamaModelRunner] Runner exited");
}

std::unique_ptr<IModelRunner> makeLlamaModelRunner(const Config& config,
                                                   DecodeCallback callback) {
  auto runner = std::make_unique<LlamaModelRunner>(config, std::move(callback));
  if (!runner->is_ready()) {
    return nullptr;
  }
  return runner;
}

}  // namespace llm_engine
