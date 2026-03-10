// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llama_model_runner.hpp"
#include "runners/llm_runner/sequence.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace py = pybind11;

namespace llm_engine {

namespace {
py::object g_runner;
py::object g_step_seq_class;
}  // namespace

bool LlamaModelRunner::initialize() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }

  bool success = false;
  try {
    py::module_ sys_mod = py::module_::import("sys");
    py::list sys_path = sys_mod.attr("path");

    const char* python_path = std::getenv("TT_PYTHON_PATH");
    if (python_path && *python_path) {
      sys_path.attr("insert")(0, python_path);
    }
    const char* metal_home = std::getenv("TT_METAL_HOME");
    if (metal_home && *metal_home) {
      sys_path.attr("insert")(0, metal_home);
    }

    py::module_ llama_mod = py::module_::import("tt_model_runners.llama_runner");
    g_step_seq_class = llama_mod.attr("StepSequence");
    py::object runner_class = llama_mod.attr("Llama31_8BRunner");

    const char* env_dev = std::getenv("TT_VISIBLE_DEVICES");
    std::string device_id = (env_dev && *env_dev) ? env_dev : "0";
    g_runner = runner_class(device_id);

    py::module_ asyncio = py::module_::import("asyncio");
    bool warmup_ok = asyncio.attr("run")(g_runner.attr("warmup")()).cast<bool>();
    if (!warmup_ok) {
      std::cerr << "[LlamaModelRunner] Warmup failed\n";
    } else {
      std::cout << "[LlamaModelRunner] Llama runner ready (in-process)\n";
      initialized_ = true;
      success = true;
    }
  } catch (const py::error_already_set& e) {
    std::cerr << "[LlamaModelRunner] Python init error: " << e.what() << "\n";
  }

  PyEval_SaveThread();
  return success;
}

void LlamaModelRunner::fail_sequences(const std::vector<Sequence*>& seqs) {
  for (Sequence* seq : seqs) {
    TokenResult dr;
    dr.task_id = seq->task_id;
    dr.token_id = 0;
    dr.is_error = true;
    decode_callback_(dr);
  }
}

LlamaModelRunner::LlamaModelRunner(const Config& config, DecodeCallback callback)
    : config_(config), decode_callback_(std::move(callback)) {
  initialize();
}

LlamaModelRunner::~LlamaModelRunner() {
  exit();
}

void LlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  if (stop_.load() || !initialized_) return;

  bool had_error = false;
  {
    py::gil_scoped_acquire acquire;
    try {
      py::list py_seqs;
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

        int current_pos = is_prefill ? 0 : static_cast<int>(seq->token_ids_.size() - 1);
        int prompt_len = static_cast<int>(seq->num_prompt_tokens_);

        const SamplingParams* sp = seq->sampling_params.get();
        int max_tokens = sp ? sp->max_tokens : 64;
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

        double presence_penalty = sp ? static_cast<double>(sp->presence_penalty) : 0.0;
        double frequency_penalty = sp ? static_cast<double>(sp->frequency_penalty) : 0.0;

        py_seqs.append(
            g_step_seq_class(seq->task_id.id, token_ids, max_tokens, temperature, ignore_eos,
                            block_table, current_pos, prompt_len, seed, top_p, top_k, min_p,
                            repetition_penalty, presence_penalty, frequency_penalty));
      }

      py::object results = g_runner.attr("run")(is_prefill, py_seqs);

      for (size_t i = 0; i < seqs.size(); ++i) {
        py::object item = results[py::int_(i)];
        TokenResult dr;
        dr.task_id.id = item.attr("task_id").cast<std::string>();
        dr.token_id = item.attr("token_id").cast<int64_t>();
        std::string error = item.attr("error").cast<std::string>();
        if (!error.empty()) {
          dr.is_error = true;
          std::cerr << "[LlamaModelRunner] sequence " << dr.task_id.id
                    << " error: " << error << "\n";
        }
        decode_callback_(dr);
      }
    } catch (const py::error_already_set& e) {
      std::cerr << "[LlamaModelRunner] Python error in run_step: " << e.what() << "\n";
      had_error = true;
    }
  }
  if (had_error) {
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
  std::cout << "[LlamaModelRunner] Runner exited\n";
}

std::unique_ptr<IModelRunner> make_llama_model_runner(const Config& config,
                                                      DecodeCallback callback) {
  auto runner = std::make_unique<LlamaModelRunner>(config, std::move(callback));
  if (!runner->is_ready()) {
    return nullptr;
  }
  return runner;
}

}  // namespace llm_engine
