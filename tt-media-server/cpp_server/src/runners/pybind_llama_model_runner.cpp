// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/pybind_llama_model_runner.hpp"
#include "runners/llm_runner/sequence.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace llm_engine {

struct PybindLlamaModelRunner::Impl {
  Config config;
  DecodeCallback decode_callback;
  std::atomic<bool> stop_{false};
  bool initialized_ = false;
  bool owns_interpreter_ = false;

  py::object runner_;
  py::object step_seq_class_;

  bool initialize() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter();
      owns_interpreter_ = true;
    }

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
      step_seq_class_ = llama_mod.attr("StepSequence");
      py::object runner_class = llama_mod.attr("Llama31_8BRunner");

      const char* env_dev = std::getenv("TT_VISIBLE_DEVICES");
      std::string device_id = (env_dev && *env_dev) ? env_dev : "1";
      runner_ = runner_class(device_id);

      py::module_ asyncio = py::module_::import("asyncio");
      bool warmup_ok = asyncio.attr("run")(runner_.attr("warmup")()).cast<bool>();
      if (!warmup_ok) {
        std::cerr << "[PybindLlama] Warmup failed\n";
        return false;
      }

      std::cout << "[PybindLlama] Llama runner ready (in-process)\n";
      initialized_ = true;
      // Release the GIL so the engine step thread (and Python background threads)
      // can acquire it later via PyGILState_Ensure.
      PyEval_SaveThread();
      return true;
    } catch (const py::error_already_set& e) {
      std::cerr << "[PybindLlama] Python init error: " << e.what() << "\n";
      PyEval_SaveThread();
      return false;
    }
  }

  void fail_sequences(const std::vector<Sequence*>& seqs) {
    for (Sequence* seq : seqs) {
      TokenResult dr;
      dr.task_id = seq->task_id;
      dr.token_id = 0;
      dr.is_error = true;
      decode_callback(dr);
    }
  }

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) {
    if (stop_.load() || !initialized_) return;

    bool had_error = false;
    {
      // RAII guard: declared before any py::object locals so the GIL is
      // released AFTER all their destructors (reverse declaration order).
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
          py_seqs.append(
              step_seq_class_(seq->task_id.id, token_ids, max_tokens, temperature, ignore_eos,
                              block_table, current_pos, prompt_len, seed));
        }

        py::object results = runner_.attr("run_step")(is_prefill, py_seqs);

        for (size_t i = 0; i < seqs.size(); ++i) {
          py::object item = results[py::int_(i)];
          TokenResult dr;
          dr.task_id.id = item.attr("task_id").cast<std::string>();
          dr.token_id = item.attr("token_id").cast<int64_t>();
          std::string error = item.attr("error").cast<std::string>();
          if (!error.empty()) {
            dr.is_error = true;
            std::cerr << "[PybindLlama] sequence " << dr.task_id.id
                      << " error: " << error << "\n";
          }
          decode_callback(dr);
        }
      } catch (const py::error_already_set& e) {
        std::cerr << "[PybindLlama] Python error in run_step: " << e.what() << "\n";
        had_error = true;
      }
    }
    if (had_error) {
      fail_sequences(seqs);
      stop_.store(true);
    }
  }

  void do_exit() {
    if (stop_.exchange(true)) return;
    if (!initialized_) return;
    {
      py::gil_scoped_acquire acquire;
      runner_ = py::object();
      step_seq_class_ = py::object();
    }
    initialized_ = false;
    std::cout << "[PybindLlama] Runner exited\n";
  }
};

PybindLlamaModelRunner::PybindLlamaModelRunner(const Config& config, DecodeCallback callback)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  impl_->decode_callback = std::move(callback);
  impl_->initialize();
}

PybindLlamaModelRunner::~PybindLlamaModelRunner() {
  exit();
}

void PybindLlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  impl_->run(seqs, is_prefill);
}

void PybindLlamaModelRunner::exit() {
  impl_->do_exit();
}

bool PybindLlamaModelRunner::is_ready() const {
  return impl_->initialized_;
}

std::unique_ptr<IModelRunner> make_pybind_llama_model_runner(const Config& config,
                                                             DecodeCallback callback) {
  auto runner = std::make_unique<PybindLlamaModelRunner>(config, std::move(callback));
  if (!runner->is_ready()) {
    return nullptr;
  }
  return runner;
}

}  // namespace llm_engine
