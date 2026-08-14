// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "runtime/runners/embedding_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace tt::runners {

namespace {

// Take the first line of a Python error (the "ValueError: ..." part); the
// full traceback is logged separately and is too long for a response field.
std::string firstLine(const std::string& s) {
  const auto pos = s.find('\n');
  return pos == std::string::npos ? s : s.substr(0, pos);
}

// Prepend the given env var's value to sys.path if present and not already
// there. Deliberate pinning: the embedded interpreter must resolve tt-metal
// model code and tt-media-server modules from configured locations, not from
// whatever the launching shell happened to have in PYTHONPATH.
void prependEnvToSysPath(py::list& sysPath, const char* envName) {
  const char* value = std::getenv(envName);
  if (!value || !*value) return;
  for (const auto& entry : sysPath) {
    if (py::str(entry).cast<std::string>() == value) return;
  }
  sysPath.attr("insert")(0, py::str(value));
  TT_LOG_INFO("[EmbeddingRunner] Prepended to sys.path from {}: {}", envName,
              value);
}

void ensureSysPath() {
  py::list sysPath = py::module_::import("sys").attr("path");
  prependEnvToSysPath(sysPath, "TT_METAL_HOME");
  // TT_PYTHON_PATH (the tt-media-server checkout) may also come from config
  // defaults rather than the environment, so use the resolved config value.
  const std::string mediaServerPath = tt::config::pythonPath();
  if (!mediaServerPath.empty()) {
    bool present = false;
    for (const auto& entry : sysPath) {
      if (py::str(entry).cast<std::string>() == mediaServerPath) {
        present = true;
        break;
      }
    }
    if (!present) {
      sysPath.attr("insert")(0, py::str(mediaServerPath));
      TT_LOG_INFO("[EmbeddingRunner] Added to sys.path: {}", mediaServerPath);
    }
  }
}

}  // namespace

struct EmbeddingRunner::Impl {
  config::EmbeddingConfig config;
  py::object runner;         // the configured Python runner instance
  py::object request_class;  // domain.text_embedding_request.TextEmbeddingRequest

  explicit Impl(const config::EmbeddingConfig& cfg) : config(cfg) {}

  ~Impl() { release(); }

  void release() {
    if (!Py_IsInitialized()) return;
    py::gil_scoped_acquire gil;
    runner = py::object();
    request_class = py::object();
    // The interpreter itself is never finalized: other components (and a
    // possible restart of this runner) may still need it.
  }

  // Python's ModelConfigs table is authoritative for max_batch_size, but the
  // service needs the number before Python exists in order to cap batches, so
  // the C++ table mirrors it. Compare the two as soon as Python is importable:
  // a silent mismatch means oversized batches and an assertion inside the
  // model later, which is far harder to read than failing here.
  void checkPythonBatchSize() const {
    const auto pythonBatch = py::module_::import("config.settings")
                                 .attr("settings")
                                 .attr("max_batch_size")
                                 .cast<size_t>();
    if (pythonBatch != config.max_batch_size) {
      throw std::runtime_error(
          "[EmbeddingRunner] max_batch_size mismatch: C++ config says " +
          std::to_string(config.max_batch_size) + ", Python resolved " +
          std::to_string(pythonBatch) +
          ". The C++ table in settings.cpp has drifted from "
          "config/constants.py, or MODEL/DEVICE reached Python with "
          "unexpected values.");
    }
    TT_LOG_INFO("[EmbeddingRunner] max_batch_size={} (agrees with Python)",
                pythonBatch);
  }

  bool initialize() {
    // Boot the interpreter once per process. pybind11 leaves the GIL held
    // after initialization; release it at the end of warmup-time setup so
    // every later entry point can acquire 
    const bool ownsInterpreter = !Py_IsInitialized();
    if (ownsInterpreter) {
      py::initialize_interpreter();
      TT_LOG_INFO("[EmbeddingRunner] Python interpreter initialized");
    }

    bool ok = false;
    {
      py::gil_scoped_acquire gil;
      try {
        ensureSysPath();

        request_class = py::module_::import("domain.text_embedding_request")
                            .attr("TextEmbeddingRequest");

        checkPythonBatchSize();

        // Which class implements the model is Python's business: the fabric
        // maps settings.model_runner (from the MODEL_RUNNER we exported) to a
        // runner class, so onboarding a model needs no class name in C++.
        runner = py::module_::import("tt_model_runners.runner_fabric")
                     .attr("get_device_runner")(config.visible_devices);
        TT_LOG_INFO("[EmbeddingRunner] Created {} for device {}",
                    py::str(runner.attr("__class__").attr("__name__"))
                        .cast<std::string>(),
                    config.visible_devices);

        runner.attr("set_device")();
        TT_LOG_INFO("[EmbeddingRunner] set_device() completed");

        // warmup() is an async coroutine; drive it to completion.
        py::object coro = runner.attr("warmup")();
        py::object result = py::module_::import("asyncio").attr("run")(coro);
        ok = result.cast<bool>();
        TT_LOG_INFO("[EmbeddingRunner] Warmup completed: {}",
                    ok ? "success" : "failed");
      } catch (const py::error_already_set& e) {
        TT_LOG_ERROR("[EmbeddingRunner] Warmup failed:\n{}", e.what());
        ok = false;
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[EmbeddingRunner] Warmup failed: {}", e.what());
        ok = false;
      }
    }

    if (ownsInterpreter) {
      PyEval_SaveThread();
    }
    return ok;
  }

  std::vector<domain::EmbeddingResponse> runInference(
      const std::vector<domain::EmbeddingRequest>& requests) {
    std::vector<domain::EmbeddingResponse> responses;
    responses.reserve(requests.size());

    py::gil_scoped_acquire gil;
    try {
      py::list pyRequests;
      for (const auto& req : requests) {
        pyRequests.append(
            request_class("model"_a = req.model, "input"_a = req.input));
      }

      py::sequence results = runner.attr("run")(pyRequests);

      // results[i] is the answer to requests[i]: the Python responses carry no
      // identity of their own, so the pairing is positional by contract.
      for (size_t i = 0; i < results.size(); ++i) {
        py::object item = results[i];
        domain::EmbeddingResponse resp(requests[i].task_id);
        resp.model = requests[i].model;
        resp.embedding = item.attr("embedding").cast<std::vector<float>>();
        resp.total_tokens = item.attr("total_tokens").cast<int>();
        responses.push_back(std::move(resp));
      }

      TT_LOG_DEBUG("[EmbeddingRunner] Processed {} embedding requests",
                   responses.size());
    } catch (const py::error_already_set& e) {
      // Surface the real Python error to every caller in the batch instead
      // of a generic "no response". The full traceback goes to the log.
      TT_LOG_ERROR("[EmbeddingRunner] Inference failed:\n{}", e.what());
      const std::string message = firstLine(e.what());
      responses.clear();
      for (const auto& req : requests) {
        domain::EmbeddingResponse resp(req.task_id);
        resp.error = message;
        responses.push_back(std::move(resp));
      }
    }

    return responses;
  }
};

// Public interface implementation

EmbeddingRunner::EmbeddingRunner(const config::EmbeddingConfig& config)
    : config_(config), impl_(std::make_unique<Impl>(config)) {
  TT_LOG_INFO("[EmbeddingRunner] Created for model {} on device {} (worker {})",
              config_.hf_model_id, config_.visible_devices, config_.worker_id);
}

EmbeddingRunner::~EmbeddingRunner() { close(); }

bool EmbeddingRunner::warmup() {
  TT_LOG_INFO("[EmbeddingRunner] Starting warmup for {} on device {}",
              config_.hf_model_id, config_.visible_devices);

  if (!impl_->initialize()) {
    return false;
  }

  TT_LOG_INFO("[EmbeddingRunner] Warmup complete for device {}",
              config_.visible_devices);
  return true;
}

void EmbeddingRunner::close() {
  if (impl_) {
    impl_->release();
  }
}

std::vector<domain::EmbeddingResponse> EmbeddingRunner::run(
    const std::vector<domain::EmbeddingRequest>& requests) {
  if (!impl_ || !impl_->runner) {
    TT_LOG_ERROR("[EmbeddingRunner] Runner not initialized");
    return {};
  }

  return impl_->runInference(requests);
}

}  // namespace tt::runners
