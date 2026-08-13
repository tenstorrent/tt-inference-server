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
  std::string device_id;
  py::object runner;         // BGELargeENRunner instance
  py::object request_class;  // domain.text_embedding_request.TextEmbeddingRequest

  explicit Impl(const std::string& devId) : device_id(devId) {}

  ~Impl() { release(); }

  void release() {
    if (!Py_IsInitialized()) return;
    py::gil_scoped_acquire gil;
    runner = py::object();
    request_class = py::object();
    // The interpreter itself is never finalized: other components (and a
    // possible restart of this runner) may still need it.
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

        py::module_ runnerModule =
            py::module_::import("tt_model_runners.embedding_runner");
        TT_LOG_INFO(
            "[EmbeddingRunner] Imported tt_model_runners.embedding_runner");

        request_class = py::module_::import("domain.text_embedding_request")
                            .attr("TextEmbeddingRequest");

        // Model selection is still hardcoded in this phase; Phase 5 makes it
        // config-driven.
        runner = runnerModule.attr("BGELargeENRunner")(device_id);
        TT_LOG_INFO(
            "[EmbeddingRunner] Created BGELargeENRunner instance for device "
            "{}",
            device_id);

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

EmbeddingRunner::EmbeddingRunner(const std::string& deviceId, int visibleDevice)
    : device_id_(deviceId),
      visible_device_(visibleDevice),
      impl_(std::make_unique<Impl>(deviceId)) {
  TT_LOG_INFO(
      "[EmbeddingRunner] EmbeddingRunner created for device {} "
      "visible_device={}",
      deviceId, visibleDevice);
}

EmbeddingRunner::~EmbeddingRunner() { close(); }

bool EmbeddingRunner::warmup() {
  TT_LOG_INFO(
      "[EmbeddingRunner] Starting warmup for device {} visible_device={}",
      device_id_, visible_device_);

  if (!impl_->initialize()) {
    return false;
  }

  TT_LOG_INFO("[EmbeddingRunner] Warmup complete for device {}", device_id_);
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

// IRunner interface implementation
void EmbeddingRunner::run() {
  if (!warmup()) {
    throw std::runtime_error("Failed to initialize EmbeddingRunner");
  }
  TT_LOG_INFO(
      "[EmbeddingRunner] EmbeddingRunner ready for requests on device {}",
      device_id_);
}

void EmbeddingRunner::stop() { close(); }

}  // namespace tt::runners
