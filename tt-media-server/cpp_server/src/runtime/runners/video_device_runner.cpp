// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_device_runner.hpp"

#include <json/json.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include "metrics/metrics.hpp"
#include "runtime/dit/dit_env.hpp"
#include "runtime/dit/dit_python_runner.hpp"
#include "runtime/runners/video_python_env.hpp"
#include "runtime/runners/video_request_convert.hpp"
#include "utils/logger.hpp"
#include "utils/video_output_dir.hpp"

namespace py = pybind11;

namespace tt::runners {

namespace {

// tt_dit keeps converted weights in TT_DIT_CACHE_DIR: the first start fills it,
// later starts load from it instead of re-reading and re-converting every HF
// checkpoint. Without the variable the cache is off. An explicit value wins;
// otherwise default to $XDG_CACHE_HOME (or ~/.cache)/tt_dit_cache.
void setDitCacheDir() {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path dir;
  if (const char* existing = std::getenv("TT_DIT_CACHE_DIR");
      existing && *existing) {
    dir = existing;
  } else {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    if (xdg && *xdg) {
      dir = fs::path(xdg) / "tt_dit_cache";
    } else if (home && *home) {
      dir = fs::path(home) / ".cache" / "tt_dit_cache";
    } else {
      dir = fs::temp_directory_path(ec) / "TT_DIT_CACHE";
    }
  }
  fs::create_directories(dir, ec);
  if (ec) {
    TT_LOG_WARN(
        "[VideoDevice] Cannot create TT_DIT_CACHE_DIR '{}' ({}); weights will "
        "not be cached",
        dir.string(), ec.message());
    return;
  }
  tt::dit::setEnvIfChanged("TT_DIT_CACHE_DIR", dir.string());
  TT_LOG_INFO("[VideoDevice] tt_dit weight cache: {}", dir.string());
}

std::string requireMp4(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec) ||
      fs::file_size(path, ec) == 0) {
    throw std::runtime_error("[VideoDevice] mp4 was not written: " + path);
  }
  return path;
}

// The runner for cfg.pythonRunner with its mesh open. The caller stores it
// before warmup() so that stop() closes the mesh if warmup throws.
tt::dit::RunnerConfig runnerConfigOf(const config::VideoConfig& cfg) {
  tt::dit::RunnerConfig runnerConfig;
  runnerConfig.modelRunner = cfg.pythonRunner;
  runnerConfig.device = cfg.device;
  runnerConfig.modelWeightsPath = cfg.model_weights_path;
  runnerConfig.outputDir = tt::utils::resolveVideoOutputDir().string();
  runnerConfig.workerId = cfg.worker_id;
  return runnerConfig;
}

std::unique_ptr<tt::dit::IDiTRunner> makeRunner(
    const config::VideoConfig& cfg) {
  prepareVideoPythonPath();
  const tt::dit::RunnerConfig runnerConfig = runnerConfigOf(cfg);
  if (const auto model = tt::dit::createModel(runnerConfig);
      model->outputKind() != tt::dit::OutputKind::Video) {
    throw std::runtime_error("[VideoDevice] MODEL_RUNNER=" + cfg.pythonRunner +
                             " is an image model; the video service serves "
                             "video models");
  }
  return tt::dit::createOpenRunner(runnerConfig);
}

}  // namespace

struct VideoDeviceRunner::PythonState {
  // The DiTRunner of cfg.pythonRunner.
  std::unique_ptr<tt::dit::IDiTRunner> runner;
};

VideoDeviceRunner::VideoDeviceRunner(config::VideoConfig config)
    : videoConfig(std::move(config)), python(std::make_unique<PythonState>()) {
  if (videoConfig.pythonRunner.empty()) {
    throw std::runtime_error(
        "[VideoDevice] pythonRunner is empty. Set MODEL_RUNNER, for example "
        "tt-wan2.2.");
  }
  prepareProcessEnvironment(videoConfig);
  TT_LOG_INFO(
      "[VideoDevice] Constructed worker={} runner={} device={} devices='{}' "
      "weights='{}'",
      videoConfig.worker_id, videoConfig.pythonRunner, videoConfig.device,
      videoConfig.visible_devices, videoConfig.model_weights_path);
}

// Everything the runner puts into the process environment. In a worker process
// the constructor does it; when the runner lives on a thread of the server,
// VideoService calls this from the main thread before the runner thread and the
// HTTP threads exist, because setenv() racing a getenv() is undefined. The
// calls the runner thread makes later then change nothing.
void VideoDeviceRunner::prepareProcessEnvironment(
    const config::VideoConfig& cfg) {
  setDitCacheDir();
  if (!cfg.visible_devices.empty()) {
    tt::dit::setEnvIfChanged("TT_VISIBLE_DEVICES", cfg.visible_devices);
  }
  try {
    for (const auto& [name, value] :
         tt::dit::createModel(runnerConfigOf(cfg))->envDefaults()) {
      ::setenv(name.c_str(), value.c_str(), 0);  // a deployment can pin a flag
    }
  } catch (const std::exception&) {
    // An unknown runner or device is reported by warmup(), as before.
  }
}

VideoDeviceRunner::~VideoDeviceRunner() { stop(); }

bool VideoDeviceRunner::warmup() {
  TT_LOG_INFO("[VideoDevice] Warmup starting runner={}",
              videoConfig.pythonRunner);
  const bool ownsInterpreter = !Py_IsInitialized();
  if (ownsInterpreter) {
    py::initialize_interpreter(videoPythonSignalHandlers());
  }
  bool warmupFailed = false;
  try {
    py::gil_scoped_acquire gil;
    python->runner = makeRunner(videoConfig);
    python->runner->warmup();
    ready = true;
    TT_LOG_INFO("[VideoDevice] Warmup completed runner={}",
                videoConfig.pythonRunner);
  } catch (const py::error_already_set& e) {
    TT_LOG_ERROR("[VideoDevice] Warmup failed: {}", e.what());
    ready = false;
    warmupFailed = true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[VideoDevice] Warmup failed: {}", e.what());
    ready = false;
    warmupFailed = true;
  }
  // initialize_interpreter() leaves the GIL held by this thread. Release it on
  // the failure path too: the worker then calls stop(), which takes the GIL
  // from another thread, and with the GIL still held here the process never
  // exits (the service would not notice the dead worker).
  if (ownsInterpreter) {
    PyEval_SaveThread();
  }
  return !warmupFailed;
}

std::string VideoDeviceRunner::run(
    const tt::domain::VideoGenerateRequest& request) {
  if (!ready || !python || !python->runner) {
    throw std::runtime_error("[VideoDevice] runner is not warmed up");
  }
  py::gil_scoped_acquire gil;
  try {
    const tt::dit::DiTResult result =
        python->runner->generate(tt::runtime::toDiTRequest(request));
    const std::string path = requireMp4(result.path);
    tt::metrics::ServerMetrics::instance().onVideoStages(result.denoiseSeconds,
                                                         result.vaeSeconds);
    TT_LOG_INFO("[VideoDevice] task {} wrote {}", request.task_id, path);
    return path;
  } catch (const tt::dit::RequestError& e) {
    throw std::runtime_error(std::string("[VideoDevice] request refused: ") +
                             e.what());
  } catch (const py::error_already_set& e) {
    throw std::runtime_error(std::string("[VideoDevice] inference failed: ") +
                             e.what());
  }
}

void VideoDeviceRunner::stop() {
  if (!python || !python->runner) {
    ready = false;
    return;
  }
  if (!Py_IsInitialized()) {
    ready = false;
    python->runner
        .release();  // nothing can close a mesh without the interpreter
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    python->runner->close();
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[VideoDevice] closing the device raised: {}", e.what());
  } catch (const std::exception& e) {
    TT_LOG_WARN("[VideoDevice] closing the device raised: {}", e.what());
  }
  python->runner.reset();
  ready = false;
}

}  // namespace tt::runners
