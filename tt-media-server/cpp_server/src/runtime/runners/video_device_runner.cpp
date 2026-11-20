// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/video_device_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "utils/logger.hpp"
#include "utils/video_output_dir.hpp"

namespace py = pybind11;

namespace tt::runners {

namespace {

void setEnv(const char* key, const std::string& value) {
  if (value.empty()) return;
  ::setenv(key, value.c_str(), 1);
}

// Python VideoManager reads TT_VIDEO_OUTPUT_DIR once, at import. The binary
// lives in cpp_server/build, so the sibling videos directory is
// cpp_server/videos.
void setVideoOutputDir() {
  if (const char* existing = std::getenv("TT_VIDEO_OUTPUT_DIR")) {
    if (*existing) return;
  }
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path dir = tt::utils::resolveVideoOutputDir();
  fs::create_directories(dir, ec);
  setEnv("TT_VIDEO_OUTPUT_DIR", dir.string());
  TT_LOG_INFO("[VideoDevice] Video output directory: {}", dir.string());
}

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
  setEnv("TT_DIT_CACHE_DIR", dir.string());
  TT_LOG_INFO("[VideoDevice] tt_dit weight cache: {}", dir.string());
}

void prependEntry(py::list sysPath, const std::string& value) {
  if (value.empty()) return;
  for (const auto& entry : sysPath) {
    if (py::str(entry).cast<std::string>() == value) return;
  }
  sysPath.attr("insert")(0, py::str(value));
}

void prependSysPath(const char* envName) {
  const char* value = std::getenv(envName);
  if (!value || !*value) return;
  prependEntry(py::module_::import("sys").attr("path"), value);
}

// The embedded interpreter does not pick up the shell PYTHONPATH, and a raw
// sys.path insert does not run .pth files. ttnn is an editable install: its
// .pth file maps the package onto the tt-metal checkout. site.addsitedir
// processes those files the same way the metal virtualenv does.
void addMetalSitePackages() {
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!metalHome || !*metalHome) return;
  namespace fs = std::filesystem;
  const fs::path libDir = fs::path(metalHome) / "python_env" / "lib";
  std::error_code ec;
  if (!fs::is_directory(libDir, ec)) return;
  py::module_ site = py::module_::import("site");
  for (const fs::directory_entry& child : fs::directory_iterator(libDir, ec)) {
    if (ec) return;
    const std::string name = child.path().filename().string();
    if (name.rfind("python", 0) != 0) continue;
    const fs::path sitePackages = child.path() / "site-packages";
    if (!fs::is_directory(sitePackages, ec)) continue;
    site.attr("addsitedir")(sitePackages.string());
    TT_LOG_INFO("[VideoDevice] Added metal site-packages: {}",
                sitePackages.string());
  }
}

void preparePythonPath() {
  const char* pythonPath = std::getenv("TT_PYTHON_PATH");
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!pythonPath || !*pythonPath) {
    throw std::runtime_error(
        "[VideoDevice] TT_PYTHON_PATH is not set. Point it at the "
        "tt-media-server directory.");
  }
  if (!metalHome || !*metalHome) {
    throw std::runtime_error(
        "[VideoDevice] TT_METAL_HOME is not set. Point it at the tt-metal "
        "checkout.");
  }
  prependSysPath("TT_METAL_HOME");
  prependSysPath("TT_PYTHON_PATH");
  addMetalSitePackages();
}

py::object buildPythonRequest(const tt::domain::VideoGenerateRequest& request) {
  py::module_ domain = py::module_::import("domain.video_generate_request");
  py::dict fields;
  fields["prompt"] = request.prompt;
  if (request.negativePrompt) {
    fields["negative_prompt"] = *request.negativePrompt;
  }
  if (request.numInferenceSteps) {
    fields["num_inference_steps"] = *request.numInferenceSteps;
  }
  if (request.seed) fields["seed"] = *request.seed;
  if (request.aspectRatio) fields["aspect_ratio"] = *request.aspectRatio;
  if (request.height) fields["height"] = *request.height;
  if (request.width) fields["width"] = *request.width;
  if (request.fps) fields["fps"] = *request.fps;
  if (request.duration) fields["duration"] = *request.duration;
  if (request.numFrames) fields["num_frames"] = *request.numFrames;
  return domain.attr("VideoGenerateRequest")(**fields);
}

std::string materializeVideo(const py::object& result, int fps) {
  if (result.is_none()) {
    throw std::runtime_error("[VideoDevice] runner returned no frames");
  }
  if (py::isinstance<py::str>(result)) {
    return result.cast<std::string>();
  }

  py::module_ videoManager = py::module_::import("utils.video_manager");
  py::object manager = videoManager.attr("VideoManager")();
  py::object audioResult = videoManager.attr("VideoAudioResult");
  if (py::isinstance(result, audioResult)) {
    return manager
        .attr("export_to_mp4_with_audio")(
            result.attr("frames"), result.attr("audio"),
            result.attr("sampling_rate"), py::arg("fps") = result.attr("fps"),
            py::arg("pixel_format") = result.attr("pixel_format"))
        .cast<std::string>();
  }

  py::object frames = result;
  if (py::isinstance<py::list>(frames) || py::isinstance<py::tuple>(frames)) {
    if (py::len(frames) == 0) {
      throw std::runtime_error("[VideoDevice] runner returned an empty batch");
    }
    frames = frames[py::int_(0)];
  }
  return manager.attr("export_to_mp4")(frames, fps).cast<std::string>();
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

}  // namespace

struct VideoDeviceRunner::PythonState {
  py::object runner;
  py::object loop;
};

VideoDeviceRunner::VideoDeviceRunner(config::VideoConfig config)
    : videoConfig(std::move(config)), python(std::make_unique<PythonState>()) {
  if (videoConfig.pythonRunner.empty()) {
    throw std::runtime_error(
        "[VideoDevice] pythonRunner is empty. Set MODEL_RUNNER, for example "
        "tt-wan2.2.");
  }
  setVideoOutputDir();
  setDitCacheDir();
  setEnv("MODEL_SERVICE", "video");
  setEnv("MODEL_RUNNER", videoConfig.pythonRunner);
  setEnv("TT_VISIBLE_DEVICES", videoConfig.visible_devices);
  // Leave DEVICE_MESH_SHAPE unset. Python settings resolves the mesh from
  // MODEL and DEVICE; the C++ default "1,1" is not a tuple and pydantic
  // rejects it.
  if (const char* mesh = std::getenv("DEVICE_MESH_SHAPE")) {
    ::unsetenv("DEVICE_MESH_SHAPE");
    TT_LOG_INFO(
        "[VideoDevice] Ignored DEVICE_MESH_SHAPE='{}'; Python settings "
        "chooses the mesh",
        mesh);
  }
  setEnv("MODEL_WEIGHTS_PATH", videoConfig.model_weights_path);
  setEnv("DEVICE", videoConfig.device);
  setEnv("WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS",
         std::to_string(videoConfig.weights_distribution_timeout_seconds));
  TT_LOG_INFO(
      "[VideoDevice] Constructed worker={} runner={} devices='{}' weights='{}'",
      videoConfig.worker_id, videoConfig.pythonRunner,
      videoConfig.visible_devices, videoConfig.model_weights_path);
}

VideoDeviceRunner::~VideoDeviceRunner() { stop(); }

bool VideoDeviceRunner::warmup() {
  TT_LOG_INFO("[VideoDevice] Warmup starting runner={}",
              videoConfig.pythonRunner);
  const bool ownsInterpreter = !Py_IsInitialized();
  if (ownsInterpreter) {
    py::initialize_interpreter();
  }
  bool warmupFailed = false;
  try {
    py::gil_scoped_acquire gil;
    preparePythonPath();
    py::module_ fabric = py::module_::import("tt_model_runners.runner_fabric");
    // The Python server passes the worker's device group ("0,1,2,3") as the
    // runner id, and setup_runner_environment() copies it into
    // TT_VISIBLE_DEVICES. Pass the same here so the shared Python code works
    // unchanged for a mesh worker. With no group configured, fall back to the
    // worker index as before.
    const std::string runnerId = videoConfig.visible_devices.empty()
                                     ? std::to_string(videoConfig.worker_id)
                                     : videoConfig.visible_devices;
    python->runner = fabric.attr("get_device_runner")(runnerId);
    python->runner.attr("set_device")();

    py::module_ asyncio = py::module_::import("asyncio");
    python->loop = asyncio.attr("new_event_loop")();
    asyncio.attr("set_event_loop")(python->loop);
    py::object warmupOk = python->loop.attr("run_until_complete")(
        python->runner.attr("warmup")());
    if (!warmupOk.cast<bool>()) {
      throw std::runtime_error("[VideoDevice] warmup reported not-ready for " +
                               videoConfig.pythonRunner);
    }
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
  // the failure path too: the worker then calls stop(), which takes the GIL from
  // another thread, and with the GIL still held here the process never exits
  // (the service would not notice the dead worker).
  if (ownsInterpreter) {
    PyEval_SaveThread();
  }
  return !warmupFailed;
}

std::string VideoDeviceRunner::run(
    const tt::domain::VideoGenerateRequest& request) {
  if (!ready || !python || python->runner.is_none()) {
    throw std::runtime_error("[VideoDevice] runner is not warmed up");
  }
  py::gil_scoped_acquire gil;
  try {
    py::object pyRequest = buildPythonRequest(request);
    py::list batch;
    batch.append(pyRequest);
    py::object result = python->runner.attr("run")(batch);
    const int fps = request.fps.has_value() && *request.fps > 0.0F
                        ? static_cast<int>(*request.fps)
                        : 16;
    const std::string path = requireMp4(materializeVideo(result, fps));
    TT_LOG_INFO("[VideoDevice] task {} wrote {}", request.task_id, path);
    return path;
  } catch (const py::error_already_set& e) {
    throw std::runtime_error(std::string("[VideoDevice] inference failed: ") +
                             e.what());
  }
}

void VideoDeviceRunner::stop() {
  // A failed warmup can leave `runner` (and `loop`) as null handles, not as
  // Python None; is_none() is false for a null handle and attr() would crash.
  if (!python || !python->runner || python->runner.is_none()) {
    ready = false;
    return;
  }
  if (!Py_IsInitialized()) {
    ready = false;
    return;
  }
  py::gil_scoped_acquire gil;
  try {
    python->runner.attr("close_device")();
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[VideoDevice] close_device raised: {}", e.what());
  }
  try {
    if (python->loop && !python->loop.is_none()) {
      python->loop.attr("close")();
    }
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[VideoDevice] event loop close raised: {}", e.what());
  }
  python->runner = py::object();
  python->loop = py::object();
  ready = false;
}

}  // namespace tt::runners
