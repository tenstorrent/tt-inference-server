// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "runners/embedding_runner.hpp"

#include <Python.h>

#include <sstream>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

namespace {

void setDictString(PyObject* dict, const char* key, const std::string& value) {
  PyObject* str = PyUnicode_FromString(value.c_str());
  PyDict_SetItemString(dict, key, str);
  Py_DECREF(str);
}

}  // namespace

struct EmbeddingRunner::Impl {
  bool python_initialized = false;
  PyObject* runner_module = nullptr;    // tt_model_runners.embedding_runner
  PyObject* runner_class = nullptr;     // BGELargeENRunner class
  PyObject* runner_instance = nullptr;  // BGELargeENRunner instance
  PyObject* request_module = nullptr;   // domain.text_embedding_request
  PyObject* request_class = nullptr;    // TextEmbeddingRequest class
  std::string device_id;

  explicit Impl(const std::string& devId) : device_id(devId) {}

  ~Impl() { cleanup(); }

  void cleanup() {
    Py_XDECREF(runner_instance);
    Py_XDECREF(runner_class);
    Py_XDECREF(runner_module);
    Py_XDECREF(request_class);
    Py_XDECREF(request_module);

    runner_instance = nullptr;
    runner_class = nullptr;
    runner_module = nullptr;
    request_class = nullptr;
    request_module = nullptr;

    // Don't finalize Python - other components may use it
  }

  bool initPython() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      python_initialized = true;
      TT_LOG_INFO("[EmbeddingRunner] Python interpreter initialized");
    }

    std::string pythonPath = tt::config::pythonPath();
    PyObject* sysModule = PyImport_ImportModule("sys");
    if (sysModule) {
      PyObject* sysPath = PyObject_GetAttrString(sysModule, "path");
      if (sysPath && PyList_Check(sysPath)) {
        PyObject* pathStr = PyUnicode_FromString(pythonPath.c_str());
        // Insert at beginning to take precedence
        PyList_Insert(sysPath, 0, pathStr);
        Py_DECREF(pathStr);
        TT_LOG_INFO("[EmbeddingRunner] Added to sys.path: {}", pythonPath);
      }
      Py_XDECREF(sysPath);
      Py_DECREF(sysModule);
    }

    return true;
  }

  bool importModules() {
    // Import the embedding runner module
    runner_module = PyImport_ImportModule("tt_model_runners.embedding_runner");
    if (!runner_module) {
      PyErr_Print();
      TT_LOG_ERROR(
          "[EmbeddingRunner] Failed to import "
          "tt_model_runners.embedding_runner");
      return false;
    }
    TT_LOG_INFO("[EmbeddingRunner] Imported tt_model_runners.embedding_runner");

    // Get BGELargeENRunner class
    runner_class = PyObject_GetAttrString(runner_module, "BGELargeENRunner");
    if (!runner_class) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to get BGELargeENRunner class");
      return false;
    }
    TT_LOG_INFO("[EmbeddingRunner] Got BGELargeENRunner class");

    // Import TextEmbeddingRequest for creating request objects
    request_module = PyImport_ImportModule("domain.text_embedding_request");
    if (!request_module) {
      PyErr_Print();
      TT_LOG_ERROR(
          "[EmbeddingRunner] Failed to import domain.text_embedding_request");
      return false;
    }

    request_class =
        PyObject_GetAttrString(request_module, "TextEmbeddingRequest");
    if (!request_class) {
      PyErr_Print();
      TT_LOG_ERROR(
          "[EmbeddingRunner] Failed to get TextEmbeddingRequest class");
      return false;
    }
    TT_LOG_INFO("[EmbeddingRunner] Got TextEmbeddingRequest class");

    return true;
  }

  bool createRunnerInstance() {
    // Create BGELargeENRunner(device_id)
    PyObject* args = Py_BuildValue("(s)", device_id.c_str());
    runner_instance = PyObject_CallObject(runner_class, args);
    Py_DECREF(args);

    if (!runner_instance) {
      PyErr_Print();
      TT_LOG_ERROR(
          "[EmbeddingRunner] Failed to create BGELargeENRunner instance");
      return false;
    }
    TT_LOG_INFO(
        "[EmbeddingRunner] Created BGELargeENRunner instance for device {}",
        device_id);

    return true;
  }

  bool callSetDevice() {
    // Call runner.set_device() to initialize the TTNN device
    PyObject* setDeviceMethod =
        PyObject_GetAttrString(runner_instance, "set_device");
    if (!setDeviceMethod) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to get set_device method");
      return false;
    }

    PyObject* result = PyObject_CallObject(setDeviceMethod, nullptr);
    Py_DECREF(setDeviceMethod);

    if (!result) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to call set_device()");
      return false;
    }

    Py_DECREF(result);
    TT_LOG_INFO("[EmbeddingRunner] set_device() completed successfully");
    return true;
  }

  bool callWarmup() {
    // Call runner.warmup() - it's async, so we need to handle coroutine
    PyObject* warmupMethod = PyObject_GetAttrString(runner_instance, "warmup");
    if (!warmupMethod) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to get warmup method");
      return false;
    }

    // Import asyncio to run the coroutine
    PyObject* asyncio = PyImport_ImportModule("asyncio");
    if (!asyncio) {
      Py_DECREF(warmupMethod);
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to import asyncio");
      return false;
    }

    // Get asyncio.run
    PyObject* asyncioRun = PyObject_GetAttrString(asyncio, "run");
    if (!asyncioRun) {
      Py_DECREF(asyncio);
      Py_DECREF(warmupMethod);
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to get asyncio.run");
      return false;
    }

    // Call warmup() to get coroutine
    PyObject* coro = PyObject_CallObject(warmupMethod, nullptr);
    Py_DECREF(warmupMethod);

    if (!coro) {
      Py_DECREF(asyncioRun);
      Py_DECREF(asyncio);
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to call warmup()");
      return false;
    }

    // Run the coroutine with asyncio.run(coro)
    PyObject* args = PyTuple_Pack(1, coro);
    PyObject* result = PyObject_CallObject(asyncioRun, args);
    Py_DECREF(args);
    Py_DECREF(coro);
    Py_DECREF(asyncioRun);
    Py_DECREF(asyncio);

    if (!result) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Warmup failed");
      return false;
    }

    bool success = PyObject_IsTrue(result);
    Py_DECREF(result);

    TT_LOG_INFO("[EmbeddingRunner] Warmup completed: {}",
                (success ? "success" : "failed"));
    return success;
  }

  std::vector<domain::EmbeddingResponse> runInference(
      const std::vector<domain::EmbeddingRequest>& requests) {
    std::vector<domain::EmbeddingResponse> responses;

    PyObject* requestList = PyList_New(requests.size());
    PyObject* emptyArgs = PyTuple_New(0);

    for (size_t i = 0; i < requests.size(); ++i) {
      const auto& req = requests[i];

      PyObject* kwargs = PyDict_New();
      setDictString(kwargs, "model", req.model);
      setDictString(kwargs, "input", req.input);

      PyObject* pyRequest = PyObject_Call(request_class, emptyArgs, kwargs);
      Py_DECREF(kwargs);

      if (!pyRequest) {
        PyErr_Print();
        TT_LOG_ERROR("[EmbeddingRunner] Failed to create TextEmbeddingRequest");
        Py_DECREF(emptyArgs);
        Py_DECREF(requestList);
        return responses;
      }

      PyList_SetItem(requestList, i, pyRequest);  // Steals reference
    }
    Py_DECREF(emptyArgs);

    PyObject* runMethod = PyObject_GetAttrString(runner_instance, "run");
    if (!runMethod) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] Failed to get run method");
      Py_DECREF(requestList);
      return responses;
    }

    PyObject* args = PyTuple_Pack(1, requestList);
    PyObject* resultList = PyObject_CallObject(runMethod, args);
    Py_DECREF(args);
    Py_DECREF(runMethod);
    Py_DECREF(requestList);

    if (!resultList) {
      PyErr_Print();
      TT_LOG_ERROR("[EmbeddingRunner] runner.start() failed");
      return responses;
    }

    // Parse results
    if (!PyList_Check(resultList)) {
      TT_LOG_ERROR(
          "[EmbeddingRunner] Expected list result from runner.start()");
      Py_DECREF(resultList);
      return responses;
    }

    Py_ssize_t numResults = PyList_Size(resultList);
    for (Py_ssize_t i = 0; i < numResults; ++i) {
      PyObject* pyResp = PyList_GetItem(resultList, i);  // Borrowed reference

      domain::EmbeddingResponse resp(requests[i].task_id);
      resp.model = requests[i].model;

      // Get embedding attribute
      PyObject* embeddingAttr = PyObject_GetAttrString(pyResp, "embedding");
      if (embeddingAttr && PyList_Check(embeddingAttr)) {
        Py_ssize_t embedSize = PyList_Size(embeddingAttr);
        resp.embedding.reserve(embedSize);
        for (Py_ssize_t j = 0; j < embedSize; ++j) {
          PyObject* val = PyList_GetItem(embeddingAttr, j);
          resp.embedding.push_back(static_cast<float>(PyFloat_AsDouble(val)));
        }
      }
      Py_XDECREF(embeddingAttr);

      // Get total_tokens attribute
      PyObject* tokensAttr = PyObject_GetAttrString(pyResp, "total_tokens");
      if (tokensAttr) {
        resp.total_tokens = static_cast<int>(PyLong_AsLong(tokensAttr));
        Py_DECREF(tokensAttr);
      }

      responses.push_back(std::move(resp));
    }

    Py_DECREF(resultList);

    TT_LOG_DEBUG("[EmbeddingRunner] Processed {} embedding requests",
                 responses.size());
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

  if (!impl_->initPython()) {
    return false;
  }

  if (!impl_->importModules()) {
    return false;
  }

  if (!impl_->createRunnerInstance()) {
    return false;
  }

  // Initialize the TTNN device before warmup
  if (!impl_->callSetDevice()) {
    return false;
  }

  if (!impl_->callWarmup()) {
    return false;
  }

  TT_LOG_INFO("[EmbeddingRunner] Warmup complete for device {}", device_id_);
  return true;
}

void EmbeddingRunner::close() {
  if (impl_) {
    impl_->cleanup();
  }
}

std::vector<domain::EmbeddingResponse> EmbeddingRunner::run(
    const std::vector<domain::EmbeddingRequest>& requests) {
  if (!impl_ || !impl_->runner_instance) {
    TT_LOG_ERROR("[EmbeddingRunner] Runner not initialized");
    return {};
  }

  return impl_->runInference(requests);
}

// IRunner interface implementation
void EmbeddingRunner::run() {
  // For embedding runners, this could be a service loop
  // For now, we'll just do warmup since embeddings are request-response based
  if (!warmup()) {
    throw std::runtime_error("Failed to initialize EmbeddingRunner");
  }
  TT_LOG_INFO(
      "[EmbeddingRunner] EmbeddingRunner ready for requests on device {}",
      device_id_);
}

void EmbeddingRunner::stop() { close(); }

}  // namespace tt::runners
