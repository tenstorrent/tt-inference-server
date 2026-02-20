// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "runners/embedding_runner.hpp"
#include "config/settings.hpp"

#include <Python.h>
#include <iostream>
#include <sstream>

namespace tt::runners {

// Logging helper
struct EmbedLog {
    std::ostringstream ss;
    const char* level;
    EmbedLog(const char* l) : level(l) {}
    ~EmbedLog() { std::cout << "[EmbeddingRunner][" << level << "] " << ss.str() << std::endl; }
    template<typename T>
    EmbedLog& operator<<(const T& v) { ss << v; return *this; }
};

#define EMBED_LOG_INFO EmbedLog("INFO")
#define EMBED_LOG_ERROR EmbedLog("ERROR")
#define EMBED_LOG_DEBUG EmbedLog("DEBUG")

/**
 * Implementation details hidden from header.
 */
struct EmbeddingRunner::Impl {
    bool python_initialized = false;
    PyObject* runner_module = nullptr;      // tt_model_runners.embedding_runner
    PyObject* runner_class = nullptr;       // BGELargeENRunner class
    PyObject* runner_instance = nullptr;    // BGELargeENRunner instance
    PyObject* request_module = nullptr;     // domain.text_embedding_request
    PyObject* request_class = nullptr;      // TextEmbeddingRequest class
    std::string device_id;

    explicit Impl(const std::string& dev_id) : device_id(dev_id) {}

    ~Impl() {
        cleanup();
    }

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

    bool init_python() {
        if (!Py_IsInitialized()) {
            Py_Initialize();
            python_initialized = true;
            EMBED_LOG_INFO << "Python interpreter initialized";
        }

        std::string python_path = tt::config::python_path();
        PyObject* sys_module = PyImport_ImportModule("sys");
        if (sys_module) {
            PyObject* sys_path = PyObject_GetAttrString(sys_module, "path");
            if (sys_path && PyList_Check(sys_path)) {
                PyObject* path_str = PyUnicode_FromString(python_path.c_str());
                // Insert at beginning to take precedence
                PyList_Insert(sys_path, 0, path_str);
                Py_DECREF(path_str);
                EMBED_LOG_INFO << "Added to sys.path: " << python_path.c_str();
            }
            Py_XDECREF(sys_path);
            Py_DECREF(sys_module);
        }

        return true;
    }

    bool import_modules() {
        // Import the embedding runner module
        runner_module = PyImport_ImportModule("tt_model_runners.embedding_runner");
        if (!runner_module) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to import tt_model_runners.embedding_runner";
            return false;
        }
        EMBED_LOG_INFO << "Imported tt_model_runners.embedding_runner";

        // Get BGELargeENRunner class
        runner_class = PyObject_GetAttrString(runner_module, "BGELargeENRunner");
        if (!runner_class) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get BGELargeENRunner class";
            return false;
        }
        EMBED_LOG_INFO << "Got BGELargeENRunner class";

        // Import TextEmbeddingRequest for creating request objects
        request_module = PyImport_ImportModule("domain.text_embedding_request");
        if (!request_module) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to import domain.text_embedding_request";
            return false;
        }

        request_class = PyObject_GetAttrString(request_module, "TextEmbeddingRequest");
        if (!request_class) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get TextEmbeddingRequest class";
            return false;
        }
        EMBED_LOG_INFO << "Got TextEmbeddingRequest class";

        return true;
    }

    bool create_runner_instance() {
        // Create BGELargeENRunner(device_id)
        PyObject* args = Py_BuildValue("(s)", device_id.c_str());
        runner_instance = PyObject_CallObject(runner_class, args);
        Py_DECREF(args);

        if (!runner_instance) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to create BGELargeENRunner instance";
            return false;
        }
        EMBED_LOG_INFO << "Created BGELargeENRunner instance for device " << device_id;

        return true;
    }

    bool call_set_device() {
        // Call runner.set_device() to initialize the TTNN device
        PyObject* set_device_method = PyObject_GetAttrString(runner_instance, "set_device");
        if (!set_device_method) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get set_device method";
            return false;
        }

        PyObject* result = PyObject_CallObject(set_device_method, nullptr);
        Py_DECREF(set_device_method);

        if (!result) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to call set_device()";
            return false;
        }

        Py_DECREF(result);
        EMBED_LOG_INFO << "set_device() completed successfully";
        return true;
    }

    bool call_warmup() {
        // Call runner.warmup() - it's async, so we need to handle coroutine
        PyObject* warmup_method = PyObject_GetAttrString(runner_instance, "warmup");
        if (!warmup_method) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get warmup method";
            return false;
        }

        // Import asyncio to run the coroutine
        PyObject* asyncio = PyImport_ImportModule("asyncio");
        if (!asyncio) {
            Py_DECREF(warmup_method);
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to import asyncio";
            return false;
        }

        // Get asyncio.run
        PyObject* asyncio_run = PyObject_GetAttrString(asyncio, "run");
        if (!asyncio_run) {
            Py_DECREF(asyncio);
            Py_DECREF(warmup_method);
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get asyncio.run";
            return false;
        }

        // Call warmup() to get coroutine
        PyObject* coro = PyObject_CallObject(warmup_method, nullptr);
        Py_DECREF(warmup_method);

        if (!coro) {
            Py_DECREF(asyncio_run);
            Py_DECREF(asyncio);
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to call warmup()";
            return false;
        }

        // Run the coroutine with asyncio.run(coro)
        PyObject* args = PyTuple_Pack(1, coro);
        PyObject* result = PyObject_CallObject(asyncio_run, args);
        Py_DECREF(args);
        Py_DECREF(coro);
        Py_DECREF(asyncio_run);
        Py_DECREF(asyncio);

        if (!result) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Warmup failed";
            return false;
        }

        bool success = PyObject_IsTrue(result);
        Py_DECREF(result);

        EMBED_LOG_INFO << "Warmup completed: " << (success ? "success" : "failed");
        return success;
    }

    std::vector<domain::EmbeddingResponse> run_inference(
        const std::vector<domain::EmbeddingRequest>& requests) {

        std::vector<domain::EmbeddingResponse> responses;

        // Build list of TextEmbeddingRequest objects
        PyObject* request_list = PyList_New(requests.size());

        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& req = requests[i];

            // Create TextEmbeddingRequest(model=..., input=...)
            PyObject* kwargs = PyDict_New();
            PyDict_SetItemString(kwargs, "model", PyUnicode_FromString(req.model.c_str()));
            PyDict_SetItemString(kwargs, "input", PyUnicode_FromString(req.input.c_str()));

            PyObject* py_request = PyObject_Call(request_class, PyTuple_New(0), kwargs);
            Py_DECREF(kwargs);

            if (!py_request) {
                PyErr_Print();
                EMBED_LOG_ERROR << "Failed to create TextEmbeddingRequest";
                Py_DECREF(request_list);
                return responses;
            }

            PyList_SetItem(request_list, i, py_request);  // Steals reference
        }

        // Call runner.run(request_list)
        PyObject* run_method = PyObject_GetAttrString(runner_instance, "run");
        if (!run_method) {
            PyErr_Print();
            EMBED_LOG_ERROR << "Failed to get run method";
            Py_DECREF(request_list);
            return responses;
        }

        PyObject* args = PyTuple_Pack(1, request_list);
        PyObject* result_list = PyObject_CallObject(run_method, args);
        Py_DECREF(args);
        Py_DECREF(run_method);
        Py_DECREF(request_list);

        if (!result_list) {
            PyErr_Print();
            EMBED_LOG_ERROR << "runner.run() failed";
            return responses;
        }

        // Parse results
        if (!PyList_Check(result_list)) {
            EMBED_LOG_ERROR << "Expected list result from runner.run()";
            Py_DECREF(result_list);
            return responses;
        }

        Py_ssize_t num_results = PyList_Size(result_list);
        for (Py_ssize_t i = 0; i < num_results; ++i) {
            PyObject* py_resp = PyList_GetItem(result_list, i);  // Borrowed reference

            domain::EmbeddingResponse resp;
            resp.task_id = requests[i].task_id;
            resp.model = requests[i].model;

            // Get embedding attribute
            PyObject* embedding_attr = PyObject_GetAttrString(py_resp, "embedding");
            if (embedding_attr && PyList_Check(embedding_attr)) {
                Py_ssize_t embed_size = PyList_Size(embedding_attr);
                resp.embedding.reserve(embed_size);
                for (Py_ssize_t j = 0; j < embed_size; ++j) {
                    PyObject* val = PyList_GetItem(embedding_attr, j);
                    resp.embedding.push_back(static_cast<float>(PyFloat_AsDouble(val)));
                }
            }
            Py_XDECREF(embedding_attr);

            // Get total_tokens attribute
            PyObject* tokens_attr = PyObject_GetAttrString(py_resp, "total_tokens");
            if (tokens_attr) {
                resp.total_tokens = static_cast<int>(PyLong_AsLong(tokens_attr));
                Py_DECREF(tokens_attr);
            }

            responses.push_back(std::move(resp));
        }

        Py_DECREF(result_list);

        EMBED_LOG_DEBUG << "Processed " << responses.size() << " embedding requests";
        return responses;
    }
};

// Public interface implementation

EmbeddingRunner::EmbeddingRunner(const std::string& device_id, int visible_device)
    : device_id_(device_id)
    , visible_device_(visible_device)
    , impl_(std::make_unique<Impl>(device_id)) {
    EMBED_LOG_INFO << "EmbeddingRunner created for device " << device_id
                   << " visible_device=" << visible_device_;
}

EmbeddingRunner::~EmbeddingRunner() {
    close();
}

bool EmbeddingRunner::warmup() {
    EMBED_LOG_INFO << "Starting warmup for device " << device_id_ << " visible_device=" << visible_device_;

    if (!impl_->init_python()) {
        return false;
    }

    if (!impl_->import_modules()) {
        return false;
    }

    if (!impl_->create_runner_instance()) {
        return false;
    }

    // Initialize the TTNN device before warmup
    if (!impl_->call_set_device()) {
        return false;
    }

    if (!impl_->call_warmup()) {
        return false;
    }

    EMBED_LOG_INFO << "Warmup complete for device " << device_id_;
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
        EMBED_LOG_ERROR << "Runner not initialized";
        return {};
    }

    return impl_->run_inference(requests);
}

} // namespace tt::runners
