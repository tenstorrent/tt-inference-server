// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <chrono>
#include <climits>
#include <string>
#include <sstream>
#include <iostream>
#include <Python.h>

#include "runners/base_device_runner.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

namespace tt::runners {

// Simple logging helpers
struct TTNNLogStream {
    std::ostringstream ss;
    const char* level;
    TTNNLogStream(const char* l) : level(l) {}
    ~TTNNLogStream() { std::cout << "[" << level << "] " << ss.str() << std::endl; }
    template<typename T>
    TTNNLogStream& operator<<(const T& v) { ss << v; return *this; }
};

#define TTNN_LOG_INFO TTNNLogStream("INFO")
#define TTNN_LOG_DEBUG TTNNLogStream("DEBUG")
#define TTNN_LOG_ERROR TTNNLogStream("ERROR")

/**
 * TTNN Test Runner for measuring device I/O overhead.
 *
 * This runner:
 * 1. Opens a TTNN mesh device with shape (1,1)
 * 2. Creates a tensor from a string
 * 3. Writes tensor to device with ttnn.to_device()
 * 4. Reads tensor from device max_tokens times with ttnn.from_device()
 * 5. Streams each read as a token
 *
 * Used to measure the overhead of device read/write operations.
 */
class TTNNTestRunner : public BaseDeviceRunner {
public:
    explicit TTNNTestRunner(const std::string& device_id)
        : BaseDeviceRunner(device_id)
        , python_initialized_(false)
        , mesh_device_(nullptr)
        , ttnn_module_(nullptr) {
        TTNN_LOG_INFO << "TTNNTestRunner initializing for device " << device_id;
    }

    ~TTNNTestRunner() {
        close();
    }

    bool warmup() override {
        TTNN_LOG_INFO << "TTNNTestRunner warmup starting for device " << device_id_;

        // Initialize Python interpreter
        if (!Py_IsInitialized()) {
            Py_Initialize();
            python_initialized_ = true;
            TTNN_LOG_INFO << "Python interpreter initialized";
        }

        // Import tt_lib first to trigger proper ttnn initialization
        PyObject* tt_lib_module = PyImport_ImportModule("tt_lib");
        if (!tt_lib_module) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to import tt_lib module";
            return false;
        }
        Py_DECREF(tt_lib_module);
        TTNN_LOG_INFO << "tt_lib module imported (triggers ttnn initialization)";

        // Import ttnn module
        ttnn_module_ = PyImport_ImportModule("ttnn");
        if (!ttnn_module_) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to import ttnn module";
            return false;
        }
        TTNN_LOG_INFO << "TTNN module imported successfully";

        // Open mesh device with shape (1, 1)
        if (!open_mesh_device()) {
            TTNN_LOG_ERROR << "Failed to open mesh device";
            return false;
        }

        TTNN_LOG_INFO << "TTNNTestRunner warmup complete for device " << device_id_;
        return true;
    }

    void close() override {
        if (mesh_device_) {
            close_mesh_device();
            mesh_device_ = nullptr;
        }

        Py_XDECREF(ttnn_module_);
        ttnn_module_ = nullptr;

        if (python_initialized_ && Py_IsInitialized()) {
            // Don't finalize Python as it may be used by other components
            // Py_Finalize();
            python_initialized_ = false;
        }
    }

    std::vector<domain::CompletionResponse> run(
        const std::vector<domain::CompletionRequest>& requests) override {

        std::vector<domain::CompletionResponse> responses;
        responses.reserve(requests.size());

        for (const auto& request : requests) {
            domain::CompletionResponse response;
            response.id = "cmpl-" + request.task_id;
            response.created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            response.model = request.model.value_or("ttnn-test-model");

            // Create tensor, write to device, read back
            std::string input_text = "hello";
            std::ostringstream result_text;

            for (int i = 0; i < request.max_tokens; ++i) {
                std::string token = read_from_device(input_text);
                result_text << token << " ";
            }

            domain::CompletionChoice choice;
            choice.text = result_text.str();
            choice.index = 0;
            choice.finish_reason = "stop";
            response.choices.push_back(choice);

            response.usage.completion_tokens = request.max_tokens;
            responses.push_back(response);
        }

        return responses;
    }

    void run_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const domain::StreamingChunkOutput&)> chunk_callback,
        std::function<void(const domain::FinalResultOutput&)> final_callback) override {

        auto start_time = std::chrono::high_resolution_clock::now();

        // Create input tensor once and write to device
        std::string input_text = "hello";
        PyObject* device_tensor = write_tensor_to_device(input_text);

        if (!device_tensor) {
            TTNN_LOG_ERROR << "Failed to write tensor to device";
            // Send error as final result
            domain::FinalResultOutput final_result;
            final_result.task_id = request.task_id;
            final_result.result.text = "[ERROR: Failed to write to device]";
            final_result.result.finish_reason = "error";
            final_result.return_result = true;
            final_callback(final_result);
            return;
        }

        // Read from device max_tokens times, streaming each read
        long long total_read_time_us = 0;
        long long total_from_device_time_us = 0;
        long long min_from_device_time_us = LLONG_MAX;
        long long max_from_device_time_us = 0;

        for (int i = 0; i < request.max_tokens; ++i) {
            auto read_start = std::chrono::high_resolution_clock::now();

            // Read tensor from device (from_device_time_us captures ONLY the ttnn.from_device call)
            long long from_device_time_us = 0;
            std::string token = read_tensor_from_device(device_tensor, from_device_time_us);

            auto read_end = std::chrono::high_resolution_clock::now();
            auto total_read_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                read_end - read_start
            ).count();

            // Track from_device stats (the actual device read)
            total_from_device_time_us += from_device_time_us;
            if (from_device_time_us < min_from_device_time_us) min_from_device_time_us = from_device_time_us;
            if (from_device_time_us > max_from_device_time_us) max_from_device_time_us = from_device_time_us;
            total_read_time_us += total_read_duration_us;

            // Create and emit chunk with the read data
            domain::StreamingChunkOutput chunk;
            chunk.task_id = request.task_id;
            chunk.chunk.text = token + "_" + std::to_string(i);
            chunk.chunk.index = i;

            chunk_callback(chunk);

            // Log periodically (every 1000 tokens)
            if ((i + 1) % 1000 == 0) {
                double avg_from_device = static_cast<double>(total_from_device_time_us) / (i + 1);
                double avg_total = static_cast<double>(total_read_time_us) / (i + 1);
                double overhead = avg_total - avg_from_device;
                TTNN_LOG_INFO << "Token " << (i + 1) << "/" << request.max_tokens
                              << " | from_device: " << from_device_time_us << " µs (avg: " << avg_from_device << " µs)"
                              << " | total_read: " << total_read_duration_us << " µs (avg: " << avg_total << " µs)"
                              << " | overhead: " << overhead << " µs";
            }
        }

        // Cleanup device tensor
        Py_XDECREF(device_tensor);

        // Send final result
        domain::FinalResultOutput final_result;
        final_result.task_id = request.task_id;
        final_result.result.text = "[DONE]";
        final_result.result.index = 0;
        final_result.result.finish_reason = "stop";
        final_result.return_result = true;

        final_callback(final_result);

        // Log actual performance
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time
        ).count();

        double actual_tokens_per_sec = (request.max_tokens * 1000000.0) / duration_us;
        double avg_total_time_us = static_cast<double>(total_read_time_us) / request.max_tokens;
        double avg_from_device_time_us = static_cast<double>(total_from_device_time_us) / request.max_tokens;
        double overhead_us = avg_total_time_us - avg_from_device_time_us;

        TTNN_LOG_INFO << "=== TTNNTestRunner Summary ===";
        TTNN_LOG_INFO << "  Total tokens: " << request.max_tokens;
        TTNN_LOG_INFO << "  Total time: " << duration_us << " µs (" << actual_tokens_per_sec << " tokens/sec)";
        TTNN_LOG_INFO << "  from_device avg: " << avg_from_device_time_us << " µs"
                      << " (min: " << min_from_device_time_us << ", max: " << max_from_device_time_us << ")";
        TTNN_LOG_INFO << "  total_read avg: " << avg_total_time_us << " µs";
        TTNN_LOG_INFO << "  overhead (to_torch + conversion): " << overhead_us << " µs per token";
    }

private:
    bool python_initialized_;
    PyObject* mesh_device_;
    PyObject* ttnn_module_;

    bool open_mesh_device() {
        // ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        PyObject* mesh_shape_class = PyObject_GetAttrString(ttnn_module_, "MeshShape");
        if (!mesh_shape_class) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to get MeshShape class";
            return false;
        }

        // Create MeshShape(1, 1)
        PyObject* mesh_shape_args = Py_BuildValue("(ii)", 1, 1);
        PyObject* mesh_shape = PyObject_CallObject(mesh_shape_class, mesh_shape_args);
        Py_DECREF(mesh_shape_args);
        Py_DECREF(mesh_shape_class);

        if (!mesh_shape) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to create MeshShape";
            return false;
        }

        // Call ttnn.open_mesh_device(mesh_shape=mesh_shape)
        PyObject* open_func = PyObject_GetAttrString(ttnn_module_, "open_mesh_device");
        if (!open_func) {
            Py_DECREF(mesh_shape);
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to get open_mesh_device function";
            return false;
        }

        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "mesh_shape", mesh_shape);

        mesh_device_ = PyObject_Call(open_func, PyTuple_New(0), kwargs);

        Py_DECREF(kwargs);
        Py_DECREF(mesh_shape);
        Py_DECREF(open_func);

        if (!mesh_device_) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to open mesh device";
            return false;
        }

        TTNN_LOG_INFO << "Mesh device opened successfully with shape (1, 1)";
        return true;
    }

    void close_mesh_device() {
        if (mesh_device_ && ttnn_module_) {
            PyObject* close_func = PyObject_GetAttrString(ttnn_module_, "close_mesh_device");
            if (close_func) {
                PyObject* result = PyObject_CallFunctionObjArgs(close_func, mesh_device_, NULL);
                Py_XDECREF(result);
                Py_DECREF(close_func);
            }
            Py_DECREF(mesh_device_);
            mesh_device_ = nullptr;
            TTNN_LOG_INFO << "Mesh device closed";
        }
    }

    PyObject* write_tensor_to_device(const std::string& text) {
        // Create a simple tensor from the text (as float values of ASCII codes)
        // Using ttnn.from_torch(torch.tensor([ord(c) for c in text], dtype=torch.float32))

        // Import torch
        PyObject* torch_module = PyImport_ImportModule("torch");
        if (!torch_module) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to import torch";
            return nullptr;
        }

        // Create list of ASCII values
        PyObject* ascii_list = PyList_New(text.size());
        for (size_t i = 0; i < text.size(); ++i) {
            PyList_SetItem(ascii_list, i, PyFloat_FromDouble(static_cast<double>(text[i])));
        }

        // Create torch tensor
        PyObject* tensor_func = PyObject_GetAttrString(torch_module, "tensor");
        PyObject* float32 = PyObject_GetAttrString(torch_module, "float32");
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "dtype", float32);

        PyObject* args = PyTuple_Pack(1, ascii_list);
        PyObject* torch_tensor = PyObject_Call(tensor_func, args, kwargs);

        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(float32);
        Py_DECREF(tensor_func);
        Py_DECREF(ascii_list);

        if (!torch_tensor) {
            PyErr_Print();
            Py_DECREF(torch_module);
            TTNN_LOG_ERROR << "Failed to create torch tensor";
            return nullptr;
        }

        // Convert to ttnn tensor: ttnn.from_torch(torch_tensor)
        PyObject* from_torch_func = PyObject_GetAttrString(ttnn_module_, "from_torch");
        if (!from_torch_func) {
            PyErr_Print();
            Py_DECREF(torch_tensor);
            Py_DECREF(torch_module);
            TTNN_LOG_ERROR << "Failed to get ttnn.from_torch";
            return nullptr;
        }

        PyObject* ttnn_tensor = PyObject_CallFunctionObjArgs(from_torch_func, torch_tensor, NULL);
        Py_DECREF(from_torch_func);
        Py_DECREF(torch_tensor);
        Py_DECREF(torch_module);

        if (!ttnn_tensor) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to convert to ttnn tensor";
            return nullptr;
        }

        // Write to device: ttnn.to_device(ttnn_tensor, mesh_device_)
        PyObject* to_device_func = PyObject_GetAttrString(ttnn_module_, "to_device");
        if (!to_device_func) {
            PyErr_Print();
            Py_DECREF(ttnn_tensor);
            TTNN_LOG_ERROR << "Failed to get ttnn.to_device";
            return nullptr;
        }

        PyObject* device_tensor = PyObject_CallFunctionObjArgs(to_device_func, ttnn_tensor, mesh_device_, NULL);
        Py_DECREF(to_device_func);
        Py_DECREF(ttnn_tensor);

        if (!device_tensor) {
            PyErr_Print();
            TTNN_LOG_ERROR << "Failed to write tensor to device";
            return nullptr;
        }

        return device_tensor;
    }

    std::string read_tensor_from_device(PyObject* device_tensor, long long& from_device_time_us) {
        // Read from device: ttnn.from_device(device_tensor)
        PyObject* from_device_func = PyObject_GetAttrString(ttnn_module_, "from_device");
        if (!from_device_func) {
            PyErr_Print();
            return "[ERROR: from_device not found]";
        }

        // Time ONLY the from_device call
        auto from_device_start = std::chrono::high_resolution_clock::now();
        PyObject* host_tensor = PyObject_CallFunctionObjArgs(from_device_func, device_tensor, NULL);
        auto from_device_end = std::chrono::high_resolution_clock::now();
        from_device_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            from_device_end - from_device_start
        ).count();

        Py_DECREF(from_device_func);

        if (!host_tensor) {
            PyErr_Print();
            return "[ERROR: from_device failed]";
        }

        // Convert back to torch: ttnn.to_torch(host_tensor)
        PyObject* to_torch_func = PyObject_GetAttrString(ttnn_module_, "to_torch");
        if (!to_torch_func) {
            PyErr_Print();
            Py_DECREF(host_tensor);
            return "[ERROR: to_torch not found]";
        }

        PyObject* torch_tensor = PyObject_CallFunctionObjArgs(to_torch_func, host_tensor, NULL);
        Py_DECREF(to_torch_func);
        Py_DECREF(host_tensor);

        if (!torch_tensor) {
            PyErr_Print();
            return "[ERROR: to_torch failed]";
        }

        // Convert tensor to list and then to string
        PyObject* tolist_method = PyObject_GetAttrString(torch_tensor, "tolist");
        if (!tolist_method) {
            Py_DECREF(torch_tensor);
            return "[ERROR: tolist not found]";
        }

        PyObject* values_list = PyObject_CallObject(tolist_method, NULL);
        Py_DECREF(tolist_method);
        Py_DECREF(torch_tensor);

        if (!values_list || !PyList_Check(values_list)) {
            Py_XDECREF(values_list);
            return "[ERROR: tolist failed]";
        }

        // Convert ASCII values back to string
        std::string result;
        Py_ssize_t size = PyList_Size(values_list);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(values_list, i);
            double val = PyFloat_AsDouble(item);
            result += static_cast<char>(static_cast<int>(val));
        }

        Py_DECREF(values_list);
        return result;
    }

    std::string read_from_device(const std::string& input) {
        PyObject* device_tensor = write_tensor_to_device(input);
        if (!device_tensor) {
            return "[ERROR]";
        }
        long long unused_time = 0;
        std::string result = read_tensor_from_device(device_tensor, unused_time);
        Py_DECREF(device_tensor);
        return result;
    }
};

#undef TTNN_LOG_INFO
#undef TTNN_LOG_DEBUG
#undef TTNN_LOG_ERROR

} // namespace tt::runners
