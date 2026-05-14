// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sdxl/sdxl_base_runner.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runners/sdxl/sdxl_python_helpers.hpp"
#include "utils/image_codec.hpp"
#include "utils/logger.hpp"

namespace {
// Window the batcher waits for additional compatible requests before running
// a non-full batch. Small relative to one SDXL step (~hundreds of ms).
constexpr std::chrono::milliseconds BATCH_FILL_WINDOW{5};
}  // namespace

namespace tt::runners::sdxl {

namespace {

using ::tt::utils::image_codec::encodeFloatChwToBase64;
using ::tt::utils::image_codec::Format;
using ::tt::utils::image_codec::parseFormat;

py::module_ importTorch() { return py::module_::import("torch"); }
py::module_ importTtnn() { return py::module_::import("ttnn"); }

void setupRunnerEnvironment(const config::ImageConfig& config) {
  setenv("OMP_NUM_THREADS", "2", 0);
  setenv("MKL_NUM_THREADS", "2", 0);
  setenv("TORCH_NUM_THREADS", "1", 0);
  if (!config.visible_devices.empty()) {
    setenv("TT_VISIBLE_DEVICES", config.visible_devices.c_str(), 1);
  }
  if (config.is_galaxy) {
    setenv("TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE", "7,7", 1);
  }
}

}  // namespace

SDXLBaseRunner::SDXLBaseRunner(const config::ImageConfig& config)
    : config_(config),
      batch_size_(config.max_batch_size),
      is_tensor_parallel_(!config.device_mesh_shape.empty() &&
                          config.device_mesh_shape[0] > 1) {
  setupRunnerEnvironment(config);
  // py::initialize_interpreter leaves the GIL held; release it so worker
  // threads can re-acquire on demand.
  const bool ownsInterpreter = !Py_IsInitialized();
  if (ownsInterpreter) {
    py::initialize_interpreter();
  }
  try {
    py::gil_scoped_acquire acquire;
    PythonHelpers::ensureSysPath();
    PythonHelpers::helpers();
    torch_module_ = importTorch();
    ttnn_module_ = importTtnn();
  } catch (const py::error_already_set& e) {
    throw std::runtime_error(std::string("[SDXL] Failed to import Python deps "
                                         "(torch / ttnn / huggingface_hub): ") +
                             e.what());
  }
  if (ownsInterpreter) {
    PyEval_SaveThread();
  }
  TT_LOG_INFO(
      "[SDXL] Constructed runner type={} mesh={}x{} galaxy={} "
      "visible_devices='{}' weights='{}' max_batch_size={}",
      config::toString(config.runner_type),
      config.device_mesh_shape.size() > 0 ? config.device_mesh_shape[0] : 0,
      config.device_mesh_shape.size() > 1 ? config.device_mesh_shape[1] : 0,
      config.is_galaxy, config.visible_devices, config.model_weights_path,
      batch_size_);
}

SDXLBaseRunner::~SDXLBaseRunner() { stop(); }

void SDXLBaseRunner::stop() {
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    batcher_stop_.store(true, std::memory_order_release);
    auto pending = std::move(queue_);
    queue_.clear();
    for (auto& slot : pending) {
      try {
        slot.promise.set_exception(std::make_exception_ptr(
            std::runtime_error("[SDXL] Runner stopped before request ran")));
      } catch (const std::future_error&) {
      }
    }
  }
  queue_cv_.notify_all();
  if (batcher_thread_.joinable()) batcher_thread_.join();

  if (!initialized_) return;
  py::gil_scoped_acquire acquire;
  try {
    if (!ttnn_device_.is_none() && !ttnn_module_.is_none()) {
      TT_LOG_INFO("[SDXL] Closing mesh device");
      ttnn_module_.attr("close_mesh_device")(ttnn_device_);
    }
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[SDXL] close_mesh_device raised: {}", e.what());
  }
  pipeline_ = py::object();
  tt_sdxl_ = py::object();
  ttnn_device_ = py::object();
  initialized_ = false;
}

void SDXLBaseRunner::runWithTimeout(const std::string& tag,
                                    unsigned timeoutSeconds,
                                    const std::function<void()>& work) {
  const auto start = std::chrono::steady_clock::now();
  {
    py::gil_scoped_acquire acquire;
    work();
  }
  if (timeoutSeconds == 0) {
    return;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - start);
  if (elapsed > std::chrono::seconds(timeoutSeconds)) {
    throw std::runtime_error("[SDXL] " + tag + " timed out after " +
                             std::to_string(timeoutSeconds) + "s");
  }
}

py::dict SDXLBaseRunner::pipelineDeviceParams() {
  py::dict params;
  // Pull constants from tt-metal so we track whatever it currently ships.
  py::module_ testCommon = py::module_::import(
      "models.demos.stable_diffusion_xl_base.tests.test_common");
  params["l1_small_size"] = testCommon.attr("SDXL_L1_SMALL_SIZE");
  if (is_tensor_parallel_) {
    params["fabric_config"] = testCommon.attr("SDXL_FABRIC_CONFIG");
  }
  return params;
}

void SDXLBaseRunner::initDevice() {
  py::dict params = pipelineDeviceParams();

  // Fabric config must be set BEFORE open_mesh_device.
  py::object fabricConfig = py::none();
  if (params.contains("fabric_config")) {
    fabricConfig = params["fabric_config"];
    params.attr("__delitem__")("fabric_config");
  }
  for (const char* key :
       {"dispatch_core_axis", "dispatch_core_type", "fabric_tensix_config"}) {
    if (params.contains(key)) {
      params.attr("__delitem__")(key);
    }
  }

  if (!fabricConfig.is_none()) {
    try {
      ttnn_module_.attr("set_fabric_config")(fabricConfig);
    } catch (const py::error_already_set& e) {
      throw std::runtime_error(
          std::string("[SDXL] set_fabric_config failed: ") + e.what());
    }
  }

  py::list shapeList;
  for (size_t v : config_.device_mesh_shape) shapeList.append(v);
  py::object meshShape = ttnn_module_.attr("MeshShape")(*py::tuple(shapeList));

  py::dict openKwargs = params;
  openKwargs["mesh_shape"] = meshShape;
  try {
    ttnn_device_ = ttnn_module_.attr("open_mesh_device")(**openKwargs);
  } catch (const py::error_already_set& e) {
    if (!fabricConfig.is_none()) {
      try {
        ttnn_module_.attr("set_fabric_config")(
            ttnn_module_.attr("FabricConfig").attr("DISABLED"));
      } catch (const py::error_already_set&) {
      }
    }
    throw std::runtime_error(std::string("[SDXL] open_mesh_device failed: ") +
                             e.what());
  }
  TT_LOG_INFO("[SDXL] Mesh device opened (num_devices={})",
              ttnn_device_.attr("get_num_devices")().cast<int>());
}

bool SDXLBaseRunner::warmup() {
  TT_LOG_INFO("[SDXL] Starting warmup (runner={})",
              config::toString(config_.runner_type));
  try {
    runFullWarmup();
    initialized_ = true;
    batcher_stop_.store(false, std::memory_order_release);
    batcher_thread_ = std::thread(&SDXLBaseRunner::batcherLoop, this);
    TT_LOG_INFO("[SDXL] Warmup completed");
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[SDXL] Warmup failed: {}", e.what());
    return false;
  }
}

void SDXLBaseRunner::runFullWarmup() {
  {
    py::gil_scoped_acquire acquire;
    initDevice();
    pipeline_ = loadDiffusersPipeline();
    TT_LOG_INFO("[SDXL] diffusers pipeline loaded");
  }

  // Distribute weights: minutes-long, GIL is released at this point and
  // runWithTimeout's worker re-acquires it.
  runWithTimeout("distribute_block",
                 config_.weights_distribution_timeout_seconds,
                 [&]() { distributeBlock(); });
  TT_LOG_INFO("[SDXL] tt-metal pipeline constructed");

  runWithTimeout("warmup_inference", 1000, [&]() {
    std::vector<domain::ImageGenerateRequest> singleton{warmupRequest()};
    const auto& head = singleton.front();
    auto prompts = processPrompts(singleton);
    injectLoraTriggers(prompts.prompts, head.lora_path);
    applyRequestSettings(head);
    ensureLoraState(head);
    applyModeSpecificSettings(head);

    tt_sdxl_.attr("compile_text_encoding")();
    py::object encoded = tt_sdxl_.attr("encode_prompts")(
        prompts.prompts,
        prompts.negative_prompts.has_value()
            ? py::cast(*prompts.negative_prompts)
            : py::none(),
        prompts.prompts_2.has_value() ? py::cast(*prompts.prompts_2)
                                      : py::none(),
        prompts.negative_prompt_2.has_value()
            ? py::cast(*prompts.negative_prompt_2)
            : py::none());
    py::object promptEmbeds = encoded[py::int_(0)];
    py::object addTextEmbeds = encoded[py::int_(1)];

    py::object tensors =
        generateInputTensors(singleton, promptEmbeds, addTextEmbeds);
    prepareInputTensorsForIteration(tensors);
    tt_sdxl_.attr("compile_image_processing")();
    tt_sdxl_.attr("generate_images")();
  });
}

SDXLBaseRunner::PromptPack SDXLBaseRunner::processPrompts(
    const std::vector<domain::ImageGenerateRequest>& requests) const {
  PromptPack pack;
  const int batchSize = static_cast<int>(requests.size());
  pack.needed_padding = static_cast<int>(batch_size_) - batchSize;
  if (pack.needed_padding < 0) pack.needed_padding = 0;

  pack.prompts.reserve(static_cast<size_t>(batch_size_));
  for (const auto& r : requests) pack.prompts.push_back(r.prompt);
  for (int i = 0; i < pack.needed_padding; ++i) pack.prompts.emplace_back("");

  // Mirror Python's `negative_prompts == [None]` collapse to absent.
  std::vector<std::string> negs;
  bool anyNeg = false;
  negs.reserve(static_cast<size_t>(batch_size_));
  for (const auto& r : requests) {
    if (r.negative_prompt.has_value()) {
      negs.push_back(*r.negative_prompt);
      anyNeg = true;
    } else {
      negs.emplace_back("");
    }
  }
  for (int i = 0; i < pack.needed_padding; ++i) negs.emplace_back("");
  if (anyNeg) pack.negative_prompts = std::move(negs);

  if (!requests.empty() && requests[0].prompt_2.has_value()) {
    std::vector<std::string> p2{*requests[0].prompt_2};
    for (int i = 0; i < pack.needed_padding; ++i) p2.emplace_back("");
    pack.prompts_2 = std::move(p2);
  }
  if (!requests.empty()) pack.negative_prompt_2 = requests[0].negative_prompt_2;
  return pack;
}

void SDXLBaseRunner::injectLoraTriggers(
    std::vector<std::string>& prompts,
    const std::optional<std::string>& loraPath) const {
  if (!loraPath.has_value() || loraPath->empty()) return;
  for (auto& p : prompts) {
    if (p.empty()) continue;
    p = PythonHelpers::injectLoraTrigger(p, *loraPath);
  }
}

void SDXLBaseRunner::applyRequestSettings(
    const domain::ImageGenerateRequest& request) {
  if (request.timesteps.has_value() && request.sigmas.has_value()) {
    throw std::invalid_argument(
        "Cannot pass both timesteps and sigmas. Choose one.");
  }
  if (request.num_inference_steps.has_value()) {
    tt_sdxl_.attr("set_num_inference_steps")(*request.num_inference_steps);
  }
  if (request.guidance_scale.has_value()) {
    tt_sdxl_.attr("set_guidance_scale")(*request.guidance_scale);
  }
  if (request.guidance_rescale.has_value()) {
    tt_sdxl_.attr("set_guidance_rescale")(*request.guidance_rescale);
  }
  if (request.crop_coords_top_left.has_value()) {
    py::tuple coords = py::make_tuple(request.crop_coords_top_left->first,
                                      request.crop_coords_top_left->second);
    tt_sdxl_.attr("set_crop_coords_top_left")(coords);
  }
}

void SDXLBaseRunner::ensureLoraState(
    const domain::ImageGenerateRequest& request) {
  const auto& requestedPath = request.lora_path;
  const auto& requestedScale = request.lora_scale;

  bool needsChange =
      (requestedPath != current_lora_path_) ||
      (requestedPath.has_value() && requestedScale != current_lora_scale_);
  if (!needsChange) return;

  if (current_lora_path_.has_value()) {
    TT_LOG_INFO("[SDXL] Unloading LoRA: {}", *current_lora_path_);
    tt_sdxl_.attr("unload_lora_weights")();
    current_lora_path_.reset();
    current_lora_scale_.reset();
  }

  if (requestedPath.has_value()) {
    try {
      std::string localPath = PythonHelpers::resolveLoraPath(*requestedPath);
      TT_LOG_INFO("[SDXL] Loading LoRA: {} (scale={})", *requestedPath,
                  requestedScale.value_or(1.0F));
      tt_sdxl_.attr("load_lora_weights")(localPath);
      tt_sdxl_.attr("fuse_lora")(
          requestedScale.has_value() ? py::cast(*requestedScale) : py::none());
      current_lora_path_ = requestedPath;
      current_lora_scale_ = requestedScale;
    } catch (const std::exception& e) {
      current_lora_path_.reset();
      current_lora_scale_.reset();
      throw std::runtime_error(std::string("Failed to load LoRA '") +
                               *requestedPath + "': " + e.what());
    }
  }
}

std::vector<std::string> SDXLBaseRunner::postProcessImages(
    const py::object& imgsList,
    const std::vector<domain::ImageGenerateRequest>& requests) const {
  std::vector<std::string> out;
  py::list imgs(imgsList);
  const int total = static_cast<int>(imgs.size());
  const int limit = std::min(total, static_cast<int>(requests.size()));
  out.reserve(static_cast<size_t>(std::max(0, limit)));
  for (int i = 0; i < limit; ++i) {
    const auto& req = requests[static_cast<size_t>(i)];
    Format format = parseFormat(req.image_return_format.value_or("JPEG"));
    int quality = req.image_quality.value_or(85);
    py::object tensor = imgs[py::int_(i)];
    tensor = tensor.attr("detach")();
    tensor =
        tensor.attr("to")(py::arg("dtype") = torch_module_.attr("float32"));
    tensor = tensor.attr("contiguous")();
    py::array_t<float> arr(tensor.attr("cpu")().attr("numpy")());
    if (arr.ndim() != 3) {
      throw std::runtime_error("[SDXL] Expected (C,H,W) tensor, got ndim=" +
                               std::to_string(arr.ndim()));
    }
    int channels = static_cast<int>(arr.shape(0));
    int height = static_cast<int>(arr.shape(1));
    int width = static_cast<int>(arr.shape(2));
    out.push_back(encodeFloatChwToBase64(arr.data(), channels, height, width,
                                         format, quality));
  }
  return out;
}

bool SDXLBaseRunner::areBatchCompatible(
    const domain::ImageGenerateRequest& a,
    const domain::ImageGenerateRequest& b) const {
  // Mirrors BaseSDXLRunner.is_request_batchable. Same-LoRA batches are
  // allowed: _ensure_lora_state is a no-op when path+scale already match.
  return a.num_inference_steps == b.num_inference_steps &&
         a.guidance_scale == b.guidance_scale &&
         a.guidance_rescale == b.guidance_rescale &&
         a.crop_coords_top_left == b.crop_coords_top_left &&
         a.timesteps == b.timesteps && a.sigmas == b.sigmas &&
         a.prompt_2 == b.prompt_2 &&
         a.negative_prompt_2 == b.negative_prompt_2 &&
         a.lora_path == b.lora_path && a.lora_scale == b.lora_scale;
}

void SDXLBaseRunner::batcherLoop() {
  while (true) {
    std::vector<BatchSlot> batch;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [&] {
        return batcher_stop_.load(std::memory_order_acquire) ||
               !queue_.empty();
      });
      if (batcher_stop_.load(std::memory_order_acquire)) return;

      batch.push_back(std::move(queue_.front()));
      queue_.pop_front();

      if (batch_size_ > 1) {
        const auto deadline =
            std::chrono::steady_clock::now() + BATCH_FILL_WINDOW;
        while (batch.size() < batch_size_) {
          if (!queue_.empty()) {
            if (areBatchCompatible(batch.front().request,
                                   queue_.front().request)) {
              batch.push_back(std::move(queue_.front()));
              queue_.pop_front();
              continue;
            }
            // Stop on incompatible head to preserve FIFO across groups.
            break;
          }
          if (std::chrono::steady_clock::now() >= deadline) break;
          queue_cv_.wait_until(lk, deadline, [&] {
            return batcher_stop_.load(std::memory_order_acquire) ||
                   !queue_.empty();
          });
          if (batcher_stop_.load(std::memory_order_acquire)) break;
        }
      }
    }
    runBatch(batch);
  }
}

void SDXLBaseRunner::runBatch(std::vector<BatchSlot>& batch) {
  if (batch.empty()) return;

  std::vector<std::string> images;
  std::exception_ptr error;
  try {
    std::vector<domain::ImageGenerateRequest> requests;
    requests.reserve(batch.size());
    for (auto& slot : batch) requests.push_back(slot.request);

    py::gil_scoped_acquire acquire;
    auto prompts = processPrompts(requests);
    // areBatchCompatible guarantees pipeline-state fields are uniform across
    // the batch; drive the pipeline from requests[0] (matches Python).
    const auto& head = requests.front();
    injectLoraTriggers(prompts.prompts, head.lora_path);
    applyRequestSettings(head);
    ensureLoraState(head);
    applyModeSpecificSettings(head);

    tt_sdxl_.attr("compile_text_encoding")();
    py::object encoded = tt_sdxl_.attr("encode_prompts")(
        prompts.prompts,
        prompts.negative_prompts.has_value()
            ? py::cast(*prompts.negative_prompts)
            : py::none(),
        prompts.prompts_2.has_value() ? py::cast(*prompts.prompts_2)
                                      : py::none(),
        prompts.negative_prompt_2.has_value()
            ? py::cast(*prompts.negative_prompt_2)
            : py::none());
    py::object promptEmbeds = encoded[py::int_(0)];
    py::object addTextEmbeds = encoded[py::int_(1)];

    py::object tensors =
        generateInputTensors(requests, promptEmbeds, addTextEmbeds);
    prepareInputTensorsForIteration(tensors);
    tt_sdxl_.attr("compile_image_processing")();

    py::object imgs = tt_sdxl_.attr("generate_images")();
    images = postProcessImages(imgs, requests);
  } catch (const py::error_already_set& e) {
    error = std::make_exception_ptr(std::runtime_error(
        std::string("[SDXL] inference failed: ") + e.what()));
  } catch (...) {
    error = std::current_exception();
  }

  if (error) {
    for (auto& slot : batch) {
      try {
        slot.promise.set_exception(error);
      } catch (const std::future_error&) {
      }
    }
    return;
  }

  if (images.size() != batch.size()) {
    auto eptr = std::make_exception_ptr(std::runtime_error(
        "[SDXL] postProcessImages returned " + std::to_string(images.size()) +
        " images for batch of " + std::to_string(batch.size())));
    for (auto& slot : batch) {
      try {
        slot.promise.set_exception(eptr);
      } catch (const std::future_error&) {
      }
    }
    return;
  }

  for (size_t i = 0; i < batch.size(); ++i) {
    try {
      batch[i].promise.set_value({images[i]});
    } catch (const std::future_error&) {
    }
  }
}

std::vector<std::string> SDXLBaseRunner::run(
    const domain::ImageGenerateRequest& request) {
  if (!initialized_) {
    throw std::runtime_error("[SDXL] Runner not initialized");
  }

  std::future<std::vector<std::string>> future;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    // Check under the lock to avoid racing with stop()'s drain pass.
    if (batcher_stop_.load(std::memory_order_acquire)) {
      throw std::runtime_error("[SDXL] Runner is stopping");
    }
    queue_.emplace_back();
    auto& slot = queue_.back();
    slot.request = request;
    future = slot.promise.get_future();
  }
  queue_cv_.notify_one();
  return future.get();
}

}  // namespace tt::runners::sdxl
