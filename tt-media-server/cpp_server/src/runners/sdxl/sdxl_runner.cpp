// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sdxl/sdxl_runner.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runners/sdxl/sdxl_python_helpers.hpp"
#include "utils/image_codec.hpp"
#include "utils/logger.hpp"

namespace tt::runners::sdxl {

namespace {

using ::tt::utils::image_codec::encodeFloatChwToBase64;
using ::tt::utils::image_codec::Format;
using ::tt::utils::image_codec::parseFormat;

py::module_ importTorch() { return py::module_::import("torch"); }
py::module_ importTtnn() { return py::module_::import("ttnn"); }

/** Set OMP / MKL / TORCH thread caps only if unset, to avoid overriding the
 * operator's environment. */
void setupRunnerEnvironment() {
  setenv("OMP_NUM_THREADS", "2", 0);
  setenv("MKL_NUM_THREADS", "2", 0);
  setenv("TORCH_NUM_THREADS", "1", 0);
}

}  // namespace

SDXLBaseRunner::SDXLBaseRunner(const config::ImageConfig& config)
    : config_(config),
      batch_size_(config.max_batch_size),
      is_tensor_parallel_(!config.device_mesh_shape.empty() &&
                          config.device_mesh_shape[0] > 1) {
  setupRunnerEnvironment();
  // py::initialize_interpreter leaves the GIL held by this thread; on that
  // path we drop it explicitly below so the worker thread pool can acquire.
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
      "[SDXL] Constructed runner type={} mesh={}x{} galaxy={} weights='{}'",
      config::toString(config.runner_type),
      config.device_mesh_shape.size() > 0 ? config.device_mesh_shape[0] : 0,
      config.device_mesh_shape.size() > 1 ? config.device_mesh_shape[1] : 0,
      config.is_galaxy, config.model_weights_path);
}

SDXLBaseRunner::~SDXLBaseRunner() { stop(); }

void SDXLBaseRunner::stop() {
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
    auto warmupReq = warmupRequest();
    // Inlined run() body: the worker already holds the GIL, so calling
    // run() directly would double-acquire.
    auto prompts = processPrompts({warmupReq});
    injectLoraTriggers(prompts.prompts, warmupReq.lora_path);
    applyRequestSettings(warmupReq);
    ensureLoraState(warmupReq);
    applyModeSpecificSettings(warmupReq);

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
        generateInputTensors(warmupReq, promptEmbeds, addTextEmbeds);
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

  // Collapse to absent when all entries are null (Python's `[None] * N`).
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
    const py::object& imgsList, const domain::ImageGenerateRequest& request,
    int neededPadding) const {
  Format format = parseFormat(request.image_return_format.value_or("JPEG"));
  int quality = request.image_quality.value_or(85);

  std::vector<std::string> out;
  py::list imgs(imgsList);
  const int total = static_cast<int>(imgs.size());
  // Drop trailing padding entries inserted by processPrompts.
  const int realCount =
      std::max(0, static_cast<int>(batch_size_) - neededPadding);
  const int limit = std::min(total, realCount);
  for (int i = 0; i < limit; ++i) {
    py::object img = imgs[py::int_(i)];
    py::object tensor = img;
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

std::vector<std::string> SDXLBaseRunner::run(
    const domain::ImageGenerateRequest& request) {
  if (!initialized_) {
    throw std::runtime_error("[SDXL] Runner not initialized");
  }

  py::gil_scoped_acquire acquire;
  try {
    auto prompts = processPrompts({request});
    injectLoraTriggers(prompts.prompts, request.lora_path);
    applyRequestSettings(request);
    ensureLoraState(request);
    applyModeSpecificSettings(request);

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
        generateInputTensors(request, promptEmbeds, addTextEmbeds);
    prepareInputTensorsForIteration(tensors);
    tt_sdxl_.attr("compile_image_processing")();

    py::object imgs = tt_sdxl_.attr("generate_images")();
    return postProcessImages(imgs, request, prompts.needed_padding);
  } catch (const py::error_already_set& e) {
    throw std::runtime_error(std::string("[SDXL] inference failed: ") +
                             e.what());
  }
}

}  // namespace tt::runners::sdxl
