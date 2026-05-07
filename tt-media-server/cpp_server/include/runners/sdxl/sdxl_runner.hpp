// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "runners/image_runner.hpp"

namespace tt::runners::sdxl {

namespace py = pybind11;

/**
 * C++ port of `tt_model_runners.base_sdxl_runner.BaseSDXLRunner`.
 *
 * The runner orchestrates the SDXL pipeline entirely in C++:
 *   1. Set up sys.path (TT_PYTHON_PATH, TT_METAL_HOME).
 *   2. Configure the fabric (`ttnn.set_fabric_config`).
 *   3. Open the mesh device (`ttnn.open_mesh_device`).
 *   4. Load the diffusers pipeline (`DiffusionPipeline.from_pretrained`).
 *   5. Construct the tt-metal SDXL pipeline
 *      (`TtSDXL{,Img2Img,Inpainting}Pipeline`).
 *   6. Per request:
 *        a. Pad prompts, inject LoRA triggers.
 *        b. Apply per-request settings (steps, guidance, ...).
 *        c. Update LoRA state (load/unload/fuse) if needed.
 *        d. Compile text encoders, encode prompts.
 *        e. (img2img/edit) Preprocess input image / mask.
 *        f. Generate / prepare input tensors, compile image processing.
 *        g. Run `tt_sdxl.generate_images()`.
 *        h. Encode each (C,H,W) torch tensor to base64 PNG/JPEG via the C++
 *           `image_codec` (no `pipeline.image_processor.postprocess` call).
 *
 * All Python interactions are guarded by `py::gil_scoped_acquire`; the
 * interpreter is left running between calls (LlamaModelRunner pattern).
 */
class SDXLBaseRunner : public ImageRunner {
 public:
  ~SDXLBaseRunner() override;

  SDXLBaseRunner(const SDXLBaseRunner&) = delete;
  SDXLBaseRunner& operator=(const SDXLBaseRunner&) = delete;

  bool warmup() override;
  std::vector<std::string> run(
      const domain::ImageGenerateRequest& request) override;
  void stop() override;

 protected:
  explicit SDXLBaseRunner(const config::ImageConfig& config);

  /** Subclass: load `self.pipeline` via diffusers. */
  virtual py::object loadDiffusersPipeline() = 0;

  /** Subclass: distribute weights to device, build `self.tt_sdxl`. */
  virtual void distributeBlock() = 0;

  /** Subclass: prepare input tensors (calls
   * `tt_sdxl.prepare_input_tensors([...])` in the order expected by the
   * specific pipeline). */
  virtual void prepareInputTensorsForIteration(py::object tensors) = 0;

  /** Subclass: returns the torch tensor list / tuple consumed by
   * `prepareInputTensorsForIteration` after `generate_input_tensors`. */
  virtual py::object generateInputTensors(
      const domain::ImageGenerateRequest& request, py::object promptEmbeds,
      py::object addTextEmbeds) = 0;

  /** Subclass: per-mode warmup payload (1 dummy request). */
  virtual domain::ImageGenerateRequest warmupRequest() const = 0;

  /** Subclass hook to apply mode-specific request settings (e.g.
   * `set_strength()` for img2img). Default: no-op. */
  virtual void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& /*request*/) {}

  /** Run a callable synchronously inside `std::async` with timeout. The
   * caller must NOT hold the GIL — the worker re-acquires it. The detached
   * future is allowed to continue running on timeout (the runner is
   * unhealthy at that point anyway). */
  static void runWithTimeout(const std::string& tag, unsigned timeoutSeconds,
                              const std::function<void()>& work);

  /** Encode each (C,H,W) torch tensor in `imgsList` to a base64 string,
   * honoring `image_return_format` / `image_quality` from the request. The
   * trailing `needed_padding` images (the dummy padding inserted by
   * processPrompts to fill out a batch) are dropped, matching the
   * `if idx >= self.batch_size - needed_padding: break` slice in Python's
   * `BaseSDXLRunner._ttnn_inference`. */
  std::vector<std::string> postProcessImages(
      const py::object& imgsList,
      const domain::ImageGenerateRequest& request, int neededPadding) const;

  /** Pad prompts to max_batch_size and return (prompts, negative_prompts,
   * prompts_2, negative_prompt_2, needed_padding). */
  struct PromptPack {
    std::vector<std::string> prompts;
    std::optional<std::vector<std::string>> negative_prompts;
    std::optional<std::vector<std::string>> prompts_2;
    std::optional<std::string> negative_prompt_2;
    int needed_padding = 0;
  };
  PromptPack processPrompts(
      const std::vector<domain::ImageGenerateRequest>& requests) const;

  /** Inject LoRA triggers into prompts, mirroring
   * `BaseSDXLRunner._inject_lora_triggers`. */
  void injectLoraTriggers(
      std::vector<std::string>& prompts,
      const std::optional<std::string>& loraPath) const;

  /** Apply per-request SDXL settings via `set_*` setters on `tt_sdxl`. */
  void applyRequestSettings(const domain::ImageGenerateRequest& request);

  /** Load/unload/fuse the LoRA adapter to match the requested state. */
  void ensureLoraState(const domain::ImageGenerateRequest& request);

  /** Helpers for opening the device (split out for testability + so subclasses
   * can override params if they ever need to). */
  void initDevice();

  /** Common warmup wrapper: load weights, distribute, run dummy inference
   * with the timeouts mirrored from BaseSDXLRunner.warmup. */
  void runFullWarmup();

  /** Subclass hook: device parameters dict for `open_mesh_device`. SDXL
   * passes l1_small_size and (when TP) fabric_config. */
  py::dict pipelineDeviceParams();

  config::ImageConfig config_;
  size_t batch_size_ = 1;
  bool is_tensor_parallel_ = false;
  bool initialized_ = false;

  // Held GIL-side; null until `warmup()` succeeds.
  py::object ttnn_device_;       // ttnn mesh device handle
  py::object pipeline_;          // diffusers pipeline
  py::object tt_sdxl_;           // tt-metal pipeline
  py::object torch_module_;      // torch (cached for tensor ops)
  py::object ttnn_module_;       // ttnn

  // LoRA state, mirroring Python's `_current_lora_path` / `_current_lora_scale`
  std::optional<std::string> current_lora_path_;
  std::optional<float> current_lora_scale_;
};

class SDXLGenerateRunner : public SDXLBaseRunner {
 public:
  explicit SDXLGenerateRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLGenerateRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(const domain::ImageGenerateRequest& request,
                                  py::object promptEmbeds,
                                  py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
};

class SDXLImageToImageRunner : public SDXLBaseRunner {
 public:
  explicit SDXLImageToImageRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLImageToImageRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(const domain::ImageGenerateRequest& request,
                                  py::object promptEmbeds,
                                  py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
  void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& request) override;

  /** Decode base64 -> PIL -> diffusers image_processor.preprocess; result is
   * a torch tensor (1, C, H, W). */
  py::object preprocessImage(const std::string& base64Image) const;
};

class SDXLEditRunner : public SDXLImageToImageRunner {
 public:
  explicit SDXLEditRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLEditRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(const domain::ImageGenerateRequest& request,
                                  py::object promptEmbeds,
                                  py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;

 private:
  py::object preprocessMask(const std::string& base64Mask) const;
};

}  // namespace tt::runners::sdxl
