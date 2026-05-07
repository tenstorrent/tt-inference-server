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
 * Orchestrates the SDXL pipeline entirely in C++ (device setup, prompt prep,
 * LoRA state, post-processing) and delegates the heavy lifting to the
 * tt-metal SDXL pipeline via pybind11. The embedded interpreter is left
 * running between calls; all Python interactions are guarded by
 * `py::gil_scoped_acquire`.
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

  virtual py::object loadDiffusersPipeline() = 0;

  /** Distribute weights to device, build `self.tt_sdxl`. */
  virtual void distributeBlock() = 0;

  /** Call `tt_sdxl.prepare_input_tensors([...])` in the order expected by
   * the specific pipeline. */
  virtual void prepareInputTensorsForIteration(py::object tensors) = 0;

  virtual py::object generateInputTensors(
      const domain::ImageGenerateRequest& request, py::object promptEmbeds,
      py::object addTextEmbeds) = 0;

  virtual domain::ImageGenerateRequest warmupRequest() const = 0;

  virtual void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& /*request*/) {}

  /** Run a callable on a detached worker with a hard timeout. The caller
   * MUST NOT hold the GIL — the worker re-acquires it. After a timeout the
   * runner is unhealthy. */
  static void runWithTimeout(const std::string& tag, unsigned timeoutSeconds,
                              const std::function<void()>& work);

  /** Encode each (C,H,W) torch tensor to a base64 string and drop the
   * trailing `neededPadding` entries (dummy padding from processPrompts). */
  std::vector<std::string> postProcessImages(
      const py::object& imgsList,
      const domain::ImageGenerateRequest& request, int neededPadding) const;

  struct PromptPack {
    std::vector<std::string> prompts;
    std::optional<std::vector<std::string>> negative_prompts;
    std::optional<std::vector<std::string>> prompts_2;
    std::optional<std::string> negative_prompt_2;
    int needed_padding = 0;
  };
  /** Pad prompts to max_batch_size. */
  PromptPack processPrompts(
      const std::vector<domain::ImageGenerateRequest>& requests) const;

  void injectLoraTriggers(
      std::vector<std::string>& prompts,
      const std::optional<std::string>& loraPath) const;

  void applyRequestSettings(const domain::ImageGenerateRequest& request);

  void ensureLoraState(const domain::ImageGenerateRequest& request);

  void initDevice();

  void runFullWarmup();

  py::dict pipelineDeviceParams();

  config::ImageConfig config_;
  size_t batch_size_ = 1;
  bool is_tensor_parallel_ = false;
  bool initialized_ = false;

  // Held GIL-side; null until `warmup()` succeeds.
  py::object ttnn_device_;
  py::object pipeline_;
  py::object tt_sdxl_;
  py::object torch_module_;
  py::object ttnn_module_;

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

  /** base64 -> PIL -> diffusers image_processor.preprocess; returns torch
   * tensor (1, C, H, W). */
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
