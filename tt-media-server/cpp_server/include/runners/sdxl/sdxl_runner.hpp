// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "runners/media_runner.hpp"

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
class SDXLBaseRunner : public IMediaRunner<domain::ImageGenerateRequest,
                                           std::vector<std::string>> {
 public:
  ~SDXLBaseRunner() override;

  SDXLBaseRunner(const SDXLBaseRunner&) = delete;
  SDXLBaseRunner& operator=(const SDXLBaseRunner&) = delete;

  bool warmup() override;
  /** Enqueue the request onto the batcher and block until the result (or an
   *  exception) is ready. Concurrent callers are coalesced into batches up to
   *  `max_batch_size` by the consumer thread; FIFO order is preserved. */
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
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) = 0;

  virtual domain::ImageGenerateRequest warmupRequest() const = 0;

  virtual void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& /*request*/) {}

  /** Run a callable on a detached worker with a hard timeout. The caller
   * MUST NOT hold the GIL — the worker re-acquires it. After a timeout the
   * runner is unhealthy. */
  static void runWithTimeout(const std::string& tag, unsigned timeoutSeconds,
                             const std::function<void()>& work);

  /** Encode the first `batchCount` entries of `imgsList` to base64 strings
   *  using the encoding hints from `formatRequest`. Trailing
   *  `max_batch_size - batchCount` entries are dummy paddings produced by
   *  `processPrompts`. */
  std::vector<std::string> postProcessImages(
      const py::object& imgsList,
      const domain::ImageGenerateRequest& formatRequest, int batchCount) const;

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

  void injectLoraTriggers(std::vector<std::string>& prompts,
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

 private:
  /** One in-flight request waiting for its slice of a batched pipeline run. */
  struct BatchSlot {
    domain::ImageGenerateRequest request;
    std::promise<std::vector<std::string>> promise;
  };

  /** True when `a` and `b` can be packed into the same pipeline call.
   *  Compares every field that influences pipeline state (scheduler steps,
   *  guidance, lora, strength, sigmas/timesteps) or output encoding. Returns
   *  false when either side has a LoRA configured (load_lora_weights mutates
   *  pipeline weights, so cross-LoRA batches would corrupt each other's
   *  outputs even when paths match). */
  static bool areBatchCompatible(const domain::ImageGenerateRequest& a,
                                 const domain::ImageGenerateRequest& b);

  /** Consumer loop running on `batcher_thread_`. Owns the embedded
   *  interpreter for inference: pops one slot, drains compatible slots up to
   *  `max_batch_size` while preserving FIFO across incompatible groups, then
   *  hands the batch to `runBatch`. */
  void batcherLoop();

  /** Single pipeline pass over `batch`. On success fulfills each slot's
   *  promise with that slot's image; on failure surfaces the same exception
   *  to every slot in the batch. */
  void runBatch(std::vector<BatchSlot>& batch);

  // Replaces the old per-call run_mutex_. The batcher_thread_ is the sole
  // owner of pipeline state, so no in-Python mutex is needed.
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<BatchSlot> queue_;
  std::atomic<bool> batcher_stop_{false};
  std::thread batcher_thread_;
};

class SDXLGenerateRunner : public SDXLBaseRunner {
 public:
  explicit SDXLGenerateRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLGenerateRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
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
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
  void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& request) override;

  /** base64 -> PIL -> diffusers image_processor.preprocess; returns torch
   * tensor (1, C, H, W). */
  py::object preprocessImage(const std::string& base64Image) const;

  /** Stack per-request preprocessed images and pad up to `max_batch_size`
   *  with copies of the first slot (the pipeline still runs on the padded
   *  rows, the outputs are dropped in `postProcessImages`). */
  py::object stackImageBatch(
      const std::vector<domain::ImageGenerateRequest>& requests) const;
};

class SDXLEditRunner : public SDXLImageToImageRunner {
 public:
  explicit SDXLEditRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLEditRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;

 private:
  py::object preprocessMask(const std::string& base64Mask) const;

  /** Like `stackImageBatch` but for masks. */
  py::object stackMaskBatch(
      const std::vector<domain::ImageGenerateRequest>& requests) const;
};

}  // namespace tt::runners::sdxl
