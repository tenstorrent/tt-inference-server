// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/dit/dit_model.hpp"

/**
 * The model classes. Values (mesh shapes, fabric, trace regions, guidance,
 * step counts, resolutions) mirror the Python server's
 * tt_model_runners/dit_runners.py and config/constants.py; a unit test
 * compares the runner list and mesh shapes with that config. Keep the rest in
 * sync by hand.
 */
namespace tt::dit {

// -- Wan2.2 -----------------------------------------------------------------

constexpr int kWan22NumFrames = 81;

/** (height, width): 720p on Galaxy-class meshes, 480p on smaller ones. */
std::pair<int, int> wan22Resolution(MeshShape mesh);

/** Wan2.2 text-to-video. The base of the family: same resolution, fabric and
 *  call arguments; subclasses change the pipeline and what differs. */
class WanModel : public DiTModel {
 public:
  WanModel(RunnerConfig config, ModelSpec spec);
  std::string adapterName() const override { return "wan_t2v"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv& host) const override;

 protected:
  int height_;
  int width_;
  /** 0: the request decides. Distilled models run a fixed number of steps. */
  virtual int fixedSteps() const { return 0; }
  virtual double guidanceScale() const { return 4.0; }
  virtual double guidanceScale2() const { return 3.0; }
  /** Distilled models bake guidance in and take no negative prompt. */
  virtual bool usesNegativePrompt() const { return true; }
  /** Trace region on a Blackhole Galaxy-class mesh, if larger than the shared
   *  default. */
  virtual long largeBhTraceRegion() const { return 0; }
  int stepsFor(const DiTRequest& request) const override;
};

/** Image-conditioned Wan2.2: image_prompts go to the pipeline. */
class WanI2VModel : public WanModel {
 public:
  WanI2VModel(RunnerConfig config, ModelSpec spec)
      : WanModel(std::move(config), std::move(spec)) {}
  std::string adapterName() const override { return "wan_i2v"; }
  Json::Value buildCall(const DiTRequest& request) const override;
  bool requiresImageConditioning() const override { return true; }
};

class WanAniSoraModel : public WanI2VModel {
 public:
  using WanI2VModel::WanI2VModel;
  std::string adapterName() const override { return "wan_anisora"; }
  std::vector<std::pair<std::string, std::string>> envDefaults() const override;

 protected:
  int fixedSteps() const override { return 8; }
  double guidanceScale() const override { return 3.5; }
  double guidanceScale2() const override { return 3.5; }
  long largeBhTraceRegion() const override { return 200'000'000; }
};

class WanDistillModel : public WanI2VModel {
 public:
  using WanI2VModel::WanI2VModel;
  std::string adapterName() const override { return "wan_distill"; }
  std::vector<std::pair<std::string, std::string>> envDefaults() const override;

 protected:
  int fixedSteps() const override { return 4; }
  double guidanceScale() const override { return 1.0; }
  double guidanceScale2() const override { return 1.0; }
  bool usesNegativePrompt() const override { return false; }
  long largeBhTraceRegion() const override { return 200'000'000; }
};

/** LoRA adapters from LORA_HIGH_PATH / LORA_LOW_PATH fused into both experts.
 */
class WanLoraModel : public WanI2VModel {
 public:
  using WanI2VModel::WanI2VModel;
  std::string adapterName() const override { return "wan_lora"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
};

/** lightx2v Wan2.2-Lightning 4-step LoRA; downloaded unless LORA_*_PATH is set.
 */
class WanLightningModel : public WanLoraModel {
 public:
  using WanLoraModel::WanLoraModel;
  Json::Value adapterOptions(const HostEnv& host) const override;

 protected:
  int fixedSteps() const override { return 4; }
  double guidanceScale() const override { return 1.0; }
  double guidanceScale2() const override { return 1.0; }
  bool usesNegativePrompt() const override { return false; }
};

/** Prodia's distilled Wan: its own call signature, no step count and no
 *  negative prompt. */
class WanProdiaModel : public WanModel {
 public:
  using WanModel::WanModel;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv& host) const override;
};

class WanProdiaT2VModel : public WanProdiaModel {
 public:
  using WanProdiaModel::WanProdiaModel;
  std::string adapterName() const override { return "wan_prodia_t2v"; }
};

class WanProdiaI2VModel : public WanProdiaModel {
 public:
  using WanProdiaModel::WanProdiaModel;
  std::string adapterName() const override { return "wan_prodia_i2v"; }
  Json::Value buildCall(const DiTRequest& request) const override;
  bool requiresImageConditioning() const override { return true; }
};

// -- Mochi, LTX -------------------------------------------------------------

/** Mochi-1: 168 frames at 480x848 (the pipeline's own defaults). */
class MochiModel : public DiTModel {
 public:
  explicit MochiModel(RunnerConfig config);
  std::string adapterName() const override { return "mochi"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
};

constexpr int kLtxNumFrames = 153;
constexpr int kLtxHeight = 1088;
constexpr int kLtxWidth = 1920;
constexpr double kLtxFps = 25.0;

/** Nearest legal LTX frame count, (n - 1) % 8 == 0. */
int ltxSnapNumFrames(long n);

/** LTX-2.3 distilled text to audio-video. The pipeline writes the mp4 (h264 +
 *  aac) itself, and the shape is baked into its traces, so a request may only
 *  ask for the served shape. */
class LtxModel : public DiTModel {
 public:
  explicit LtxModel(RunnerConfig config);
  std::string adapterName() const override { return "ltx"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv& host) const override;
  void validate(const DiTRequest& request) const override;
};

// -- MiniMax-H3 -------------------------------------------------------------

constexpr int kMiniMaxSteps = 50;
constexpr int kMiniMaxFastH3Steps = 4;
constexpr int kMiniMaxLightX2VSteps = 5;

/** MiniMax-H3 text to video with sound (task t2va). Subclasses change the
 *  task, the step count, the pipeline class and the extra inputs. */
class MiniMaxModel : public DiTModel {
 public:
  MiniMaxModel(RunnerConfig config, ModelSpec spec);
  std::string adapterName() const override { return "minimax"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv& host) const override;
  bool runsWarmupRequest() const override { return false; }

 protected:
  virtual std::string task() const { return "t2va"; }
  /** nullopt: FSDP on Wormhole, off on Blackhole. */
  virtual std::optional<bool> ditFsdp() const { return std::nullopt; }
  /** nullopt: 65536 on Blackhole, 32768 on Wormhole. */
  virtual std::optional<int> l1SmallBytes() const { return std::nullopt; }
  virtual int numInferenceSteps() const { return kMiniMaxSteps; }
  /** Adds the model-specific create() options. */
  virtual void addPipelineOptions(Json::Value&) const {}
  /** Adds the model-specific call arguments (keyframes, references). */
  virtual void addCallInputs(Json::Value&, const DiTRequest&) const {}
  int stepsFor(const DiTRequest&) const override { return numInferenceSteps(); }
};

class MiniMaxFastH3Model : public MiniMaxModel {
 public:
  using MiniMaxModel::MiniMaxModel;

 protected:
  int numInferenceSteps() const override { return kMiniMaxFastH3Steps; }
  void addPipelineOptions(Json::Value& options) const override;
};

/** Optional first (frame_pos 0) and last (frame_pos -1) keyframes; a text-only
 *  request runs as t2va. */
class MiniMaxFL2VAModel : public MiniMaxModel {
 public:
  using MiniMaxModel::MiniMaxModel;
  void validate(const DiTRequest& request) const override;

 protected:
  std::string task() const override { return "fl2va"; }
  void addCallInputs(Json::Value& call, const DiTRequest& r) const override;
};

/** references.images / .videos / .audios (base64). Loads transformer_ref/,
 *  which needs FSDP to fit. */
class MiniMaxRef2VAModel : public MiniMaxModel {
 public:
  using MiniMaxModel::MiniMaxModel;
  void validate(const DiTRequest& request) const override;

 protected:
  std::string task() const override { return "ref2va"; }
  std::optional<bool> ditFsdp() const override { return true; }
  // The video VAE's taps=3 encoder clashes with a 65536 pool.
  std::optional<int> l1SmallBytes() const override { return 16384; }
  void addCallInputs(Json::Value& call, const DiTRequest& r) const override;
};

/** Binds a lightx2v 4-step Turbo adapter (MINIMAX_H3_LORA_PATH). Each adapter
 *  was distilled at its own (video, audio) shift; a wrong shift still runs,
 *  over the wrong sigma grid, so it is pinned per model. */
template <class Base>
class LightX2VModel : public Base {
 public:
  LightX2VModel(RunnerConfig config, ModelSpec spec, double videoShift,
                double audioShift)
      : Base(std::move(config), std::move(spec)),
        videoShift_(videoShift),
        audioShift_(audioShift) {}

 protected:
  int numInferenceSteps() const override { return kMiniMaxLightX2VSteps; }
  void addPipelineOptions(Json::Value& options) const override;

 private:
  double videoShift_;
  double audioShift_;
};

/** The shared part of LightX2VModel::addPipelineOptions. */
void addLightX2VOptions(Json::Value& options, const std::string& modelName,
                        double videoShift, double audioShift);

template <class Base>
void LightX2VModel<Base>::addPipelineOptions(Json::Value& options) const {
  options["pipeline"] = "turbo";
  addLightX2VOptions(options, this->spec().logName, videoShift_, audioShift_);
}

// -- Image models -----------------------------------------------------------

/** Text-to-image through the pipeline's run_single_prompt (SD3.5, FLUX.1 dev
 *  and schnell, Motif, Qwen-Image). generate() returns PNG bytes. */
class ImageModel : public DiTModel {
 public:
  ImageModel(RunnerConfig config, ModelSpec spec, DeviceParams device,
             std::string adapter);
  std::string adapterName() const override { return adapter_; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv&) const override { return device_; }

 protected:
  DeviceParams device_;
  std::string adapter_;
};

/** FLUX.1-Kontext-dev: instruction-based editing (reference image + prompt);
 *  with no image it generates from text. Validated untraced. */
class FluxKontextModel : public ImageModel {
 public:
  FluxKontextModel(RunnerConfig config, ModelSpec spec, DeviceParams device);
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
};

/** TRACE_REGION_SIZE if set, else the device's value, else the Python
 *  Settings default (34541598). */
long settingsTraceRegion(const std::string& device, const std::string& runner);

}  // namespace tt::dit
