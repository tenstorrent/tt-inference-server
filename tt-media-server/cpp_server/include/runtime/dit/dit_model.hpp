// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "runtime/dit/dit_types.hpp"

namespace tt::dit {

/** The fixed facts of one model, written once per model class. */
struct ModelSpec {
  std::string logName;
  OutputKind kind = OutputKind::Video;
  /** DEVICE -> mesh, the device types the model is deployed on. */
  std::map<std::string, MeshShape> meshes;
  /** Hugging Face repo id used when no weights path is configured. */
  std::string checkpoint;
  /** false: always load `checkpoint` (MODEL_WEIGHTS_DIR / MODEL_WEIGHTS_PATH
   *  are ignored, as in the Python server for SD3.5, Motif and Mochi). */
  bool checkpointFromConfig = true;
  /** The `traced` argument of the pipeline; TT_VIDEO_TRACED overrides it. */
  bool defaultTraced = true;
  /** DEVICE values on which `traced` defaults to false although the model
   *  traces elsewhere: tracing gives noise frames there (Wan2.2 on a T3K).
   *  TT_VIDEO_TRACED=1 still turns it on. */
  std::set<std::string> untracedDevices;
  int defaultSteps = 20;
  double defaultFps = 16.0;
  std::string warmupPrompt = "Sunrise on a beach";
  int warmupSteps = 2;
};

/**
 * What a model is, without touching the device or Python: its mesh, how the
 * mesh is opened, which requests it accepts and the arguments of its pipeline
 * call. DiTRunner (dit_runner.hpp) runs a model; the Python adapter of the
 * same name (python/cpp_dit_adapters) holds the only code that calls tt_dit.
 *
 * A new model is one subclass: pass a ModelSpec to the constructor, implement
 * adapterName(), adapterOptions() and buildCall(), override deviceParams() or
 * validate() where it differs, and add it to the table in dit_registry.cpp.
 * Models of one family share a base class (WanModel, MiniMaxModel, ...).
 */
class DiTModel {
 public:
  DiTModel(RunnerConfig config, ModelSpec spec);
  virtual ~DiTModel() = default;

  const RunnerConfig& config() const { return config_; }
  const ModelSpec& spec() const { return spec_; }
  const std::string& runnerName() const { return config_.modelRunner; }
  OutputKind outputKind() const { return spec_.kind; }
  MeshShape mesh() const { return mesh_; }
  const std::string& checkpoint() const { return checkpoint_; }
  bool traced() const { return traced_; }

  /** Name of the Python adapter (cpp_dit_adapters.ADAPTERS key). */
  virtual std::string adapterName() const = 0;
  /** What the adapter's create() needs: checkpoint, shape, flags. */
  virtual Json::Value adapterOptions(const HostEnv& host) const = 0;
  /** Keyword arguments of one pipeline call; see the adapter for the keys it
   *  maps to Python objects (images, references). */
  virtual Json::Value buildCall(const DiTRequest& request) const = 0;
  virtual DeviceParams deviceParams(const HostEnv&) const { return {}; }
  /** Refuse a request this model cannot serve (throws RequestError). */
  virtual void validate(const DiTRequest& request) const;
  /** Environment variables to set (without overriding) before create(). */
  virtual std::vector<std::pair<std::string, std::string>> envDefaults() const {
    return {};
  }
  virtual DiTRequest warmupRequest() const;
  /** false: the pipeline warms itself up when it is built. */
  virtual bool runsWarmupRequest() const { return true; }
  virtual bool requiresImageConditioning() const { return false; }

  int steps(const DiTRequest& request) const;
  static int seed(const DiTRequest& request);
  double fps(const DiTRequest& request) const;

 protected:
  RunnerConfig config_;
  ModelSpec spec_;
  MeshShape mesh_;
  std::string checkpoint_;
  bool traced_;
  /** The step count the pipeline runs; distilled models ignore the request. */
  virtual int stepsFor(const DiTRequest& request) const;
};

using DiTModelFactory =
    std::unique_ptr<DiTModel> (*)(const RunnerConfig& config);

/** MODEL_RUNNER names with a model, sorted. */
std::vector<std::string> supportedRunners();

/** Throws std::invalid_argument for an unknown MODEL_RUNNER or a DEVICE the
 *  model is not deployed on. */
std::unique_ptr<DiTModel> createModel(const RunnerConfig& config);

/** The mesh for a device type: an env override (SD_3_5_FAST, SD_3_5_BASE, TP2,
 *  SP_MESH_4X32) wins, as in the Python server. */
MeshShape resolveMesh(const std::map<std::string, MeshShape>& meshes,
                      const std::string& runner, const std::string& device);

/** MODEL_WEIGHTS_DIR if it is a non-empty directory, else the configured
 *  path, else the model's Hugging Face repo id. */
std::string resolveCheckpoint(const std::string& modelWeightsPath,
                              const std::string& defaultCheckpoint);

}  // namespace tt::dit
