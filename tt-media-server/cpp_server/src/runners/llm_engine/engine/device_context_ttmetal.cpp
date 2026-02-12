// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/engine/device_context_ttmetal.hpp"
#include "llm_engine/engine/debug.hpp"

#include <cstdlib>
#include <exception>
#include <memory>

static void ensure_tt_metal_runtime_root() {
  if (std::getenv("TT_METAL_RUNTIME_ROOT") != nullptr) return;
  const char* home = std::getenv("TT_METAL_HOME");
  if (home != nullptr) {
    setenv("TT_METAL_RUNTIME_ROOT", home, 1);
  }
#if defined(TT_METAL_RUNTIME_ROOT_DEFAULT)
  else {
    setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_RUNTIME_ROOT_DEFAULT, 1);
  }
#endif
}

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

namespace llm_engine {

namespace {

struct TTMetalMeshContext {
  using MeshDevice = tt::tt_metal::distributed::MeshDevice;
  using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;
  using MeshShape = tt::tt_metal::distributed::MeshShape;

  std::shared_ptr<MeshDevice> mesh_device;
};

}  // namespace

void* create_ttmetal_decode_context_and_config(Config* config) {
  ensure_tt_metal_runtime_root();

  try {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    if (GetNumAvailableDevices() == 0) {
      LLM_ENGINE_LOG("device_context_ttmetal") << "No devices available." << std::endl;
      return nullptr;
    }

    auto ctx = new TTMetalMeshContext{};
    ctx->mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    config->mesh_device = ctx->mesh_device.get();

    LLM_ENGINE_LOG("device_context_ttmetal") << "Opened mesh device (1,1)." << std::endl;
    return ctx;
  } catch (const std::exception& e) {
    LLM_ENGINE_LOG("device_context_ttmetal") << "tt-metal device open failed: " << e.what()
                                            << "." << std::endl;
    return nullptr;
  }
}

void destroy_ttmetal_decode_context(void* ctx) {
  if (!ctx) return;
  auto* c = static_cast<TTMetalMeshContext*>(ctx);
  Finish(c->mesh_device->mesh_command_queue());
  c->mesh_device->close();
  delete c;
  LLM_ENGINE_LOG("device_context_ttmetal") << "Device closed." << std::endl;
}

}  // namespace llm_engine
