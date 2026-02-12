#include "llm_engine/engine/spoofed_blitz_decode.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_context_ttmetal.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace llm_engine {

namespace {

constexpr uint32_t kSeqIdMaxBytes = 36;

struct WireFormat {
  uint32_t token_id;
  uint32_t position;
  char seq_id[kSeqIdMaxBytes];
};
static_assert(sizeof(WireFormat) <= 64, "wire format must fit in one page");

constexpr uint32_t kPageSize = 64;

constexpr uint32_t kDataSizeOnePage = 64;
constexpr uint32_t kFifoSize = 2048;
constexpr uint32_t kNumIterationsStreaming = 65536;
constexpr const char* kLoopbackKernelPath =
    "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp";

class TTMetalDecodeContext {
 public:
  using MeshDevice = tt::tt_metal::distributed::MeshDevice;
  using H2DSocket = tt::tt_metal::distributed::H2DSocket;
  using D2HSocket = tt::tt_metal::distributed::D2HSocket;
  using MeshCoreCoord = tt::tt_metal::distributed::MeshCoreCoord;
  using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
  using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;
  using MeshShape = tt::tt_metal::distributed::MeshShape;
  using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
  using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;

  static std::unique_ptr<TTMetalDecodeContext> create(Config* config) {
#if defined(TT_METAL_RUNTIME_ROOT_DEFAULT)
    if (std::getenv("TT_METAL_RUNTIME_ROOT") == nullptr) {
      setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_RUNTIME_ROOT_DEFAULT, 1);
    }
#endif

    try {
      using namespace tt::tt_metal;
      using namespace tt::tt_metal::distributed;

      if (GetNumAvailableDevices() == 0) {
        LLM_ENGINE_LOG("device_context_ttmetal") << "No devices; skipping real sockets."
                                                 << std::endl;
        return nullptr;
      }

      auto ctx = std::unique_ptr<TTMetalDecodeContext>(new TTMetalDecodeContext());
      ctx->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
      ctx->socket_core_ = MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0));

      ctx->h2d_socket_ = std::make_unique<H2DSocket>(
          ctx->mesh_device_, ctx->socket_core_, BufferType::L1, kFifoSize, H2DMode::HOST_PUSH);
      ctx->h2d_socket_->set_page_size(kPageSize);

      ctx->d2h_socket_ = std::make_unique<D2HSocket>(ctx->mesh_device_, ctx->socket_core_, kFifoSize);
      ctx->d2h_socket_->set_page_size(kPageSize);

      auto loopback_program = CreateProgram();
      CreateKernel(
          loopback_program,
          kLoopbackKernelPath,
          ctx->socket_core_.core_coord,
          DataMovementConfig{
              .processor = DataMovementProcessor::RISCV_0,
              .noc = NOC::RISCV_0_default,
              .compile_args = {
                  static_cast<uint32_t>(ctx->h2d_socket_->get_config_buffer_address()),
                  static_cast<uint32_t>(ctx->d2h_socket_->get_config_buffer_address()),
                  kPageSize,
                  kDataSizeOnePage,
                  kNumIterationsStreaming,
                  static_cast<uint32_t>(false),
              }});
      MeshWorkload mesh_workload;
      mesh_workload.add_program(
          MeshCoordinateRange(ctx->socket_core_.device_coord), std::move(loopback_program));
      EnqueueMeshWorkload(ctx->mesh_device_->mesh_command_queue(), mesh_workload, false);

      if (config) {
        config->mesh_device = ctx->mesh_device_.get();
        config->h2d_socket = ctx->h2d_socket_.get();
        config->d2h_socket = ctx->d2h_socket_.get();
      }

      LLM_ENGINE_LOG("device_context_ttmetal")
          << "Opened 1x1 mesh, H2D/D2H sockets (page_size=" << kPageSize
          << "), loopback kernel enqueued once (" << kNumIterationsStreaming << " iterations)."
          << std::endl;
      return ctx;
    } catch (const std::exception& e) {
      LLM_ENGINE_LOG("device_context_ttmetal") << "tt-metal device open failed: " << e.what()
                                              << "." << std::endl;
      return nullptr;
    }
  }

  ~TTMetalDecodeContext() {
    if (!mesh_device_) return;
    Finish(mesh_device_->mesh_command_queue());
    mesh_device_->close();
    LLM_ENGINE_LOG("device_context_ttmetal") << "Device closed." << std::endl;
  }

  TTMetalDecodeContext(const TTMetalDecodeContext&) = delete;
  TTMetalDecodeContext& operator=(const TTMetalDecodeContext&) = delete;

  H2DSocket* h2d_socket() const { return h2d_socket_.get(); }
  D2HSocket* d2h_socket() const { return d2h_socket_.get(); }

 private:
  TTMetalDecodeContext() = default;

  std::shared_ptr<MeshDevice> mesh_device_;
  std::unique_ptr<H2DSocket> h2d_socket_;
  std::unique_ptr<D2HSocket> d2h_socket_;
  MeshCoreCoord socket_core_;
};

}  // namespace

void* create_ttmetal_decode_context_and_config(Config* config) {
  auto ctx = TTMetalDecodeContext::create(config);
  return ctx.release();
}

void destroy_ttmetal_decode_context(void* ctx) {
  if (!ctx) return;
  delete static_cast<TTMetalDecodeContext*>(ctx);
}

SpoofedBlitzDecode::SpoofedBlitzDecode(const Config& config, DecodeCallback decode_callback)
    : config_{config}, decode_callback_{std::move(decode_callback)} {}

SpoofedBlitzDecode::~SpoofedBlitzDecode() { exit(); }

void SpoofedBlitzDecode::run() {
  device_ctx_ = create_ttmetal_decode_context_and_config(&config_);
  if (!device_ctx_ || !config_.h2d_socket || !config_.d2h_socket) {
    throw std::runtime_error("SpoofedBlitzDecode: tt-metal device and H2D/D2H sockets required");
  }
  receiver_thread_ = std::thread{&SpoofedBlitzDecode::receiver_loop, this};
  LLM_ENGINE_LOG("spoofed_blitz_decode")
      << "H2D/D2H sockets ready (long-running receiver thread)" << std::endl;
}

void SpoofedBlitzDecode::receiver_loop() {
  auto* d2h = static_cast<tt::tt_metal::distributed::D2HSocket*>(config_.d2h_socket);
  while (!receiver_shutdown_.load()) {
    WireFormat recv{};
    d2h->read(reinterpret_cast<uint32_t*>(&recv), 1);
    d2h->barrier();

    int64_t token_id = static_cast<int64_t>(recv.token_id);
    SequenceID seq_id;
    const char* end =
        static_cast<const char*>(std::memchr(recv.seq_id, '\0', sizeof(recv.seq_id)));
    seq_id.id.assign(recv.seq_id, end ? static_cast<size_t>(end - recv.seq_id) : sizeof(recv.seq_id));
    LLM_ENGINE_LOG("spoofed_blitz_decode")
        << "D2H recv token_id=" << token_id << " seq_id=" << seq_id << std::endl;
    if (decode_callback_) decode_callback_({seq_id, token_id});
  }
}

void SpoofedBlitzDecode::decode(const std::vector<Sequence*>& seqs) {
  if (seqs.empty()) return;

  auto* h2d = static_cast<tt::tt_metal::distributed::H2DSocket*>(config_.h2d_socket);
  for (Sequence* s : seqs) {
    WireFormat wf{};
    wf.token_id = static_cast<uint32_t>((s->last_token + 1) & 0xFFFFFFFFu);
    wf.position = static_cast<uint32_t>(s->size() & 0xFFFFFFFFu);
    std::memset(wf.seq_id, 0, sizeof(wf.seq_id));
    std::strncpy(wf.seq_id, s->seq_id.id.c_str(), sizeof(wf.seq_id) - 1);

    LLM_ENGINE_LOG("spoofed_blitz_decode")
        << "H2D send token_id=" << wf.token_id << " position=" << wf.position
        << " seq_id=" << s->seq_id << std::endl;
    h2d->write(reinterpret_cast<uint32_t*>(&wf), 1);
  }
}

void SpoofedBlitzDecode::exit() {
  if (receiver_thread_.joinable()) {
    receiver_shutdown_.store(true);
    receiver_thread_.join();
  }
  if (device_ctx_) {
    LLM_ENGINE_LOG("spoofed_blitz_decode") << "exit (destroy device context)" << std::endl;
    destroy_ttmetal_decode_context(device_ctx_);
    device_ctx_ = nullptr;
  }
}

}  // namespace llm_engine
