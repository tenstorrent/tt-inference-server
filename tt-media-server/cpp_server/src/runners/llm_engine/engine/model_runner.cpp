#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/host_interface.hpp"
#include <cstring>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {

constexpr uint32_t kFifoSize = 64*64*2;
constexpr uint32_t kNumIterationsStreaming = 10000000;
const char* kLoopbackKernelPath =
    "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp";

void DecodeQueue::push(const DecodeResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(result);
}

std::vector<DecodeResult> DecodeQueue::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<DecodeResult> out;
  out.swap(pending_);
  return out;
}

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback)
    : config_(config),
      dummy_token_((config.eos == 0) ? 1 : 0),
      decode_callback_(std::move(callback)) {

        try {
            mesh_device_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        } catch (const std::exception& e) {
            throw;
        } catch (...) {
            throw;
        }

        tt::tt_metal::distributed::MeshCoordinate device_coord{0, 0};
        tt::tt_metal::CoreCoord core_coord{0, 0};
        tt::tt_metal::distributed::MeshCoreCoord socket_core{device_coord, core_coord};

        try {
            h2d_socket_ = std::make_unique<tt::tt_metal::distributed::H2DSocket>(
                mesh_device_,
                socket_core,
                tt::tt_metal::BufferType::L1,
                kFifoSize,
                tt::tt_metal::distributed::H2DMode::HOST_PUSH);
            h2d_socket_->set_page_size(Sequence::page_size());
        } catch (const std::exception& e) {
            throw;
        } catch (...) {
            throw;
        }

        try {
            d2h_socket_ = std::make_unique<tt::tt_metal::distributed::D2HSocket>(
                mesh_device_, socket_core, kFifoSize);
            d2h_socket_->set_page_size(Sequence::page_size());
        } catch (const std::exception& e) {
            throw;
        } catch (...) {
            throw;
        }

        host_io_ = std::make_unique<HostInterface>();
        host_io_->run(h2d_socket_.get(), d2h_socket_.get(), mesh_device_.get(), kNumIterationsStreaming);

        reader_thread_ = std::thread([this] { reader_loop(); });
      }

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::reader_loop() {
  try {
    std::vector<char> output(Sequence::page_size(), 0);
    while (!stop_.load(std::memory_order_relaxed)) {
      d2h_socket_->read(output.data(), 1);
      SequenceID seq_id = SequenceID::deserialize(output.data(), SequenceID::kSerializedSize);
      int64_t last_token;
      std::memcpy(&last_token, output.data() + SequenceID::kSerializedSize, sizeof(last_token));
      decode_callback_(DecodeResult{seq_id, last_token});
    }
  } catch (const std::exception& e) {
  } catch (...) {
  } 
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->seq_id, seq->last_token + 1});
    }
  } else {
    h2d_socket_->write(seqs[0]->to_h2d_input().data(), 1);
  }
}

void ModelRunnerStub::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  host_io_->terminate();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  return std::make_unique<ModelRunnerStub>(config, std::move(callback));
}

}  // namespace llm_engine
