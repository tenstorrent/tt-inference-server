#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_context_ttmetal.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llm_engine {

namespace {

struct DecodeEntry {
  uint32_t token_id;
  uint32_t user_id;
  uint32_t position_id;
};
constexpr uint32_t kWordsPerEntry = 3;
constexpr uint32_t kBytesPerEntry = kWordsPerEntry * sizeof(uint32_t);
static_assert(sizeof(DecodeEntry) == kBytesPerEntry, "DecodeEntry must be 12 bytes");

/** Page size must be PCIe-aligned (e.g. 64). One page = one DecodeEntry (first 12 bytes) + padding. Must match device_context_ttmetal and loopback kernel. */
constexpr uint32_t kPageSize = 64;
constexpr uint32_t kPageSizeWords = kPageSize / sizeof(uint32_t);
static_assert(kPageSize >= kBytesPerEntry, "page must fit one DecodeEntry");

}  // namespace

class ModelRunnerSocketSim : public IModelRunner {
 public:
  explicit ModelRunnerSocketSim(const Config& config);
  ~ModelRunnerSocketSim() override { exit(); }

  std::vector<int64_t> run(const std::vector<Sequence*>& seqs,
                           bool is_prefill) override;
  void exit() override;

 private:
  Config config_;
  int eos_;
  void* device_ctx_ = nullptr;
};

ModelRunnerSocketSim::ModelRunnerSocketSim(const Config& config)
    : config_(config), eos_(config.eos) {
  device_ctx_ = create_ttmetal_decode_context_and_config(&config_);
  if (!device_ctx_ || !config_.h2d_socket || !config_.d2h_socket) {
    throw std::runtime_error("model_runner: tt-metal device and H2D/D2H sockets required");
  }
  LLM_ENGINE_LOG("model_runner") << "H2D/D2H sockets ready (per-request sender/receiver threads)"
                                 << std::endl;
}

std::vector<int64_t> ModelRunnerSocketSim::run(const std::vector<Sequence*>& seqs,
                                               bool is_prefill) {
  if (is_prefill) {
    LLM_ENGINE_LOG("model_runner") << "prefill batch_size=" << seqs.size()
                                   << std::endl;
    return std::vector<int64_t>(seqs.size(), seqs[0]->last_token+1);
  }

  if (seqs.empty()) {
    return {};
  }

  Sequence* s = seqs[0];
  DecodeEntry e;
  e.token_id = static_cast<uint32_t>(s->last_token & 0xFFFFFFFFu);
  e.user_id = static_cast<uint32_t>(s->seq_id & 0xFFFFFFFFu);
  e.position_id = static_cast<uint32_t>(s->size() & 0xFFFFFFFFu);

  uint32_t send_buf[kPageSizeWords] = {};
  send_buf[0] = e.token_id;
  send_buf[1] = e.user_id;
  send_buf[2] = e.position_id;

  uint32_t recv_buf[kPageSizeWords] = {};
  int64_t result_token = 0;

  auto* h2d = static_cast<tt::tt_metal::distributed::H2DSocket*>(config_.h2d_socket);
  auto* d2h = static_cast<tt::tt_metal::distributed::D2HSocket*>(config_.d2h_socket);

  std::thread sender([h2d, &send_buf]() {
    h2d->write(send_buf, 1);
    h2d->barrier();
  });
  std::thread receiver([d2h, &recv_buf, &result_token]() {
    d2h->read(recv_buf, 1);
    d2h->barrier();
    result_token = static_cast<int64_t>(recv_buf[0])+1;
  });

  sender.join();
  receiver.join();

  LLM_ENGINE_LOG("model_runner") << "decode seq_id=" << s->seq_id
                                << " last_token=" << s->last_token
                                << " -> token=" << result_token << std::endl;

  if (seqs.size() == 1) {
    return {result_token};
  }
  std::vector<int64_t> out(seqs.size(), static_cast<int64_t>(eos_));
  out[0] = result_token;
  return out;
}

void ModelRunnerSocketSim::exit() {
  if (device_ctx_) {
    LLM_ENGINE_LOG("model_runner") << "exit (destroy device context)" << std::endl;
    destroy_ttmetal_decode_context(device_ctx_);
    device_ctx_ = nullptr;
  }
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config) {
  return std::make_unique<ModelRunnerSocketSim>(config);
}

}  // namespace llm_engine
