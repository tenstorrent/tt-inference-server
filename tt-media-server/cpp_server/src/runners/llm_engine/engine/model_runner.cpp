#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_context_ttmetal.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <mutex>
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

  void run(const std::vector<Sequence*>& seqs,
           bool is_prefill,
           OnTokensCallback on_tokens) override;
  void exit() override;

 private:
  void sender_loop();

  Config config_;
  int eos_;
  void* device_ctx_ = nullptr;

  std::thread sender_thread_;
  std::mutex sender_mutex_;
  std::condition_variable sender_cv_request_;
  std::condition_variable sender_cv_done_;
  bool sender_work_ready_ = false;
  std::atomic<bool> sender_shutdown_{false};
  bool sender_done_ = false;
  uint32_t sender_buf_[kPageSizeWords] = {};
};

ModelRunnerSocketSim::ModelRunnerSocketSim(const Config& config)
    : config_(config), eos_(config.eos) {
  device_ctx_ = create_ttmetal_decode_context_and_config(&config_);
  if (!device_ctx_ || !config_.h2d_socket || !config_.d2h_socket) {
    throw std::runtime_error("model_runner: tt-metal device and H2D/D2H sockets required");
  }
  sender_thread_ = std::thread{&ModelRunnerSocketSim::sender_loop, this};
  LLM_ENGINE_LOG("model_runner") << "H2D/D2H sockets ready (long-running sender thread)"
                                 << std::endl;
}

void ModelRunnerSocketSim::sender_loop() {
  auto* h2d = static_cast<tt::tt_metal::distributed::H2DSocket*>(config_.h2d_socket);
  while (true) {
    uint32_t buf[kPageSizeWords];
    {
      std::unique_lock lock{sender_mutex_};
      sender_cv_request_.wait(lock, [this] {
        return sender_work_ready_ || sender_shutdown_.load();
      });
      if (sender_shutdown_.load()) break;
      std::copy(std::begin(sender_buf_), std::end(sender_buf_), std::begin(buf));
      sender_work_ready_ = false;
    }
    h2d->write(buf, 1);
    // h2d->barrier();
    {
      std::lock_guard lock{sender_mutex_};
      sender_done_ = true;
      sender_cv_done_.notify_one();
    }
  }
}

void ModelRunnerSocketSim::run(const std::vector<Sequence*>& seqs,
                               bool is_prefill,
                               OnTokensCallback on_tokens) {
  if (is_prefill) {
    LLM_ENGINE_LOG("model_runner") << "prefill batch_size=" << seqs.size()
                                   << std::endl;
    std::vector<int64_t> token_ids(seqs.size(), seqs[0]->last_token + 1);
    on_tokens(seqs, std::move(token_ids));
    return;
  }

  if (seqs.empty()) {
    return;
  }

  Sequence* s = seqs[0];
  DecodeEntry e;
  e.token_id = static_cast<uint32_t>(s->last_token & 0xFFFFFFFFu);
  e.user_id = static_cast<uint32_t>(s->seq_id & 0xFFFFFFFFu);
  e.position_id = static_cast<uint32_t>(s->size() & 0xFFFFFFFFu);

  {
    std::unique_lock lock{sender_mutex_};
    sender_buf_[0] = e.token_id;
    sender_buf_[1] = e.user_id;
    sender_buf_[2] = e.position_id;
    sender_done_ = false;
    sender_work_ready_ = true;
    sender_cv_request_.notify_one();
    sender_cv_done_.wait(lock, [this] { return sender_done_; });
  }

  auto* d2h = static_cast<tt::tt_metal::distributed::D2HSocket*>(config_.d2h_socket);
  std::thread receiver([this, d2h, seqs, on_tokens = std::move(on_tokens)]() {
    uint32_t recv_buf[kPageSizeWords] = {};
    d2h->read(recv_buf, 1);
    d2h->barrier();
    int64_t result_token = static_cast<int64_t>(recv_buf[0]) + 1;
    LLM_ENGINE_LOG("model_runner") << "decode seq_id=" << seqs[0]->seq_id
                                  << " last_token=" << seqs[0]->last_token
                                  << " -> token=" << result_token << std::endl;
    std::vector<int64_t> token_ids;
    if (seqs.size() == 1) {
      token_ids = {result_token};
    } else {
      token_ids.assign(seqs.size(), static_cast<int64_t>(eos_));
      token_ids[0] = result_token;
    }
    on_tokens(seqs, std::move(token_ids));
  });
  receiver.detach();
}

void ModelRunnerSocketSim::exit() {
  if (sender_thread_.joinable()) {
    sender_shutdown_.store(true);
    sender_cv_request_.notify_one();
    sender_thread_.join();
  }
  if (device_ctx_) {
    LLM_ENGINE_LOG("model_runner") << "exit (destroy device context)" << std::endl;
    destroy_ttmetal_decode_context(device_ctx_);
    device_ctx_ = nullptr;
  }
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config) {
  return std::make_unique<ModelRunnerSocketSim>(config);
}

ModelRunnerStub::ModelRunnerStub(const Config& config)
    : config_(config), eos_(config.eos) {}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill,
                          OnTokensCallback on_tokens) {
  if (seqs.empty()) return;
  std::vector<int64_t> token_ids(seqs.size(), seqs[0]->last_token + 1);
  on_tokens(seqs, std::move(token_ids));
}

void ModelRunnerStub::exit() {}

}  // namespace llm_engine
