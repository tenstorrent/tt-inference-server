// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/image_service.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct RunnerState {
  std::atomic<size_t> warmups{0};
  std::atomic<size_t> calls{0};
  std::mutex runMutex;
};

class FakeImageRunner : public tt::services::ImageService::Runner {
 public:
  FakeImageRunner(std::string runnerName,
                  std::shared_ptr<RunnerState> runnerState)
      : name(std::move(runnerName)), state(std::move(runnerState)) {}

  bool warmup() override {
    state->warmups.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  std::vector<std::string> run(
      const tt::domain::ImageGenerateRequest& request) override {
    std::lock_guard<std::mutex> lock(state->runMutex);
    state->calls.fetch_add(1, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return {name + ":" + std::to_string(request.task_id)};
  }

  const char* runnerType() const override { return name.c_str(); }

 private:
  std::string name;
  std::shared_ptr<RunnerState> state;
};

tt::domain::ImageGenerateRequest makeRequest(uint32_t taskId) {
  tt::domain::ImageGenerateRequest request(taskId);
  request.prompt = "test prompt";
  return request;
}

void waitUntilReady(tt::services::ImageService& service) {
  for (int i = 0; i < 100; ++i) {
    if (service.isModelReady()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL() << "image service did not become ready";
}

}  // namespace

TEST(ImageServiceTest, WarmsAllConfiguredRunners) {
  auto state0 = std::make_shared<RunnerState>();
  auto state1 = std::make_shared<RunnerState>();

  tt::services::ImageService::RunnerList runners;
  runners.push_back(std::make_unique<FakeImageRunner>("runner0", state0));
  runners.push_back(std::make_unique<FakeImageRunner>("runner1", state1));

  tt::services::ImageService service(tt::config::ImageConfig{},
                                     std::move(runners));
  service.start();
  waitUntilReady(service);

  EXPECT_EQ(state0->warmups.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(state1->warmups.load(std::memory_order_relaxed), 1u);
}

TEST(ImageServiceTest, DispatchesConcurrentRequestsAcrossRunners) {
  auto state0 = std::make_shared<RunnerState>();
  auto state1 = std::make_shared<RunnerState>();

  tt::services::ImageService::RunnerList runners;
  runners.push_back(std::make_unique<FakeImageRunner>("runner0", state0));
  runners.push_back(std::make_unique<FakeImageRunner>("runner1", state1));

  tt::services::ImageService service(tt::config::ImageConfig{},
                                     std::move(runners));
  service.start();
  waitUntilReady(service);

  auto first = std::async(std::launch::async, [&] {
    return service.submitRequest(makeRequest(1));
  });
  auto second = std::async(std::launch::async, [&] {
    return service.submitRequest(makeRequest(2));
  });

  EXPECT_TRUE(first.get().error.empty());
  EXPECT_TRUE(second.get().error.empty());
  EXPECT_EQ(state0->calls.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(state1->calls.load(std::memory_order_relaxed), 1u);
}
