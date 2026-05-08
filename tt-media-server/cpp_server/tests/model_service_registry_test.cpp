// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "api/route_registry.hpp"
#include "config/types.hpp"
#include "runners/media_runner.hpp"
#include "services/base_service.hpp"
#include "services/service_container.hpp"
#include "services/service_registry.hpp"
#include "utils/runner_registry.hpp"

namespace {

using tt::api::RouteRegistry;
using tt::config::ModelRunnerType;
using tt::config::ModelService;
using tt::services::IService;
using tt::services::ServiceContainer;
using tt::services::ServiceRegistry;
using tt::utils::RunnerRegistry;

class FakeService : public IService {
 public:
  explicit FakeService(std::string tag) : name(std::move(tag)) {}
  void start() override {}
  void stop() override {}
  bool isModelReady() const override { return true; }
  tt::services::SystemStatus getSystemStatus() const override {
    return tt::services::SystemStatus{};
  }
  const std::string& tag() const { return name; }

 private:
  std::string name;
};

}  // namespace

// ---------------------------------------------------------------------------
// ServiceRegistry
// ---------------------------------------------------------------------------

TEST(ServiceRegistryTest, RegistersAndCreatesService) {
  ServiceRegistry::instance().clear();
  ServiceRegistry::instance().registerService(
      ModelService::EMBEDDING, []() -> std::shared_ptr<IService> {
        return std::make_shared<FakeService>("embed");
      });

  ASSERT_TRUE(ServiceRegistry::instance().has(ModelService::EMBEDDING));
  EXPECT_FALSE(ServiceRegistry::instance().has(ModelService::LLM));

  auto created = ServiceRegistry::instance().create(ModelService::EMBEDDING);
  ASSERT_NE(created, nullptr);
  auto fake = std::dynamic_pointer_cast<FakeService>(created);
  ASSERT_NE(fake, nullptr);
  EXPECT_EQ(fake->tag(), "embed");
}

TEST(ServiceRegistryTest, OverridesExistingFactory) {
  ServiceRegistry::instance().clear();
  ServiceRegistry::instance().registerService(
      ModelService::LLM, []() -> std::shared_ptr<IService> {
        return std::make_shared<FakeService>("v1");
      });
  ServiceRegistry::instance().registerService(
      ModelService::LLM, []() -> std::shared_ptr<IService> {
        return std::make_shared<FakeService>("v2");
      });

  auto fake = std::dynamic_pointer_cast<FakeService>(
      ServiceRegistry::instance().create(ModelService::LLM));
  ASSERT_NE(fake, nullptr);
  EXPECT_EQ(fake->tag(), "v2");
}

TEST(ServiceRegistryTest, MissingFactoryReturnsNullptr) {
  ServiceRegistry::instance().clear();
  EXPECT_EQ(ServiceRegistry::instance().create(ModelService::LLM), nullptr);
}

// ---------------------------------------------------------------------------
// RunnerRegistry
// ---------------------------------------------------------------------------

namespace {

class FakeIpcRunner : public tt::runners::IRunner {
 public:
  explicit FakeIpcRunner(std::string tag) : name(std::move(tag)) {}
  void stop() override {}
  const char* runnerType() const override { return name.c_str(); }

 private:
  void run() override {}
  std::string name;
};

}  // namespace

TEST(RunnerRegistryTest, IpcExactMatchPreferredOverFallback) {
  RunnerRegistry::instance().clear();
  RunnerRegistry::instance().registerIpcRunner(
      ModelService::LLM, ModelRunnerType::MOCK,
      [](const tt::config::RunnerConfig&, tt::ipc::IResultQueue*,
         tt::ipc::ITaskQueue*,
         tt::ipc::ICancelQueue*) -> std::unique_ptr<tt::runners::IRunner> {
        return std::make_unique<FakeIpcRunner>("mock");
      });
  RunnerRegistry::instance().registerIpcRunner(
      ModelService::LLM, ModelRunnerType::LLAMA,
      [](const tt::config::RunnerConfig&, tt::ipc::IResultQueue*,
         tt::ipc::ITaskQueue*,
         tt::ipc::ICancelQueue*) -> std::unique_ptr<tt::runners::IRunner> {
        return std::make_unique<FakeIpcRunner>("llama");
      });

  tt::config::RunnerConfig cfg = tt::config::LLMConfig{};
  auto llama = RunnerRegistry::instance().createIpc(ModelService::LLM,
                                                    ModelRunnerType::LLAMA, cfg,
                                                    nullptr, nullptr, nullptr);
  ASSERT_NE(llama, nullptr);
  EXPECT_STREQ(llama->runnerType(), "llama");
}

TEST(RunnerRegistryTest, IpcFallsBackToMockWhenTypeNotRegistered) {
  RunnerRegistry::instance().clear();
  RunnerRegistry::instance().registerIpcRunner(
      ModelService::LLM, ModelRunnerType::MOCK,
      [](const tt::config::RunnerConfig&, tt::ipc::IResultQueue*,
         tt::ipc::ITaskQueue*,
         tt::ipc::ICancelQueue*) -> std::unique_ptr<tt::runners::IRunner> {
        return std::make_unique<FakeIpcRunner>("mock");
      });

  tt::config::RunnerConfig cfg = tt::config::LLMConfig{};
  auto runner = RunnerRegistry::instance().createIpc(
      ModelService::LLM, ModelRunnerType::PIPELINE_MANAGER, cfg, nullptr,
      nullptr, nullptr);
  ASSERT_NE(runner, nullptr);
  EXPECT_STREQ(runner->runnerType(), "mock");
}

TEST(RunnerRegistryTest, IpcNoMatchReturnsNullptr) {
  RunnerRegistry::instance().clear();
  tt::config::RunnerConfig cfg = tt::config::LLMConfig{};
  EXPECT_EQ(RunnerRegistry::instance().createIpc(ModelService::LLM,
                                                 ModelRunnerType::MOCK, cfg,
                                                 nullptr, nullptr, nullptr),
            nullptr);
}

namespace {

class FakeMediaRunner : public tt::runners::IMediaRunner<int, std::string> {
 public:
  explicit FakeMediaRunner(std::string tag) : tag_(std::move(tag)) {}
  std::string run(const int& request) override {
    return tag_ + ":" + std::to_string(request);
  }
  const char* runnerType() const override { return tag_.c_str(); }

 private:
  std::string tag_;
};

}  // namespace

TEST(RunnerRegistryTest, MediaExactMatchInstantiatesTypedRunner) {
  RunnerRegistry::instance().clear();
  RunnerRegistry::instance().registerMediaRunner(
      ModelService::IMAGE, ModelRunnerType::TT_SDXL_GENERATE,
      [](const tt::config::RunnerConfig&)
          -> std::unique_ptr<tt::runners::IRunnerBase> {
        return std::make_unique<FakeMediaRunner>("sdxl-generate");
      });

  tt::config::RunnerConfig cfg = tt::config::ImageConfig{};
  auto runner = RunnerRegistry::instance().createMedia<FakeMediaRunner>(
      ModelService::IMAGE, ModelRunnerType::TT_SDXL_GENERATE, cfg);
  ASSERT_NE(runner, nullptr);
  EXPECT_EQ(runner->run(7), "sdxl-generate:7");
}

TEST(RunnerRegistryTest, MediaWrongRunnerShapeThrows) {
  RunnerRegistry::instance().clear();
  RunnerRegistry::instance().registerMediaRunner(
      ModelService::IMAGE, ModelRunnerType::TT_SDXL_GENERATE,
      [](const tt::config::RunnerConfig&)
          -> std::unique_ptr<tt::runners::IRunnerBase> {
        return std::make_unique<FakeMediaRunner>("sdxl-generate");
      });

  tt::config::RunnerConfig cfg = tt::config::ImageConfig{};
  using OtherShape = tt::runners::IMediaRunner<std::string, int>;
  EXPECT_THROW(RunnerRegistry::instance().createMedia<OtherShape>(
                   ModelService::IMAGE, ModelRunnerType::TT_SDXL_GENERATE, cfg),
               std::runtime_error);
}

TEST(RunnerRegistryTest, MediaNoMatchReturnsNullptr) {
  RunnerRegistry::instance().clear();
  tt::config::RunnerConfig cfg = tt::config::ImageConfig{};
  auto runner = RunnerRegistry::instance().createMedia<FakeMediaRunner>(
      ModelService::IMAGE, ModelRunnerType::TT_SDXL_GENERATE, cfg);
  EXPECT_EQ(runner, nullptr);
}

// ---------------------------------------------------------------------------
// RouteRegistry
// ---------------------------------------------------------------------------

TEST(RouteRegistryTest, ExactPathAllowedForActiveService) {
  RouteRegistry::instance().clear();
  RouteRegistry::instance().registerRoute(ModelService::LLM, "POST",
                                          "/v1/chat/completions", "");

  EXPECT_TRUE(RouteRegistry::instance().isAllowed(ModelService::LLM, "POST",
                                                  "/v1/chat/completions"));
  EXPECT_FALSE(RouteRegistry::instance().isAllowed(ModelService::LLM, "GET",
                                                   "/v1/chat/completions"));
  EXPECT_FALSE(RouteRegistry::instance().isAllowed(
      ModelService::EMBEDDING, "POST", "/v1/chat/completions"));
}

TEST(RouteRegistryTest, AlwaysExemptIsServiceAgnostic) {
  RouteRegistry::instance().clear();
  RouteRegistry::instance().registerAlwaysExempt("/health");
  RouteRegistry::instance().registerAlwaysExempt("/health");  // dedupe

  EXPECT_TRUE(
      RouteRegistry::instance().isAllowed(ModelService::LLM, "GET", "/health"));
  EXPECT_TRUE(RouteRegistry::instance().isAllowed(ModelService::EMBEDDING,
                                                  "GET", "/health"));
  EXPECT_EQ(RouteRegistry::instance().alwaysExemptPaths().size(), 1u);
}

TEST(RouteRegistryTest, IsAlwaysExemptDoesNotConsultPerServiceRoutes) {
  RouteRegistry::instance().clear();
  RouteRegistry::instance().registerAlwaysExempt("/metrics");
  RouteRegistry::instance().registerRoute(ModelService::LLM, "POST",
                                          "/v1/chat/completions", "");

  EXPECT_TRUE(RouteRegistry::instance().isAlwaysExempt("/metrics"));
  EXPECT_FALSE(
      RouteRegistry::instance().isAlwaysExempt("/v1/chat/completions"));
  EXPECT_FALSE(RouteRegistry::instance().isAlwaysExempt("/unknown"));
}

TEST(RouteRegistryTest, MalformedTemplateSegmentIsTreatedAsLiteral) {
  RouteRegistry::instance().clear();
  // Empty body and nested-brace shapes are NOT wildcards; they should match
  // their literal selves only.
  RouteRegistry::instance().registerRoute(ModelService::LLM, "GET", "/v1/{}",
                                          "empty");
  RouteRegistry::instance().registerRoute(ModelService::LLM, "GET",
                                          "/v1/{a{b}c}", "nested");

  EXPECT_TRUE(
      RouteRegistry::instance().isAllowed(ModelService::LLM, "GET", "/v1/{}"));
  EXPECT_FALSE(RouteRegistry::instance().isAllowed(ModelService::LLM, "GET",
                                                   "/v1/anything"));
  EXPECT_TRUE(RouteRegistry::instance().isAllowed(ModelService::LLM, "GET",
                                                  "/v1/{a{b}c}"));
}

TEST(RouteRegistryTest, RoutesForReturnsRegistrationOrder) {
  RouteRegistry::instance().clear();
  RouteRegistry::instance().registerRoute(ModelService::LLM, "post",
                                          "/v1/chat/completions", "chat");
  RouteRegistry::instance().registerRoute(ModelService::LLM, "get",
                                          "/v1/models", "models");

  auto routes = RouteRegistry::instance().routesFor(ModelService::LLM);
  ASSERT_EQ(routes.size(), 2u);
  EXPECT_EQ(routes[0].method, "POST");
  EXPECT_EQ(routes[0].path, "/v1/chat/completions");
  EXPECT_EQ(routes[0].description, "chat");
  EXPECT_EQ(routes[1].method, "GET");
  EXPECT_EQ(routes[1].path, "/v1/models");
}

// ---------------------------------------------------------------------------
// ServiceContainer
// ---------------------------------------------------------------------------

TEST(ServiceContainerTest, RegisterAndRetrieveByKey) {
  auto& c = ServiceContainer::instance();
  auto llm = std::make_shared<FakeService>("llm");
  auto emb = std::make_shared<FakeService>("emb");

  c.registerService(ModelService::LLM, llm);
  c.registerService(ModelService::EMBEDDING, emb);

  auto retrievedLlm =
      std::dynamic_pointer_cast<FakeService>(c.getService(ModelService::LLM));
  auto retrievedEmb = std::dynamic_pointer_cast<FakeService>(
      c.getService(ModelService::EMBEDDING));
  ASSERT_NE(retrievedLlm, nullptr);
  ASSERT_NE(retrievedEmb, nullptr);
  EXPECT_EQ(retrievedLlm->tag(), "llm");
  EXPECT_EQ(retrievedEmb->tag(), "emb");

  // Cleanup so we don't leak fake pointers to tests in other suites.
  c.registerService(ModelService::LLM, nullptr);
  c.registerService(ModelService::EMBEDDING, nullptr);
}

TEST(ServiceContainerTest, MissingKeyReturnsNullptr) {
  auto& c = ServiceContainer::instance();
  c.registerService(ModelService::LLM, nullptr);
  c.registerService(ModelService::EMBEDDING, nullptr);

  EXPECT_EQ(c.getService(ModelService::LLM), nullptr);
  EXPECT_EQ(c.getService(ModelService::EMBEDDING), nullptr);
}
