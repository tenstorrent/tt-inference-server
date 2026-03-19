// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llama_model_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "runners/llm_runner/sequence.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;

namespace llm_engine {

using Config = tt::config::LLMConfig;

namespace {
py::object gRunner;
py::object gStepSeqClass;

// ---------------------------------------------------------------------------
// KV cache serialization (page_tables <-> raw bytes)
//
// page_tables from Python: list[list[torch.Tensor]]
//   outer list: per sequence
//   inner list: per block
//   each tensor shape: (n_layers, 2, n_kv_heads, block_size, head_dim)
//
// Payload format (for one sequence):
//   [4B n_layers][4B n_kv_heads][4B block_size][4B head_dim]
//   [4B element_size]
//   [4B num_blocks]
//   for each block:
//     [raw tensor bytes]
// ---------------------------------------------------------------------------

constexpr size_t SHAPE_HEADER_SIZE = 6 * sizeof(uint32_t);

std::vector<uint8_t> serializePageTables(py::object pageTables, size_t seqIdx) {
  py::list seqBlocks = pageTables[py::int_(seqIdx)];
  auto numBlocks = static_cast<uint32_t>(py::len(seqBlocks));

  if (numBlocks == 0) return {};

  py::object firstTensor = seqBlocks[py::int_(0)];
  py::tuple shape = firstTensor.attr("shape");
  uint32_t nLayers = shape[py::int_(0)].cast<uint32_t>();
  uint32_t nKvHeads = shape[py::int_(2)].cast<uint32_t>();
  uint32_t blockSize = shape[py::int_(3)].cast<uint32_t>();
  uint32_t headDim = shape[py::int_(4)].cast<uint32_t>();
  uint32_t elemSize = firstTensor.attr("element_size")().cast<uint32_t>();

  size_t tensorBytes = static_cast<size_t>(nLayers) * 2 * nKvHeads * blockSize *
                       headDim * elemSize;
  size_t total = SHAPE_HEADER_SIZE + numBlocks * tensorBytes;

  std::vector<uint8_t> buf(total);
  uint8_t* p = buf.data();

  auto write32 = [&p](uint32_t v) {
    std::memcpy(p, &v, 4);
    p += 4;
  };

  write32(nLayers);
  write32(nKvHeads);
  write32(blockSize);
  write32(headDim);
  write32(elemSize);
  write32(numBlocks);

  for (uint32_t i = 0; i < numBlocks; ++i) {
    py::object t = seqBlocks[py::int_(i)].attr("contiguous")();
    auto dataPtr = t.attr("data_ptr")().cast<uintptr_t>();
    std::memcpy(p, reinterpret_cast<const void*>(dataPtr), tensorBytes);
    p += tensorBytes;
  }

  return buf;
}

py::list deserializePageTables(const std::vector<uint8_t>& payload) {
  if (payload.size() < SHAPE_HEADER_SIZE) {
    return py::list();
  }

  const uint8_t* p = payload.data();
  auto read32 = [&p]() -> uint32_t {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
  };

  uint32_t nLayers = read32();
  uint32_t nKvHeads = read32();
  uint32_t blockSize = read32();
  uint32_t headDim = read32();
  uint32_t elemSize = read32();
  uint32_t numBlocks = read32();

  size_t tensorBytes = static_cast<size_t>(nLayers) * 2 * nKvHeads * blockSize *
                       headDim * elemSize;

  py::module_ torch = py::module_::import("torch");

  // Map element_size to torch dtype
  py::object dtype;
  if (elemSize == 2) {
    dtype = torch.attr("bfloat16");
  } else if (elemSize == 4) {
    dtype = torch.attr("float32");
  } else {
    dtype = torch.attr("float16");
  }

  py::tuple shape = py::make_tuple(nLayers, 2, nKvHeads, blockSize, headDim);

  py::list blockTensors;
  for (uint32_t i = 0; i < numBlocks; ++i) {
    py::object t = torch.attr("empty")(shape, py::arg("dtype") = dtype);
    auto dstPtr = t.attr("data_ptr")().cast<uintptr_t>();
    std::memcpy(reinterpret_cast<void*>(dstPtr), p, tensorBytes);
    p += tensorBytes;
    blockTensors.append(t);
  }

  return blockTensors;
}

}  // namespace

bool LlamaModelRunner::initialize() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }

  bool success = false;
  try {
    py::module_ sysMod = py::module_::import("sys");
    py::list sysPath = sysMod.attr("path");

    const char* pythonPath = std::getenv("TT_PYTHON_PATH");
    if (pythonPath && *pythonPath) {
      sysPath.attr("insert")(0, pythonPath);
    }
    const char* metalHome = std::getenv("TT_METAL_HOME");
    if (metalHome && *metalHome) {
      sysPath.attr("insert")(0, metalHome);
    }

    py::module_ llamaMod = py::module_::import("tt_model_runners.llama_runner");
    gStepSeqClass = llamaMod.attr("StepSequence");
    py::object runnerClass = llamaMod.attr("Llama31_8BRunner");

    const char* envDev = std::getenv("TT_VISIBLE_DEVICES");
    std::string deviceId = (envDev && *envDev) ? envDev : "0";
    gRunner = runnerClass(deviceId);

    py::module_ asyncio = py::module_::import("asyncio");
    bool warmupOk = asyncio.attr("run")(gRunner.attr("warmup")()).cast<bool>();
    if (!warmupOk) {
      TT_LOG_ERROR("[LlamaModelRunner] Warmup failed");
    } else {
      TT_LOG_INFO("[LlamaModelRunner] Llama runner ready (in-process)");
      initialized_ = true;
      success = true;
    }
  } catch (const py::error_already_set& e) {
    TT_LOG_ERROR("[LlamaModelRunner] Python init error: {}", e.what());
  }

  PyEval_SaveThread();
  return success;
}

void LlamaModelRunner::failSequences(const std::vector<Sequence*>& seqs) {
  for (Sequence* seq : seqs) {
    TokenResult dr(seq->taskId, 0, {}, true);
    decode_callback_(dr);
  }
}

LlamaModelRunner::LlamaModelRunner(const Config& config,
                                   DecodeCallback callback,
                                   std::unique_ptr<IKVCacheMigrator> migrator)
    : config_(config),
      decode_callback_(std::move(callback)),
      migrator_(std::move(migrator)) {
  initialize();

  if (migrator_) {
    migrator_->setReceiveCallback([](KVCacheMigrationData data) {
      py::gil_scoped_acquire acquire;
      try {
        py::list pageTensors = deserializePageTables(data.payload);
        py::list blockIds;
        for (int id : data.block_ids) {
          blockIds.append(id);
        }
        py::module_ asyncio = py::module_::import("asyncio");
        asyncio.attr("run")(
            gRunner.attr("write_page_table")(blockIds, pageTensors));
        TT_LOG_INFO("[LlamaModelRunner] write_page_table completed for task {}",
                    data.task_id);
      } catch (const py::error_already_set& e) {
        TT_LOG_ERROR(
            "[LlamaModelRunner] write_page_table failed for task {}: {}",
            data.task_id, e.what());
      }
    });
    migrator_->start();
    TT_LOG_INFO("[LlamaModelRunner] KV cache migrator started");
  }
}

LlamaModelRunner::~LlamaModelRunner() { exit(); }

void LlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool isPrefill) {
  if (stop_.load() || !initialized_) return;

  bool hadError = false;
  {
    py::gil_scoped_acquire acquire;
    try {
      py::list pySeqs;
      for (Sequence* seq : seqs) {
        py::list tokenIds;
        if (isPrefill) {
          for (int64_t t : seq->tokenIds) tokenIds.append(t);
        } else {
          tokenIds.append(seq->tokenIds.back());
        }

        py::list blockTable;
        for (int bid : seq->blockTable) {
          blockTable.append(bid);
        }

        int currentPos =
            isPrefill ? 0 : static_cast<int>(seq->tokenIds.size() - 1);
        int promptLen = static_cast<int>(seq->numPromptTokens);

        const SamplingParams* sp = seq->samplingParams.get();
        double temperature = sp ? static_cast<double>(sp->temperature) : 1.0;
        bool ignoreEos = sp ? sp->ignore_eos : false;

        py::object seed = py::none();
        if (sp && sp->seed.has_value()) {
          seed = py::int_(*sp->seed);
        }

        py::object topP = py::none();
        if (sp && sp->top_p.has_value()) {
          topP = py::float_(*sp->top_p);
        }

        py::object topK = py::none();
        if (sp && sp->top_k.has_value()) {
          topK = py::int_(*sp->top_k);
        }

        py::object minP = py::none();
        if (sp && sp->min_p.has_value()) {
          minP = py::float_(*sp->min_p);
        }

        py::object repetitionPenalty = py::none();
        if (sp && sp->repetition_penalty.has_value()) {
          repetitionPenalty = py::float_(*sp->repetition_penalty);
        }

        double presencePenalty =
            sp ? static_cast<double>(sp->presence_penalty) : 0.0;
        double frequencyPenalty =
            sp ? static_cast<double>(sp->frequency_penalty) : 0.0;

        pySeqs.append(gStepSeqClass(
            seq->taskId, tokenIds, temperature, ignoreEos, blockTable,
            currentPos, promptLen, seed, topP, topK, minP, repetitionPenalty,
            presencePenalty, frequencyPenalty));
      }

      // First decode step after prefill must set reset_batch=true so on-device
      // sampling state is initialized correctly.
      bool resetBatch = !isPrefill && lastStepWasPrefill_;
      lastStepWasPrefill_ = isPrefill;

      py::object runResult = gRunner.attr("run")(isPrefill, pySeqs, resetBatch);
      py::list results = runResult.attr("results");

      py::object lastPrefillPageTables = py::none();
      if (isPrefill && migrator_) {
        py::object pageTables = runResult.attr("page_tables");
        if (!pageTables.is_none()) {
          for (size_t i = 0; i < seqs.size(); ++i) {
            auto serialized = serializePageTables(pageTables, i);
            if (!serialized.empty()) {
              migrator_->send({seqs[i]->task_id.id, seqs[i]->block_table_,
                               std::move(serialized)});
            }
          }
        }
      }

      for (size_t i = 0; i < seqs.size(); ++i) {
        py::object item = results[py::int_(i)];
        uint32_t drTaskId(item.attr("task_id").cast<std::string>());
        uint64_t drTokenId =
            static_cast<uint64_t>(item.attr("token_id").cast<int64_t>());
        std::string error = item.attr("error").cast<std::string>();
        bool drIsError = !error.empty();
        if (drIsError) {
          TT_LOG_ERROR("[LlamaModelRunner] sequence {} error: {}", drTaskId.id,
                       error);
        }
        TokenResult dr(drTaskId, drTokenId, {}, drIsError);
        decode_callback_(dr);
      }
    } catch (const py::error_already_set& e) {
      TT_LOG_ERROR("[LlamaModelRunner] Python error in run_step: {}", e.what());
      hadError = true;
    }
  }
  if (hadError) {
    failSequences(seqs);
    stop_.store(true);
  }
}

void LlamaModelRunner::exit() {
  if (stop_.exchange(true)) return;
  if (migrator_) {
    migrator_->stop();
  }
  if (!initialized_) return;
  {
    py::gil_scoped_acquire acquire;
    gRunner = py::object();
    gStepSeqClass = py::object();
  }
  initialized_ = false;
  TT_LOG_INFO("[LlamaModelRunner] Runner exited");
}

std::unique_ptr<IModelRunner> makeLlamaModelRunner(
    const Config& config, DecodeCallback callback,
    std::unique_ptr<IKVCacheMigrator> migrator) {
  auto runner = std::make_unique<LlamaModelRunner>(config, std::move(callback),
                                                   std::move(migrator));
  if (!runner->isReady()) {
    return nullptr;
  }
  return runner;
}

}  // namespace llm_engine
