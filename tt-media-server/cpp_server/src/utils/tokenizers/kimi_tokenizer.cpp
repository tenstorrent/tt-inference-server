// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/kimi_tokenizer.hpp"

#include <json/json.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::utils::tokenizers {

namespace {

// Pre-tokenization regex copied verbatim from moonshotai/Kimi-K2.6's
// tokenization_kimi.py. Combined into one alternation, same as the Python
// side. Kept here so we don't depend on parsing their .py at runtime.
constexpr const char* kKimiPatStr =
    R"([\p{Han}]+|)"
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
    R"(\p{N}{1,3}|)"
    R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
    R"(\s*[\r\n]+|)"
    R"(\s+(?!\S)|)"
    R"(\s+)";

constexpr int kNumReservedSpecialTokens = 256;

/**
 * Initialize the embedded Python interpreter exactly once per process. After
 * init, release the GIL via pybind11's idiom so worker threads can acquire on
 * demand. The release object lives for the process lifetime — we don't ever
 * re-acquire on the initializing thread, only on the thread doing tokenizer
 * work (via gil_scoped_acquire). Matches the lifecycle other embedded-Python
 * runners in this codebase use.
 */
void ensurePythonInitialized() {
  static std::once_flag flag;
  // Static unique_ptr so the release survives until process exit and pybind11
  // sees the GIL as "released by main thread" for the whole run.
  static std::unique_ptr<py::gil_scoped_release> mainThreadReleased;
  std::call_once(flag, []() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter();
      mainThreadReleased = std::make_unique<py::gil_scoped_release>();
    }
  });
}

std::string readFile(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    throw std::runtime_error("[KimiTokenizer] Failed to open: " + p.string());
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/**
 * Convert JsonCpp Value to a Python object by round-tripping through
 * a compact JSON string + json.loads. Used to push Tool definitions into
 * the Jinja environment without hand-walking the JsonCpp tree.
 */
py::object jsoncppToPy(const Json::Value& v) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::string s = Json::writeString(builder, v);
  py::object json_mod = py::module_::import("json");
  return json_mod.attr("loads")(s);
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct KimiTokenizer::Impl {
  // Held under the GIL; constructed once in the ctor, called per encode/decode.
  py::object encoding;     // tiktoken.Encoding
  py::object jinja_tmpl;   // jinja2.Template
  int64_t eos_id = 163586;

  // py::object's destructor releases a Python reference, which requires the
  // GIL. If KimiTokenizer's ctor throws partway through, stack unwinding will
  // destroy this Impl outside any gil_scoped_acquire — so we must take the
  // GIL ourselves before the py::object fields die.
  ~Impl() {
    if (!encoding.ptr() && !jinja_tmpl.ptr()) return;
    try {
      py::gil_scoped_acquire gil;
      encoding.release().dec_ref();
      jinja_tmpl.release().dec_ref();
    } catch (...) {
      // During process shutdown the interpreter may already be gone; OK.
    }
  }
};

KimiTokenizer::KimiTokenizer(const std::string& tiktokenModelPath)
    : Tokenizer(), impl_(std::make_unique<Impl>()) {
  ensurePythonInitialized();

  std::filesystem::path tiktokenPath(tiktokenModelPath);
  if (!std::filesystem::exists(tiktokenPath)) {
    throw std::runtime_error("[KimiTokenizer] Missing tiktoken.model at " +
                             tiktokenPath.string());
  }
  auto modelDir = tiktokenPath.parent_path();
  auto cfgPath = modelDir / "tokenizer_config.json";
  auto tmplPath = modelDir / "chat_template.jinja";

  if (!std::filesystem::exists(cfgPath)) {
    throw std::runtime_error("[KimiTokenizer] Missing tokenizer_config.json at " +
                             cfgPath.string());
  }
  if (!std::filesystem::exists(tmplPath)) {
    throw std::runtime_error("[KimiTokenizer] Missing chat_template.jinja at " +
                             tmplPath.string());
  }

  // Load tokenizer_config.json to discover special-token names + their IDs.
  Json::Value cfgRoot;
  {
    std::ifstream f(cfgPath, std::ios::binary);
    Json::CharReaderBuilder rb;
    std::string errs;
    if (!Json::parseFromStream(rb, f, &cfgRoot, &errs)) {
      throw std::runtime_error("[KimiTokenizer] Failed to parse " +
                               cfgPath.string() + ": " + errs);
    }
  }

  std::string templateSource = readFile(tmplPath);

  try {
    py::gil_scoped_acquire gil;

    // Import tiktoken and load tiktoken.model into a {bytes -> rank} dict.
    py::module_ tiktoken = py::module_::import("tiktoken");
    py::module_ tiktoken_load = py::module_::import("tiktoken.load");
    py::object mergeable_ranks =
        tiktoken_load.attr("load_tiktoken_bpe")(tiktokenPath.string());
    int64_t num_base = py::len(mergeable_ranks);

    // Build the special-tokens dict the same way tokenization_kimi.py does:
    // 256 contiguous slots starting at num_base, populated from
    // added_tokens_decoder where named, else <|reserved_token_<id>|>.
    py::dict specialTokens;
    Json::Value added = cfgRoot.get("added_tokens_decoder", Json::Value(Json::objectValue));
    for (int64_t i = num_base; i < num_base + kNumReservedSpecialTokens; ++i) {
      std::string key = std::to_string(i);
      std::string content;
      if (added.isMember(key)) {
        const auto& entry = added[key];
        if (entry.isObject() && entry.isMember("content")) {
          content = entry["content"].asString();
        } else if (entry.isString()) {
          content = entry.asString();
        }
      }
      if (content.empty()) {
        content = "<|reserved_token_" + std::to_string(i) + "|>";
      }
      specialTokens[py::cast(content)] = py::cast(i);
    }

    py::object Encoding = tiktoken.attr("Encoding");
    impl_->encoding = Encoding(
        py::arg("name") = "moonshotai/Kimi-K2.6",
        py::arg("pat_str") = kKimiPatStr,
        py::arg("mergeable_ranks") = mergeable_ranks,
        py::arg("special_tokens") = specialTokens);

    // Compile the Jinja chat template. Enable loopcontrols (for {% break %}
    // / {% continue %} — Kimi K2.6's template uses break). Match the
    // configuration transformers.apply_chat_template uses.
    py::module_ jinja2 = py::module_::import("jinja2");
    py::object Environment = jinja2.attr("Environment");
    py::list extensions;
    extensions.append("jinja2.ext.loopcontrols");
    py::object env = Environment(
        py::arg("extensions") = extensions,
        py::arg("trim_blocks") = false,
        py::arg("lstrip_blocks") = false,
        py::arg("keep_trailing_newline") = true);
    // Add tojson filter (Jinja2 doesn't include it by default in Environment).
    py::object json_mod = py::module_::import("json");
    env.attr("filters")["tojson"] = json_mod.attr("dumps");
    impl_->jinja_tmpl = env.attr("from_string")(templateSource);

    // Cache the EOS id from tokenizer_config so we don't reach into Python
    // every time stopTokenIds() is called. (Hard-coded fallback matches
    // generation_config.json's eos_token_id.)
    // Note: stopTokenIds() returns {163586} statically; this is here for
    // future-proofing if we expose configurable stop sets.

  } catch (const py::error_already_set& e) {
    // py::error_already_set::what() formats the Python exception via the
    // Python API, so it needs the GIL.
    py::gil_scoped_acquire gil;
    throw std::runtime_error(std::string("[KimiTokenizer] Python init failed: ") +
                             e.what());
  }

  TT_LOG_INFO("[KimiTokenizer] Loaded moonshotai/Kimi-K2.6 from {} (tiktoken backend)",
              modelDir.string());
}

// Out-of-line so the unique_ptr<Impl> destructor sees the complete Impl type.
// The actual py::object cleanup happens inside Impl::~Impl under the GIL.
KimiTokenizer::~KimiTokenizer() = default;

bool KimiTokenizer::isLoaded() const {
  return impl_ && impl_->encoding && impl_->jinja_tmpl;
}

std::vector<int> KimiTokenizer::encode(const std::string& text) const {
  if (!isLoaded()) {
    throw std::runtime_error("[KimiTokenizer] Not loaded, cannot encode");
  }
  if (text.empty()) return {};
  try {
    py::gil_scoped_acquire gil;
    // allowed_special="all" so the Jinja-rendered template's literal special
    // tokens (e.g. "<|im_user|>") get encoded as their reserved IDs instead of
    // being BPE-split as text.
    py::object ids = impl_->encoding.attr("encode")(
        text, py::arg("allowed_special") = "all");
    return ids.cast<std::vector<int>>();
  } catch (const py::error_already_set& e) {
    py::gil_scoped_acquire gil;
    throw std::runtime_error(std::string("[KimiTokenizer] encode failed: ") +
                             e.what());
  }
}

std::string KimiTokenizer::decode(const std::vector<int>& tokenIds,
                                  bool skipSpecialTokens) const {
  if (!isLoaded()) {
    throw std::runtime_error("[KimiTokenizer] Not loaded, cannot decode");
  }
  if (tokenIds.empty()) return "";
  try {
    py::gil_scoped_acquire gil;
    // tiktoken.Encoding.decode accepts a list of ints and returns a str.
    // Special tokens are decoded as their literal strings unless filtered.
    // tiktoken doesn't accept a skip_special_tokens flag, so we mask them
    // out ourselves.
    if (skipSpecialTokens) {
      std::vector<int> filtered;
      filtered.reserve(tokenIds.size());
      // Anything at or above num_base is in the 256 reserved slot range.
      // We could call impl_->encoding.attr("n_vocab") - 256, but the cheaper
      // path is the eot_token from added_tokens_decoder; here we just filter
      // all known specials via Python.
      py::list pyAll = py::cast(tokenIds);
      py::object filteredPy = impl_->encoding.attr("decode")(
          pyAll, py::arg("errors") = "replace");
      // Strip the special-token strings post-hoc. tiktoken's decode emits
      // literal text for them ("<|im_end|>" etc.); for skipSpecialTokens we'd
      // want them removed. Easiest: have tiktoken decode with
      // decode_with_offsets style or use decode_tokens_bytes. For minimal
      // β we do per-id filtering via the Python encoding's special_tokens_set.
      py::set specials = impl_->encoding.attr("special_tokens_set");
      py::list pyFiltered;
      // We can't easily ask tiktoken "is id N special?" without the inverse
      // mapping; instead use the encoding's special_tokens dict.
      py::dict specialMap = impl_->encoding.attr("_special_tokens");
      py::set specialIds;
      for (auto kv : specialMap) {
        specialIds.add(kv.second);
      }
      for (int id : tokenIds) {
        if (!specialIds.contains(py::cast(id))) {
          pyFiltered.append(id);
        }
      }
      py::object decoded =
          impl_->encoding.attr("decode")(pyFiltered, py::arg("errors") = "replace");
      return decoded.cast<std::string>();
    }
    py::list pyIds = py::cast(tokenIds);
    py::object decoded =
        impl_->encoding.attr("decode")(pyIds, py::arg("errors") = "replace");
    return decoded.cast<std::string>();
  } catch (const py::error_already_set& e) {
    py::gil_scoped_acquire gil;
    throw std::runtime_error(std::string("[KimiTokenizer] decode failed: ") +
                             e.what());
  }
}

std::string KimiTokenizer::applyChatTemplate(
    const std::vector<tt::domain::llm::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools,
    bool enableReasoning, bool skipApplyChatTemplate) const {
  if (skipApplyChatTemplate) {
    std::ostringstream out;
    for (const auto& m : messages) out << m.content;
    return out.str();
  }
  if (!isLoaded()) {
    throw std::runtime_error(
        "[KimiTokenizer] Not loaded, cannot apply chat template");
  }
  try {
    py::gil_scoped_acquire gil;

    py::list pyMessages;
    for (const auto& m : messages) {
      py::dict d;
      d["role"] = m.role;
      d["content"] = m.content;
      if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
        py::list arr;
        for (const auto& tc : *m.tool_calls) {
          arr.append(jsoncppToPy(tc.toJson()));
        }
        d["tool_calls"] = arr;
      }
      if (m.tool_call_id.has_value()) {
        d["tool_call_id"] = *m.tool_call_id;
      }
      pyMessages.append(d);
    }

    py::object pyTools = py::none();
    if (tools.has_value() && !tools->empty()) {
      py::list arr;
      for (const auto& t : *tools) {
        arr.append(jsoncppToPy(t.toJson()));
      }
      pyTools = arr;
    }

    // Expose enable_reasoning under both names so the template can use either.
    py::object rendered = impl_->jinja_tmpl.attr("render")(
        py::arg("messages") = pyMessages,
        py::arg("tools") = pyTools,
        py::arg("add_generation_prompt") = addGenerationPrompt,
        py::arg("enable_thinking") = enableReasoning,
        py::arg("enable_reasoning") = enableReasoning);
    return rendered.cast<std::string>();
  } catch (const py::error_already_set& e) {
    py::gil_scoped_acquire gil;
    throw std::runtime_error(
        std::string("[KimiTokenizer] applyChatTemplate failed: ") + e.what());
  }
}

}  // namespace tt::utils::tokenizers
