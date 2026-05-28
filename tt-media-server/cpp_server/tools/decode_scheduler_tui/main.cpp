// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// decode_scheduler_tui — standalone TUI that drives tt-llm-engine's
// DecodeScheduler exactly the way runtime/runners/blaze_runner/blaze_runner.cpp
// does (ALLOCATE / SUBMIT / CONTINUE / STOP / EVICT), but from a terminal UI
// instead of from IPC queues. Use it to validate scheduler behavior against a
// deployed model (or PipelineSimulatorConfig) without bringing up the rest of
// the inference server.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_llm_engine/pipeline/pipeline_types.hpp>
#include <tt_llm_engine/scheduler/decode/decode_scheduler.hpp>
#include <tt_llm_engine/scheduler/decode/decode_types.hpp>

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

namespace ds = tt_llm_engine::scheduler::decode;
namespace pl = tt_llm_engine::pipeline;

namespace {

constexpr size_t kRecentTokensPerSlot = 16;
constexpr size_t kEventLogCap = 500;
constexpr auto kPollInterval = std::chrono::milliseconds(2);
constexpr auto kUiTick = std::chrono::milliseconds(50);

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

struct CliArgs {
  std::string backend = "sim";        // "sim" or "socket"
  uint32_t max_users = ds::DEFAULT_MAX_USERS;
  uint32_t max_seq_len = ds::DEFAULT_MAX_SEQ_LEN;
  uint32_t eos_token = ds::DEFAULT_EOS_TOKEN;
  // SocketConfig
  std::string h2d_socket_id = "tt_llm_h2d";
  std::string d2h_socket_id = "tt_llm_d2h";
  uint32_t connect_timeout_ms = 30000;
  bool use_deepseek_md_format = false;
  // PipelineSimulatorConfig
  uint32_t sim_num_stages = 64;
  uint32_t sim_stage_duration_us = 44;
  uint32_t sim_decode_token_id = pl::EMPTY_TOKEN;
  float sim_accept_rate = 1.0f;
};

void printUsage() {
  std::cerr <<
      "usage: decode_scheduler_tui [options]\n"
      "  --backend=sim|socket           (default: sim)\n"
      "  --max-users=N                  (default: 64)\n"
      "  --max-seq-len=N                (default: 131072)\n"
      "  --eos-token=ID                 (default: 1)\n"
      "  socket backend:\n"
      "    --h2d=NAME --d2h=NAME --connect-timeout-ms=N --deepseek-md\n"
      "  sim backend:\n"
      "    --sim-num-stages=N --sim-stage-us=N --sim-decode-token=ID --sim-accept-rate=F\n"
      "\n"
      "in-TUI commands:\n"
      "  alloc                       — push ALLOCATE (slot assigned by scheduler)\n"
      "  submit <slot> <t1,t2,..> [max=N] [temp=F] [top_p=F] [top_k=N] [ignore_eos]\n"
      "                              [spec] [stop=t1,t2]\n"
      "  continue <slot> <t1,t2,..>  [same flags as submit; pos=N optional]\n"
      "  stop <slot>                 — push STOP\n"
      "  evict <slot>                — push EVICT\n"
      "  dump                        — print scheduler diagnostics to event log\n"
      "  clear                       — clear event log\n"
      "  help                        — show this help inside the TUI\n"
      "  quit                        — exit\n";
}

bool parseUint(const std::string& s, uint32_t& out) {
  try {
    size_t end = 0;
    auto v = std::stoul(s, &end);
    if (end != s.size()) return false;
    out = static_cast<uint32_t>(v);
    return true;
  } catch (...) { return false; }
}

bool parseFloat(const std::string& s, float& out) {
  try {
    size_t end = 0;
    out = std::stof(s, &end);
    return end == s.size();
  } catch (...) { return false; }
}

std::vector<std::string> splitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::optional<CliArgs> parseCli(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? "" : arg.substr(eq + 1);

    if (key == "--help" || key == "-h") { printUsage(); return std::nullopt; }
    else if (key == "--backend") a.backend = val;
    else if (key == "--max-users") { if (!parseUint(val, a.max_users)) return std::nullopt; }
    else if (key == "--max-seq-len") { if (!parseUint(val, a.max_seq_len)) return std::nullopt; }
    else if (key == "--eos-token") { if (!parseUint(val, a.eos_token)) return std::nullopt; }
    else if (key == "--h2d") a.h2d_socket_id = val;
    else if (key == "--d2h") a.d2h_socket_id = val;
    else if (key == "--connect-timeout-ms") { if (!parseUint(val, a.connect_timeout_ms)) return std::nullopt; }
    else if (key == "--deepseek-md") a.use_deepseek_md_format = true;
    else if (key == "--sim-num-stages") { if (!parseUint(val, a.sim_num_stages)) return std::nullopt; }
    else if (key == "--sim-stage-us") { if (!parseUint(val, a.sim_stage_duration_us)) return std::nullopt; }
    else if (key == "--sim-decode-token") { if (!parseUint(val, a.sim_decode_token_id)) return std::nullopt; }
    else if (key == "--sim-accept-rate") { if (!parseFloat(val, a.sim_accept_rate)) return std::nullopt; }
    else { std::cerr << "unknown arg: " << arg << "\n"; printUsage(); return std::nullopt; }
  }
  if (a.backend != "sim" && a.backend != "socket") {
    std::cerr << "--backend must be sim or socket\n";
    return std::nullopt;
  }
  return a;
}

pl::PipelineConfig makePipelineConfig(const CliArgs& a) {
  if (a.backend == "socket") {
    return pl::SocketConfig{
        .h2d_socket_id = a.h2d_socket_id,
        .d2h_socket_id = a.d2h_socket_id,
        .connect_timeout_ms = a.connect_timeout_ms,
        .use_deepseek_md_format = a.use_deepseek_md_format,
    };
  }
  return pl::PipelineSimulatorConfig{
      .num_stages = a.sim_num_stages,
      .stage_duration_us = a.sim_stage_duration_us,
      .decode_token_id = a.sim_decode_token_id,
      .accept_rate = a.sim_accept_rate,
  };
}

// ---------------------------------------------------------------------------
// Pretty-printers for engine types (so we can log events meaningfully)
// ---------------------------------------------------------------------------

const char* toString(ds::RequestType t) {
  switch (t) {
    case ds::RequestType::ALLOCATE: return "ALLOCATE";
    case ds::RequestType::SUBMIT:   return "SUBMIT";
    case ds::RequestType::CONTINUE: return "CONTINUE";
    case ds::RequestType::EVICT:    return "EVICT";
    case ds::RequestType::STOP:     return "STOP";
  }
  return "?";
}

const char* toString(ds::UserState s) {
  switch (s) {
    case ds::UserState::INACTIVE: return "INACTIVE";
    case ds::UserState::PREFILL:  return "PREFILL";
    case ds::UserState::DECODE:   return "DECODE";
    case ds::UserState::COMPLETE: return "COMPLETE";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Shared state: poller thread writes, UI thread reads (under mutex).
// ---------------------------------------------------------------------------

struct SlotView {
  std::deque<uint32_t> recent_tokens;
  uint32_t tokens_generated = 0;
  uint32_t last_position = 0;
  uint32_t last_generation = 0;
  bool last_complete = false;
};

struct EventEntry {
  std::chrono::steady_clock::time_point t;
  std::string text;
};

class AppState {
 public:
  explicit AppState(std::chrono::steady_clock::time_point start) : start_(start) {}

  void logEvent(std::string text) {
    std::lock_guard lk(m_);
    events_.push_back({std::chrono::steady_clock::now(), std::move(text)});
    while (events_.size() > kEventLogCap) events_.pop_front();
  }

  void recordOutput(const ds::OutputMessage& o) {
    std::lock_guard lk(m_);
    auto& s = slots_[o.slot_id];
    s.recent_tokens.push_back(o.token_id);
    while (s.recent_tokens.size() > kRecentTokensPerSlot) s.recent_tokens.pop_front();
    s.tokens_generated = o.tokens_generated;
    s.last_position = o.position_id;
    s.last_generation = o.generation;
    s.last_complete = o.is_complete;

    std::ostringstream os;
    os << "tok slot=" << o.slot_id << " id=" << o.token_id
       << " pos=" << o.position_id << " gen=" << o.tokens_generated
       << (o.is_complete ? " FINAL" : "")
       << (o.ctx_exhausted ? " CTX_EXHAUSTED" : "");
    events_.push_back({std::chrono::steady_clock::now(), os.str()});
    while (events_.size() > kEventLogCap) events_.pop_front();
  }

  uint32_t nextRequestId() {
    std::lock_guard lk(m_);
    return next_request_id_++;
  }

  // Snapshot for UI thread. Cheap small copies; called on UI tick.
  struct Snapshot {
    std::unordered_map<uint32_t, SlotView> slots;
    std::vector<EventEntry> events;
    uint32_t next_request_id = 0;
  };
  Snapshot snapshot() {
    std::lock_guard lk(m_);
    Snapshot s;
    s.slots = slots_;
    s.events.assign(events_.begin(), events_.end());
    s.next_request_id = next_request_id_;
    return s;
  }

  void clearLog() {
    std::lock_guard lk(m_);
    events_.clear();
  }

  std::chrono::steady_clock::time_point start() const { return start_; }

 private:
  std::mutex m_;
  std::unordered_map<uint32_t, SlotView> slots_;
  std::deque<EventEntry> events_;
  uint32_t next_request_id_ = 1;
  std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Background poller — drains responses and outputs from the scheduler.
// ---------------------------------------------------------------------------

class Poller {
 public:
  Poller(ds::DecodeScheduler& sched, AppState& state)
      : sched_(sched), state_(state) {}

  void start() {
    running_.store(true);
    thread_ = std::thread([this] { run(); });
  }
  void stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
  }

 private:
  void run() {
    while (running_.load(std::memory_order_relaxed)) {
      bool did_work = false;
      ds::SchedulerResponse resp{};
      while (sched_.try_pop_response(resp)) {
        std::ostringstream os;
        os << "ack " << toString(resp.request_type) << " req=" << resp.request_id
           << " slot=" << (resp.slot_id == pl::INVALID_SLOT
                               ? std::string("INVALID")
                               : std::to_string(resp.slot_id))
           << " err=" << resp.error_code;
        state_.logEvent(os.str());
        did_work = true;
      }
      ds::OutputMessage out{};
      while (sched_.try_pop_output(out)) {
        state_.recordOutput(out);
        did_work = true;
      }
      if (!did_work) std::this_thread::sleep_for(kPollInterval);
    }
  }

  ds::DecodeScheduler& sched_;
  AppState& state_;
  std::thread thread_;
  std::atomic<bool> running_{false};
};

// ---------------------------------------------------------------------------
// Command parser — turns a line of text into an ISRequest (or a side effect).
// Mirrors the request shapes blaze_runner builds via blaze_utils helpers.
// ---------------------------------------------------------------------------

struct ParsedSamplingFlags {
  uint32_t max_new_tokens = 32;
  float temperature = 0.0f;
  float top_p = 1.0f;
  int32_t top_k = -1;
  bool ignore_eos = false;
  bool spec_decode = false;
  bool disaggregated = false;
  std::optional<uint32_t> position_id;
  std::vector<uint32_t> stop_tokens;
};

bool parseFlagPairs(const std::vector<std::string>& tokens, size_t start,
                    ParsedSamplingFlags& out, std::string& err) {
  for (size_t i = start; i < tokens.size(); ++i) {
    const auto& tk = tokens[i];
    if (tk == "ignore_eos") { out.ignore_eos = true; continue; }
    if (tk == "spec")       { out.spec_decode = true; continue; }
    if (tk == "disagg")     { out.disaggregated = true; continue; }
    auto eq = tk.find('=');
    if (eq == std::string::npos) { err = "bad flag '" + tk + "'"; return false; }
    std::string k = tk.substr(0, eq);
    std::string v = tk.substr(eq + 1);
    if (k == "max")        { if (!parseUint(v, out.max_new_tokens)) { err = "bad max"; return false; } }
    else if (k == "temp")  { if (!parseFloat(v, out.temperature))   { err = "bad temp"; return false; } }
    else if (k == "top_p") { if (!parseFloat(v, out.top_p))         { err = "bad top_p"; return false; } }
    else if (k == "top_k") {
      try { out.top_k = static_cast<int32_t>(std::stol(v)); }
      catch (...) { err = "bad top_k"; return false; }
    }
    else if (k == "pos")   {
      uint32_t p = 0;
      if (!parseUint(v, p)) { err = "bad pos"; return false; }
      out.position_id = p;
    }
    else if (k == "stop")  {
      for (const auto& s : splitCsv(v)) {
        uint32_t id = 0;
        if (!parseUint(s, id)) { err = "bad stop id '" + s + "'"; return false; }
        out.stop_tokens.push_back(id);
      }
    }
    else { err = "unknown flag '" + k + "'"; return false; }
  }
  return true;
}

ds::GenerationParams toGenerationParams(const ParsedSamplingFlags& f) {
  ds::GenerationParams g;
  g.max_new_tokens = f.max_new_tokens;
  g.spec_decode = f.spec_decode;
  g.ignore_eos = f.ignore_eos;
  g.temperature = f.temperature;
  g.top_p = f.top_p;
  g.top_k = f.top_k;
  g.disaggregated_decode = f.disaggregated;
  g.position_id = f.position_id;
  g.stop_tokens = f.stop_tokens;
  return g;
}

std::vector<std::string> tokenizeLine(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
        c == '\v' || c == '\f') {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool parseSlotAndTokens(const std::vector<std::string>& argv, size_t base,
                        uint32_t& slot, std::vector<uint32_t>& tokens,
                        std::string& err) {
  if (argv.size() < base + 2) { err = "need <slot> <tokens>"; return false; }
  if (!parseUint(argv[base], slot)) { err = "bad slot"; return false; }
  for (const auto& s : splitCsv(argv[base + 1])) {
    uint32_t id = 0;
    if (!parseUint(s, id)) { err = "bad token '" + s + "'"; return false; }
    tokens.push_back(id);
  }
  if (tokens.empty()) { err = "no tokens"; return false; }
  return true;
}

enum class CommandResult { Handled, Quit, Unknown };

CommandResult handleCommand(const std::string& line,
                            ds::DecodeScheduler& sched,
                            AppState& state) {
  auto tokens = tokenizeLine(line);
  if (tokens.empty()) return CommandResult::Handled;
  const std::string& cmd = tokens[0];

  auto logErr = [&](const std::string& msg) {
    state.logEvent("error: " + msg);
  };

  if (cmd == "quit" || cmd == "exit") return CommandResult::Quit;

  if (cmd == "help") {
    state.logEvent("commands: alloc | submit S T,T [flags] | continue S T,T [flags] |");
    state.logEvent("          stop S | evict S | dump | clear | help | quit");
    state.logEvent("flags: max=N temp=F top_p=F top_k=N pos=N stop=t,t ignore_eos spec disagg");
    return CommandResult::Handled;
  }

  if (cmd == "clear") { state.clearLog(); return CommandResult::Handled; }

  if (cmd == "dump") {
    std::ostringstream os;
    sched.dump_diagnostics(os);
    std::string line2;
    std::istringstream is(os.str());
    while (std::getline(is, line2)) state.logEvent("[dump] " + line2);
    return CommandResult::Handled;
  }

  if (cmd == "alloc" || cmd == "allocate") {
    ds::ISRequest req{};
    req.type = ds::RequestType::ALLOCATE;
    req.request_id = state.nextRequestId();
    if (!sched.push_request(req)) {
      logErr("push_request returned false (queue full)");
      return CommandResult::Handled;
    }
    state.logEvent("pushed ALLOCATE req=" + std::to_string(req.request_id));
    return CommandResult::Handled;
  }

  if (cmd == "stop" || cmd == "evict") {
    if (tokens.size() < 2) { logErr("usage: " + cmd + " <slot>"); return CommandResult::Handled; }
    uint32_t slot = 0;
    if (!parseUint(tokens[1], slot)) { logErr("bad slot"); return CommandResult::Handled; }
    ds::ISRequest req{};
    req.type = (cmd == "stop") ? ds::RequestType::STOP : ds::RequestType::EVICT;
    req.request_id = state.nextRequestId();
    req.slot_id = slot;
    if (!sched.push_request(req)) {
      logErr("push_request returned false (queue full)");
      return CommandResult::Handled;
    }
    std::ostringstream os;
    os << "pushed " << toString(req.type) << " req=" << req.request_id
       << " slot=" << slot;
    state.logEvent(os.str());
    return CommandResult::Handled;
  }

  if (cmd == "submit" || cmd == "continue") {
    uint32_t slot = 0;
    std::vector<uint32_t> toks;
    std::string err;
    if (!parseSlotAndTokens(tokens, 1, slot, toks, err)) {
      logErr(cmd + ": " + err);
      return CommandResult::Handled;
    }
    ParsedSamplingFlags flags;
    if (!parseFlagPairs(tokens, 3, flags, err)) {
      logErr(cmd + ": " + err);
      return CommandResult::Handled;
    }
    ds::ISRequest req{};
    req.type = (cmd == "submit") ? ds::RequestType::SUBMIT : ds::RequestType::CONTINUE;
    req.request_id = state.nextRequestId();
    req.slot_id = slot;
    req.tokens = toks;
    req.gen = toGenerationParams(flags);
    if (!sched.push_request(req)) {
      logErr("push_request returned false (queue full)");
      return CommandResult::Handled;
    }
    std::ostringstream os;
    os << "pushed " << toString(req.type) << " req=" << req.request_id
       << " slot=" << slot << " ntokens=" << toks.size()
       << " max=" << flags.max_new_tokens
       << " temp=" << flags.temperature
       << (flags.ignore_eos ? " ignore_eos" : "")
       << (flags.spec_decode ? " spec" : "");
    state.logEvent(os.str());
    return CommandResult::Handled;
  }

  return CommandResult::Unknown;
}

// ---------------------------------------------------------------------------
// FTXUI layout
// ---------------------------------------------------------------------------

std::string formatElapsed(std::chrono::steady_clock::time_point start,
                          std::chrono::steady_clock::time_point t) {
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t - start).count();
  std::ostringstream os;
  os << "[+" << (ms / 1000) << "." << (ms % 1000) / 100 << (ms % 100) / 10 << (ms % 10) << "]";
  return os.str();
}

ftxui::Element renderSlots(const AppState::Snapshot& snap,
                           ds::DecodeScheduler& sched,
                           uint32_t max_users) {
  using namespace ftxui;
  std::vector<Element> rows;
  rows.push_back(hbox({
      text("#")          | size(WIDTH, EQUAL, 4)  | bold,
      text("State")      | size(WIDTH, EQUAL, 10) | bold,
      text("InFlight")   | size(WIDTH, EQUAL, 9)  | bold,
      text("Tok")        | size(WIDTH, EQUAL, 6)  | bold,
      text("Pos")        | size(WIDTH, EQUAL, 7)  | bold,
      text("Gen")        | size(WIDTH, EQUAL, 5)  | bold,
      text("Max")        | size(WIDTH, EQUAL, 6)  | bold,
      text("EvP")        | size(WIDTH, EQUAL, 5)  | bold,
      text("StP")        | size(WIDTH, EQUAL, 5)  | bold,
      text("Spec a/r")   | size(WIDTH, EQUAL, 10) | bold,
  }));
  rows.push_back(separator());

  for (uint32_t i = 0; i < max_users; ++i) {
    auto user_state = sched.get_user_state(i);
    auto in_flight = sched.get_in_flight_count(i);
    bool active = user_state != ds::UserState::INACTIVE || in_flight > 0;
    auto slot_it = snap.slots.find(i);
    if (!active && slot_it == snap.slots.end()) continue;

    auto color_for = [](ds::UserState s) -> Color {
      switch (s) {
        case ds::UserState::DECODE:   return Color::Green;
        case ds::UserState::PREFILL:  return Color::Yellow;
        case ds::UserState::COMPLETE: return Color::Blue;
        case ds::UserState::INACTIVE: return Color::GrayDark;
      }
      return Color::Default;
    };
    std::ostringstream specs;
    specs << sched.get_spec_accepts(i) << "/" << sched.get_spec_rejects(i);
    rows.push_back(hbox({
        text(std::to_string(i))                          | size(WIDTH, EQUAL, 4),
        text(toString(user_state)) | color(color_for(user_state)) | size(WIDTH, EQUAL, 10),
        text(std::to_string(in_flight))                  | size(WIDTH, EQUAL, 9),
        text(std::to_string(sched.get_tokens_generated(i))) | size(WIDTH, EQUAL, 6),
        text(std::to_string(sched.get_current_position(i))) | size(WIDTH, EQUAL, 7),
        text(std::to_string(sched.get_generation(i)))    | size(WIDTH, EQUAL, 5),
        text(std::to_string(sched.get_max_new_tokens(i))) | size(WIDTH, EQUAL, 6),
        text(sched.get_evict_pending(i) ? "Y" : "-")     | size(WIDTH, EQUAL, 5),
        text(sched.get_stop_pending(i) ? "Y" : "-")      | size(WIDTH, EQUAL, 5),
        text(specs.str())                                | size(WIDTH, EQUAL, 10),
    }));
  }
  if (rows.size() == 2) {
    rows.push_back(text("  (no active slots)") | dim);
  }
  return window(text(" Slots "), vbox(std::move(rows)));
}

ftxui::Element renderRecentTokens(const AppState::Snapshot& snap) {
  using namespace ftxui;
  std::vector<Element> rows;
  for (const auto& [slot_id, view] : snap.slots) {
    std::ostringstream os;
    os << "slot " << slot_id << ": ";
    bool first = true;
    for (auto t : view.recent_tokens) {
      if (!first) os << " ";
      os << t;
      first = false;
    }
    if (view.last_complete) os << "  [FINAL]";
    rows.push_back(text(os.str()));
  }
  if (rows.empty()) rows.push_back(text("  (no tokens yet)") | dim);
  return window(text(" Recent tokens "), vbox(std::move(rows)));
}

ftxui::Element renderEvents(const AppState::Snapshot& snap,
                            std::chrono::steady_clock::time_point start,
                            int height) {
  using namespace ftxui;
  std::vector<Element> rows;
  size_t shown = static_cast<size_t>(std::max(1, height));
  size_t skip = snap.events.size() > shown ? snap.events.size() - shown : 0;
  for (size_t i = skip; i < snap.events.size(); ++i) {
    const auto& e = snap.events[i];
    rows.push_back(hbox({
        text(formatElapsed(start, e.t)) | color(Color::GrayDark),
        text(" "),
        text(e.text),
    }));
  }
  if (rows.empty()) rows.push_back(text("  (no events)") | dim);
  return window(text(" Event log "), vbox(std::move(rows)));
}

ftxui::Element renderHeader(const CliArgs& a, ds::DecodeScheduler& sched,
                            uint32_t next_request_id) {
  using namespace ftxui;
  std::string backend_desc;
  if (a.backend == "socket") {
    backend_desc = "socket(h2d=" + a.h2d_socket_id + " d2h=" + a.d2h_socket_id + ")";
  } else {
    backend_desc = "sim(stages=" + std::to_string(a.sim_num_stages) +
                   " us=" + std::to_string(a.sim_stage_duration_us) + ")";
  }
  return hbox({
      text(" Decode Scheduler TUI ") | bold | inverted,
      text("  backend=" + backend_desc),
      filler(),
      text(sched.is_running() ? "RUNNING " : "STOPPED ")
          | color(sched.is_running() ? Color::Green : Color::Red),
      text(" next_req_id=" + std::to_string(next_request_id)),
      text("  prefill_q=" + std::to_string(sched.get_prefill_queue_size())),
      text(" decode_staging=" + std::to_string(sched.get_decode_staging_size())),
  });
}

void runTui(const CliArgs& args, ds::DecodeScheduler& sched, AppState& state) {
  using namespace ftxui;
  auto screen = ScreenInteractive::Fullscreen();

  std::string input_text;
  auto input_opt = InputOption();
  // FTXUI v5.0.0 defaults Input to multiline=true, which makes Enter insert a
  // '\n' and (in some build paths) deliver "cmd\n" to on_enter. We want a
  // single-line command box where Enter submits.
  input_opt.multiline = false;
  input_opt.on_enter = [&] {
    auto res = handleCommand(input_text, sched, state);
    input_text.clear();
    if (res == CommandResult::Unknown) {
      state.logEvent("error: unknown command (type 'help')");
    }
    if (res == CommandResult::Quit) {
      screen.Exit();
    }
  };
  auto input = Input(&input_text, "type a command, 'help' for usage", input_opt);

  // Periodic redraw so live scheduler state (in-flight counts, prefill queue
  // size) updates even when the user isn't typing.
  std::atomic<bool> ticker_running{true};
  std::thread ticker([&] {
    while (ticker_running.load()) {
      std::this_thread::sleep_for(kUiTick);
      screen.PostEvent(Event::Custom);
    }
  });

  auto layout = Container::Vertical({input});

  auto renderer = Renderer(layout, [&] {
    auto snap = state.snapshot();
    auto term = Terminal::Size();
    int log_height = std::max(6, term.dimy - 22);
    return vbox({
        renderHeader(args, sched, snap.next_request_id),
        separator(),
        renderSlots(snap, sched, args.max_users),
        renderRecentTokens(snap),
        renderEvents(snap, state.start(), log_height),
        hbox({text(" > ") | bold, input->Render()}) | border,
        text(" alloc | submit S T,T [max=N temp=F ..] | continue S T,T | stop S | evict S | dump | help | quit")
            | color(Color::GrayDark),
    });
  });

  // Esc exits. (Ctrl-C is handled by ScreenInteractive directly.)
  auto with_global_keys = CatchEvent(renderer, [&](Event e) {
    if (e == Event::Escape) {
      screen.Exit();
      return true;
    }
    return false;
  });

  screen.Loop(with_global_keys);
  ticker_running.store(false);
  ticker.join();
}

}  // namespace

int main(int argc, char** argv) {
  auto args_opt = parseCli(argc, argv);
  if (!args_opt) return 1;
  const auto& args = *args_opt;

  ds::SchedulerParams params;
  params.max_users = args.max_users;
  params.max_seq_len = args.max_seq_len;
  params.eos_token = args.eos_token;

  std::cerr << "decode_scheduler_tui: constructing DecodeScheduler (backend="
            << args.backend << ", max_users=" << args.max_users << ")...\n";

  ds::DecodeScheduler sched(makePipelineConfig(args), params);
  sched.start();
  std::cerr << "decode_scheduler_tui: scheduler started; entering TUI...\n";

  AppState state(std::chrono::steady_clock::now());
  state.logEvent("scheduler started (backend=" + args.backend + ")");

  Poller poller(sched, state);
  poller.start();

  runTui(args, sched, state);

  std::cerr << "decode_scheduler_tui: stopping...\n";
  poller.stop();
  sched.stop();
  return 0;
}
