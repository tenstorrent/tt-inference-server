// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/stream_event_formatter.hpp"

#include <chrono>
#include <utility>

#include "api/sse_stream_writer.hpp"
#include "domain/chat_completion_response.hpp"
#include "domain/responses_response.hpp"

namespace tt::api {

namespace {

std::string jsonToCompactString(const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, value);
}

Json::Value buildMessageItem(const std::string& itemId, const std::string& text,
                             const std::string& status) {
  Json::Value item;
  item["id"] = itemId;
  item["type"] = "message";
  item["role"] = "assistant";
  item["status"] = status;

  Json::Value content(Json::arrayValue);
  Json::Value textPart;
  textPart["type"] = "output_text";
  textPart["text"] = text;
  textPart["annotations"] = Json::Value(Json::arrayValue);
  content.append(std::move(textPart));
  item["content"] = std::move(content);
  return item;
}

}  // namespace

// -- ChatCompletionEventFormatter --------------------------------------------

std::string ChatCompletionEventFormatter::formatInitialEvents(
    const StreamParams& params,
    const std::optional<domain::CompletionUsage>& initialUsage) {
  auto initialChunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
      params.completionId, params.model, params.created, initialUsage);
  return initialChunk.toSSE();
}

std::string ChatCompletionEventFormatter::formatTokenEvents(
    const StreamParams& params, const domain::LLMStreamChunk& chunk,
    const std::optional<domain::CompletionUsage>& usage, int /*currentTokens*/,
    const std::string& /*accumulatedText*/) {
  auto streamChunk = domain::ChatCompletionStreamChunk::makeContentChunk(
      params.completionId, params.model, params.created, chunk.choices[0],
      usage);
  return streamChunk.toSSE();
}

std::string ChatCompletionEventFormatter::formatFinalEvents(
    const StreamParams& params, const domain::CompletionUsage& usage,
    const std::string& /*accumulatedText*/,
    const std::optional<std::string>& /*finishReason*/, bool includeUsage) {
  std::string out;
  if (includeUsage) {
    out += domain::ChatCompletionStreamChunk::makeUsageChunk(
               params.completionId, params.model, params.created, usage)
               .toSSE();
  }
  out += "data: [DONE]\n\n";
  return out;
}

// -- ResponsesEventFormatter -------------------------------------------------

ResponsesEventFormatter::ResponsesEventFormatter(
    std::shared_ptr<domain::ResponsesRequest> request,
    tt::runners::llm_engine::SamplingParams samplingParams)
    : request_(std::move(request)),
      sampling_params_(std::move(samplingParams)) {}

std::string ResponsesEventFormatter::formatEvent(const std::string& eventName,
                                                 const Json::Value& payload) {
  Json::Value withType = payload;
  withType["type"] = eventName;
  withType["sequence_number"] = sequence_number_++;

  std::string body = jsonToCompactString(withType);
  std::string out;
  out.reserve(body.size() + eventName.size() + 16);
  out.append("event: ").append(eventName).append("\n");
  out.append("data: ").append(body).append("\n\n");
  return out;
}

std::string ResponsesEventFormatter::buildResponseObjectJson(
    int64_t createdAt, const std::string& status, const Json::Value& output,
    const std::optional<domain::CompletionUsage>& usage) const {
  std::optional<domain::ResponseUsage> respUsage;
  if (usage.has_value()) {
    domain::ResponseUsage u;
    u.input_tokens = usage->prompt_tokens;
    u.output_tokens = usage->completion_tokens;
    u.total_tokens = usage->total_tokens;
    respUsage = std::move(u);
  }

  auto resp = domain::ResponsesResponse::fromRequest(
      request_->task_id, *request_, sampling_params_,
      request_->model.value_or("default"), createdAt, output, status,
      std::move(respUsage));
  return jsonToCompactString(resp.toOpenaiJson());
}

std::string ResponsesEventFormatter::formatInitialEvents(
    const StreamParams& /*params*/,
    const std::optional<domain::CompletionUsage>& /*initialUsage*/) {
  const int64_t createdAt = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  Json::Value emptyOutput(Json::arrayValue);
  auto responseJsonStr = buildResponseObjectJson(createdAt, "in_progress",
                                                 emptyOutput, std::nullopt);
  Json::Value responseObject;
  Json::CharReaderBuilder rb;
  std::string errs;
  const auto* begin = responseJsonStr.data();
  std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
  reader->parse(begin, begin + responseJsonStr.size(), &responseObject, &errs);

  std::string out;

  {
    Json::Value payload;
    payload["response"] = responseObject;
    out += formatEvent("response.created", payload);
  }
  {
    Json::Value payload;
    payload["response"] = responseObject;
    out += formatEvent("response.in_progress", payload);
  }
  {
    Json::Value payload;
    payload["output_index"] = kOutputIndex;
    payload["item"] = buildMessageItem(kItemId, "", "in_progress");
    out += formatEvent("response.output_item.added", payload);
  }
  {
    Json::Value payload;
    payload["item_id"] = kItemId;
    payload["output_index"] = kOutputIndex;
    payload["content_index"] = kContentIndex;
    Json::Value part;
    part["type"] = "output_text";
    part["text"] = "";
    part["annotations"] = Json::Value(Json::arrayValue);
    payload["part"] = std::move(part);
    out += formatEvent("response.content_part.added", payload);
  }
  return out;
}

std::string ResponsesEventFormatter::formatTokenEvents(
    const StreamParams& /*params*/, const domain::LLMStreamChunk& chunk,
    const std::optional<domain::CompletionUsage>& /*usage*/,
    int /*currentTokens*/, const std::string& /*accumulatedText*/) {
  if (chunk.choices.empty()) return {};
  const auto& choice = chunk.choices[0];
  if (choice.text.empty()) return {};

  Json::Value payload;
  payload["item_id"] = kItemId;
  payload["output_index"] = kOutputIndex;
  payload["content_index"] = kContentIndex;
  payload["delta"] = choice.text;
  return formatEvent("response.output_text.delta", payload);
}

std::string ResponsesEventFormatter::formatFinalEvents(
    const StreamParams& /*params*/, const domain::CompletionUsage& usage,
    const std::string& accumulatedText,
    const std::optional<std::string>& finishReason, bool /*includeUsage*/) {
  const bool isIncomplete = finishReason.value_or("stop") == "length";
  const std::string terminalStatus = isIncomplete ? "incomplete" : "completed";
  const std::string terminalEvent =
      isIncomplete ? "response.incomplete" : "response.completed";

  const int64_t createdAt = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  std::string out;

  {
    Json::Value payload;
    payload["item_id"] = kItemId;
    payload["output_index"] = kOutputIndex;
    payload["content_index"] = kContentIndex;
    payload["text"] = accumulatedText;
    out += formatEvent("response.output_text.done", payload);
  }
  {
    Json::Value payload;
    payload["item_id"] = kItemId;
    payload["output_index"] = kOutputIndex;
    payload["content_index"] = kContentIndex;
    Json::Value part;
    part["type"] = "output_text";
    part["text"] = accumulatedText;
    part["annotations"] = Json::Value(Json::arrayValue);
    payload["part"] = std::move(part);
    out += formatEvent("response.content_part.done", payload);
  }
  {
    Json::Value payload;
    payload["output_index"] = kOutputIndex;
    payload["item"] = buildMessageItem(kItemId, accumulatedText, "completed");
    out += formatEvent("response.output_item.done", payload);
  }

  Json::Value finalOutput(Json::arrayValue);
  finalOutput.append(buildMessageItem(kItemId, accumulatedText, "completed"));
  auto responseJsonStr =
      buildResponseObjectJson(createdAt, terminalStatus, finalOutput, usage);
  Json::Value responseObject;
  Json::CharReaderBuilder rb;
  std::string errs;
  const auto* begin = responseJsonStr.data();
  std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
  reader->parse(begin, begin + responseJsonStr.size(), &responseObject, &errs);

  Json::Value payload;
  payload["response"] = std::move(responseObject);
  out += formatEvent(terminalEvent, payload);
  return out;
}

}  // namespace tt::api
