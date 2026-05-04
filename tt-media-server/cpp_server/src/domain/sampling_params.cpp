#include "domain/sampling_params.hpp"

#include <json/json.h>

namespace tt::domain {

namespace {

template <typename T>
void writeScalar(std::ostream& os, const T& value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template <typename T>
void writeOptional(std::ostream& os, const std::optional<T>& opt) {
  bool hasValue = opt.has_value();
  writeScalar(os, hasValue);
  if (hasValue) {
    writeScalar(os, *opt);
  }
}

template <typename T>
std::optional<T> readOptional(std::istream& is) {
  if (readScalar<bool>(is)) {
    return readScalar<T>(is);
  }
  return std::nullopt;
}

template <typename T>
void writeVector(std::ostream& os, const std::vector<T>& vec) {
  size_t size = vec.size();
  writeScalar(os, size);
  os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
std::vector<T> readVector(std::istream& is) {
  size_t size = readScalar<size_t>(is);
  std::vector<T> vec(size);
  is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
  return vec;
}

void writeString(std::ostream& os, const std::string& str) {
  size_t len = str.size();
  writeScalar(os, len);
  os.write(str.data(), static_cast<std::streamsize>(len));
}

std::string readString(std::istream& is) {
  size_t len = readScalar<size_t>(is);
  std::string str(len, '\0');
  is.read(str.data(), static_cast<std::streamsize>(len));
  return str;
}

void writeJsonValue(std::ostream& os, const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  std::string jsonStr = Json::writeString(writer, value);
  writeString(os, jsonStr);
}

Json::Value readJsonValue(std::istream& is) {
  std::string jsonStr = readString(is);
  Json::CharReaderBuilder reader;
  Json::Value value;
  std::string errors;
  std::istringstream iss(jsonStr);
  if (!Json::parseFromStream(reader, iss, &value, &errors)) {
    throw std::runtime_error("Failed to parse JSON: " + errors);
  }
  return value;
}

}  // anonymous namespace

void SamplingParams::serialize(std::ostream& os) const {
  writeScalar(os, temperature);
  writeOptional(os, max_tokens);
  writeScalar(os, ignore_eos);
  writeOptional(os, top_p);
  writeScalar(os, presence_penalty);
  writeScalar(os, frequency_penalty);
  writeOptional(os, seed);
  writeScalar(os, use_beam_search);
  writeOptional(os, top_k);
  writeOptional(os, min_p);
  writeOptional(os, repetition_penalty);
  writeScalar(os, length_penalty);
  writeVector(os, stop_token_ids);
  writeScalar(os, include_stop_str_in_output);
  writeScalar(os, min_tokens);
  writeScalar(os, skip_special_tokens);
  writeScalar(os, spaces_between_special_tokens);

  bool hasAllowed = allowed_token_ids.has_value();
  writeScalar(os, hasAllowed);
  if (hasAllowed) {
    writeVector(os, *allowed_token_ids);
  }

  writeOptional(os, prompt_logprobs);
  writeOptional(os, truncate_prompt_tokens);
  writeScalar(os, fast_mode);

  writeScalar(os, static_cast<uint8_t>(response_format_type));
  bool hasSchema = json_schema_str.has_value();
  writeScalar(os, hasSchema);
  if (hasSchema) {
    size_t len = json_schema_str->size();
    writeScalar(os, len);
    os.write(json_schema_str->data(), static_cast<std::streamsize>(len));
  }

  // Serialize tool_choice
  bool hasToolChoice = tool_choice.has_value();
  writeScalar(os, hasToolChoice);
  if (hasToolChoice) {
    writeString(os, tool_choice->type);
    bool hasFunction = tool_choice->function.has_value();
    writeScalar(os, hasFunction);
    if (hasFunction) {
      writeString(os, tool_choice->function.value());
    }
  }

  // Serialize tools
  bool hasTools = tools.has_value();
  writeScalar(os, hasTools);
  if (hasTools) {
    size_t toolCount = tools->size();
    writeScalar(os, toolCount);
    for (const auto& tool : *tools) {
      writeString(os, tool.type);
      writeString(os, tool.functionDefinition.name);
      bool hasDescription = tool.functionDefinition.description.has_value();
      writeScalar(os, hasDescription);
      if (hasDescription) {
        writeString(os, tool.functionDefinition.description.value());
      }
      writeJsonValue(os, tool.functionDefinition.parameters);
    }
  }
}

std::unique_ptr<SamplingParams> SamplingParams::deserialize(std::istream& is) {
  auto params = std::make_unique<SamplingParams>();

  params->temperature = readScalar<float>(is);
  params->max_tokens = readOptional<int>(is);
  params->ignore_eos = readScalar<bool>(is);
  params->top_p = readOptional<float>(is);
  params->presence_penalty = readScalar<float>(is);
  params->frequency_penalty = readScalar<float>(is);
  params->seed = readOptional<int>(is);
  params->use_beam_search = readScalar<bool>(is);
  params->top_k = readOptional<int>(is);
  params->min_p = readOptional<float>(is);
  params->repetition_penalty = readOptional<float>(is);
  params->length_penalty = readScalar<float>(is);
  params->stop_token_ids = readVector<int>(is);
  params->include_stop_str_in_output = readScalar<bool>(is);
  params->min_tokens = readScalar<int>(is);
  params->skip_special_tokens = readScalar<bool>(is);
  params->spaces_between_special_tokens = readScalar<bool>(is);

  if (readScalar<bool>(is)) {
    params->allowed_token_ids = readVector<int>(is);
  }

  params->prompt_logprobs = readOptional<int>(is);
  params->truncate_prompt_tokens = readOptional<int>(is);
  params->fast_mode = readScalar<bool>(is);

  params->response_format_type =
      static_cast<ResponseFormatType>(readScalar<uint8_t>(is));
  if (readScalar<bool>(is)) {
    size_t len = readScalar<size_t>(is);
    std::string schema(len, '\0');
    is.read(schema.data(), static_cast<std::streamsize>(len));
    params->json_schema_str = std::move(schema);
  }

  // Deserialize tool_choice
  if (readScalar<bool>(is)) {
    tool_calls::ToolChoice choice;
    choice.type = readString(is);
    if (readScalar<bool>(is)) {
      choice.function = readString(is);
    }
    params->tool_choice = choice;
  }

  // Deserialize tools
  if (readScalar<bool>(is)) {
    size_t toolCount = readScalar<size_t>(is);
    std::vector<tool_calls::Tool> toolList;
    toolList.reserve(toolCount);
    for (size_t i = 0; i < toolCount; ++i) {
      tool_calls::Tool tool;
      tool.type = readString(is);
      tool.functionDefinition.name = readString(is);
      if (readScalar<bool>(is)) {
        tool.functionDefinition.description = readString(is);
      }
      tool.functionDefinition.parameters = readJsonValue(is);
      toolList.push_back(std::move(tool));
    }
    params->tools = std::move(toolList);
  }

  return params;
}

}  // namespace tt::domain