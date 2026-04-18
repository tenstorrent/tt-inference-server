// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <string>
#include <vector>

namespace tt::domain {

struct ModelObject {
  std::string id;
  std::string object = "model";
  std::string owned_by = "tenstorrent";

  Json::Value toJson() const {
    Json::Value json;
    json["id"] = id;
    json["object"] = object;
    json["owned_by"] = owned_by;
    return json;
  }
};

struct ModelsResponse {
  std::string object = "list";
  std::vector<ModelObject> data;

  Json::Value toJson() const {
    Json::Value json;
    json["object"] = object;
    Json::Value dataArray(Json::arrayValue);
    for (const auto& model : data) {
      dataArray.append(model.toJson());
    }
    json["data"] = dataArray;
    return json;
  }
};

}  // namespace tt::domain
