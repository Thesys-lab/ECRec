/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ps-plus/common/base_parity_utils.h"
#include "ps-plus/client/partitioner/index.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/types.h"

namespace ps {
namespace client {
namespace partitioner {

Status IndexDataType::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (VARIABLE_NAMES_WITH_PARITY.find(info->name) != VARIABLE_NAMES_WITH_PARITY.end()){
    auto tmp = *info;
    BaseParityScheme pu(&tmp, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
    pu.AdaptVariableInfoToServerSpace(&tmp);
    info = &tmp;
  }
  dst->clear();
  for (size_t i = 0; i < info->parts.size(); i++) {
    Data* d = new WrapperData<DataType>(info->datatype);
    ctx->AddDeleter(d);
    dst->push_back(d);
  }
  return Status::Ok();
}

Status IndexShape::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  VariableInfo tmp = *info;
  if (VARIABLE_NAMES_WITH_PARITY.find(info->name) != VARIABLE_NAMES_WITH_PARITY.end()){
    BaseParityScheme pu(&tmp, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
    pu.AdaptVariableInfoToServerSpace(&tmp);
    info = &tmp;
  }
  std::vector<size_t> dims(info->shape.begin(), info->shape.end());
  dst->clear();
  for (size_t i = 0; i < info->parts.size(); i++) {
    if (!dims.empty()) {
      dims[0] = info->parts[i].size;
    }
    Data* d = new WrapperData<TensorShape>(dims);
    ctx->AddDeleter(d);
    dst->push_back(d);
  }
  return Status::Ok();
}

Status IndexOffset::Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  VariableInfo* info = ctx->GetVariableInfo();
  if (VARIABLE_NAMES_WITH_PARITY.find(info->name) != VARIABLE_NAMES_WITH_PARITY.end()){
    auto tmp = *info;
    BaseParityScheme pu(&tmp, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
    pu.AdaptVariableInfoToServerSpace(&tmp);
    info = &tmp;
  }
  size_t offset = 0;
  dst->clear();
  for (size_t i = 0; i < info->parts.size(); i++) {
    Data* d = new WrapperData<size_t>(offset);
    ctx->AddDeleter(d);
    dst->push_back(d);
    offset += info->parts[i].size;
  }
  return Status::Ok();
}

}
}
}

