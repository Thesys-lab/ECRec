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

#include "ps-plus/client/client.h"
#include "ps-plus/client/partitioner/dense.h"
#include "ps-plus/client/partitioner/logic.h"
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/client/partitioner/broadcast.h"
#include "ps-plus/client/partitioner/merged_broadcast.h"
#include "ps-plus/client/partitioner/index.h"
#include "ps-plus/client/partitioner/hash.h"
#include "ps-plus/client/partitioner/merged_hash.h"

#include <iostream>
#include <cstdlib>

#define RETURN_ASYNC(STATUS) do { cb(STATUS); return; } while (0)

#define CHECK_ASYNC(STATUS) do {                                                                                \
    Status st_ = STATUS;                                                                                        \
    if (!st_.IsOk()) {                                                                                          \
        st_.Msg() += "\nCHECKED BY [" #STATUS "] @ FILE[" __FILE__ "] LINE[" + std::to_string(__LINE__) + "]";  \
        RETURN_ASYNC(st_);                                                                                      \
    }                                                                                                           \
} while (0)

namespace ps {
namespace client {

void Client::IndexInitializer(const std::string& variable_name,
                              Initializer* init,
                              const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(0, 0, 0, std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs = new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::IndexDataType,
    new partitioner::IndexShape,
    new partitioner::IndexOffset,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {};
  UdfData udf("IndexVariableInitializer",
              UdfData(0),
              UdfData(1),
              UdfData(2),
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs,
          splitter, combiner, outputs, realcb);
}

void Client::IdentityInitializer(const std::string& variable_name,
                                 const Tensor& init,
                                 const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(0, 0, 0, init);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::IndexDataType,
    new partitioner::IndexShape,
    new partitioner::IndexOffset,
    new partitioner::Dense
  };
  std::vector<Partitioner*> combiner = {};
  UdfData udf("IdentityIndexVariableInitializer",
              UdfData(0),
              UdfData(1),
              UdfData(2),
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::HashInitializer(const std::string& variable_name,
                             Initializer* init,
                             const Client::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  std::string extra_info;
  for (auto& arg : info.args) {
    extra_info += arg.first + "=" + arg.second + "&";
  }
  if (!extra_info.empty()) { extra_info.pop_back(); }
  std::vector<Data*> inputs = Args(0, 0, extra_info, std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::HashDataType,
    new partitioner::HashShape,
    new partitioner::Broadcast,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {};
  UdfData udf("HashVariableInitializer", UdfData(0), UdfData(1), UdfData(2), UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::IsInitialized(const std::string& variable_name,
                           bool* inited,
                           const Callback& cb) {
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IsInitialized");
  std::vector<Partitioner*> combiner = {new partitioner::Logic};
  Callback realcb = [cb, outputs, inited](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    WrapperData<bool>* output_ptr =
      dynamic_cast<WrapperData<bool>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *inited = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf, '^' + variable_name, {}, {}, combiner, outputs, realcb);
}

void Client::DensePull(const std::string& variable_name,
                       Tensor* result,
                       const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(false);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = { new partitioner::Broadcast };
  std::vector<Partitioner*> combiner = { new partitioner::Dense };
  UdfData udf("BuildDenseSlice", UdfData(0));
  UdfData udf_chain("TransSlice", udf);
  Callback realcb = [this, cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on DensePull"));
      return;
    }

    WrapperData<Tensor>* output_ptr =
      dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };
  Process(udf_chain, variable_name, inputs, splitter,
          combiner, outputs, realcb);
  char* vp_var = std::getenv("vp_method");
  char* meta_var = std::getenv("meta_dir");
  std::string vp_string;
  if (vp_var != NULL) { vp_string = vp_var; }
  if (vp_string == "anneal" && meta_var != NULL) {
    CHECK_ASYNC(UpdateVariableVisitInfo(variable_name, -1));
  }
}

void Client::DensePush(const std::string& variable_name,
                       const std::string& updater,
                       const std::vector<Data*>& data,
                       const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(true);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = { new partitioner::Broadcast };
  std::vector<Partitioner*> combiner = {};
  std::vector<UdfData> next_udf_inputs = {
    UdfData("BuildDenseSlice", UdfData(0))
  };

  size_t start_index = 1;
  if (sync_mode_ &&
      updater != "AssignUpdater" &&
      updater != "AssignAddUpdater" &&
      updater != "AssignSubUpdater" &&
      updater != "MovingAverageUpdater") {
    inputs.push_back(Args(token_)[0]);
    inputs.push_back(Args(worker_count_)[0]);
    next_udf_inputs.push_back(UdfData(1));
    next_udf_inputs.push_back(UdfData(2));
    next_udf_inputs.push_back(UdfData(3));
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::Dense);
    UdfData aggregate("AggregateSlice", next_udf_inputs);
    next_udf_inputs = {aggregate};
    start_index = 4;
  }

  inputs.insert(inputs.end(), data.begin(), data.end());
  for (size_t i = start_index; i < inputs.size(); i++) {
    if (dynamic_cast<WrapperData<Tensor>*>(inputs[i]) != nullptr
      || dynamic_cast<WrapperData<std::vector<Tensor>>*>(inputs[i]) != nullptr) {
      splitter.push_back(new partitioner::Dense);
    } else {
      splitter.push_back(new partitioner::Broadcast);
    }

    next_udf_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, next_udf_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::SparsePull(const std::string& variable_name,
                        const Tensor& ids,
                        Tensor* result,
                        const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, false);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::SparseId,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {
    new partitioner::SparseData
  };
  UdfData udf("BuildSparseSlice", UdfData(0), UdfData(1));
  UdfData udf_chain("TransSlice", udf);
  Callback realcb = [cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on SparsePull"));
      return;
    }

    WrapperData<Tensor>* output_ptr =
      dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };
  Process(udf_chain, variable_name, inputs, splitter,
          combiner, outputs, realcb);
  char* vp_var = std::getenv("vp_method");
  char* meta_var = std::getenv("meta_dir");
  std::string vp_string;
  if (vp_var != NULL) { vp_string = vp_var; }
  if (vp_string == "anneal" && meta_var != NULL) {
    CHECK_ASYNC(UpdateVariableVisitInfo(variable_name, ids.Shape()[0]));
  }
}

void Client::SparsePush(const std::string& variable_name,
                        const Tensor& ids,
                        const std::string& updater,
                        const std::vector<Data*>& data,
                        const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, true);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::SparseId,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {};
  std::vector<UdfData> next_udf_inputs = {
    UdfData("BuildSparseSlice", UdfData(0), UdfData(1))
  };

  size_t start_index = 2;
  if (sync_mode_ &&
      updater != "AssignUpdater" &&
      updater != "AssignAddUpdater" &&
      updater != "AssignSubUpdater" &&
      updater != "MovingAverageUpdater") {
    inputs.push_back(Args(token_)[0]);
    inputs.push_back(Args(worker_count_)[0]);
    next_udf_inputs.push_back(UdfData(2));
    next_udf_inputs.push_back(UdfData(3));
    next_udf_inputs.push_back(UdfData(4));
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::SparseData);
    UdfData aggregate("AggregateSlice", next_udf_inputs);
    next_udf_inputs = {aggregate};
    start_index = 5;
  }

  inputs.insert(inputs.end(), data.begin(), data.end());
  for (size_t i = start_index; i < inputs.size(); i++) {
    if (dynamic_cast<WrapperData<Tensor>*>(inputs[i]) != nullptr
      || dynamic_cast<WrapperData<std::vector<Tensor>>*>(inputs[i]) != nullptr) {
      splitter.push_back(new partitioner::SparseData);
    } else {
      splitter.push_back(new partitioner::Broadcast);
    }

    next_udf_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, next_udf_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };
  Process(udf, variable_name, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::HashPull(const std::string& variable_name,
                      const Tensor& ids,
                      const float& save_ratio,
                      Tensor* result,
                      const Client::Callback& cb) {
  std::vector<Tensor> ids_vec = {ids};
  std::vector<std::string> name_vec = {variable_name};
  std::vector<float> save_ratio_vec = {save_ratio};
  std::vector<Data*> inputs = Args(ids_vec, name_vec, save_ratio_vec, false, true);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::HashId,
    new partitioner::Broadcast,
    new partitioner::Broadcast,
    new partitioner::Broadcast,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {
    new partitioner::HashData
  };
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4));
  UdfData udf_chain("TransSlice", udf);
  Callback realcb = [cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on HashPull"));
      return;
    }

    WrapperData<Tensor>* output_ptr =
      dynamic_cast<WrapperData<Tensor>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, splitter,
          combiner, outputs, realcb);
  char* vp_var = std::getenv("vp_method");
  char* meta_var = std::getenv("meta_dir");
  std::string vp_string;
  if (vp_var != NULL) { vp_string = vp_var; }
  if (vp_string == "anneal" && meta_var != NULL) {
    CHECK_ASYNC(UpdateVariableVisitInfo(variable_name, ids.Shape()[0]));
  }
}

void Client::MergedHashPull(const std::vector<std::string>& var_names,
                            const std::vector<Tensor>& ids,
                            const std::vector<float>& save_ratios,
                            std::vector<Tensor>* result,
                            const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, false, true);
  std::vector<std::vector<std::unique_ptr<Data>>>* outputs =
    new std::vector<std::vector<std::unique_ptr<Data>>>;
  std::vector<MergedPartitioner*> splitter = {
    new partitioner::MergedHashId,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast
  };
  std::vector<MergedPartitioner*> combiner = {
    new partitioner::MergedHashData
  };
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4));
  Callback realcb = [cb, result, outputs, var_names](const Status& st) {
    std::unique_ptr<std::vector<std::vector<std::unique_ptr<Data>>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on MergedHashPull"));
      return;
    }

    std::vector<std::unique_ptr<Data>>& output_vec = (*outputs)[0];
    if (output_vec.size() != var_names.size()) {
      cb(Status::ArgumentError("Output[0] Size Should be the Same with Variable Number"));
      return;
    }
    for (auto& output : output_vec) {
      WrapperData<Tensor>* output_ptr = dynamic_cast<WrapperData<Tensor>*>(output.get());
      if (output_ptr == nullptr) {
        cb(Status::ArgumentError("Output[0] should be tensor vector"));
        return;
      }
      (*result).push_back(output_ptr->Internal());
    }
    cb(Status::Ok());
  };

  Process(udf, var_names, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::HashPush(const std::string& variable_name,
                      const Tensor& ids,
                      const float& save_ratio,
                      const bool& insertable,
                      const std::string& updater,
                      const std::vector<Data*>& data,
                      const Client::Callback& cb) {
  std::vector<Tensor> ids_vec = {ids};
  std::vector<std::string> name_vec = {variable_name};
  std::vector<float> save_ratio_vec = {save_ratio};
  std::vector<Data*> inputs = Args(ids_vec, name_vec, save_ratio_vec, true, insertable);
  size_t start_index = 5;
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  std::vector<Partitioner*> splitter = {
    new partitioner::HashId,
    new partitioner::Broadcast,
    new partitioner::Broadcast,
    new partitioner::Broadcast,
    new partitioner::Broadcast
  };
  std::vector<Partitioner*> combiner = {};
  std::vector<UdfData> next_udf_inputs = {
    UdfData("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4))
  };

  if (sync_mode_ &&
      updater != "AssignUpdater" &&
      updater != "AssignAddUpdater" &&
      updater != "AssignSubUpdater" &&
      updater != "MovingAverageUpdater") {
    inputs.push_back(Args(token_)[0]);
    inputs.push_back(Args(worker_count_)[0]);
    next_udf_inputs.push_back(UdfData(5));
    next_udf_inputs.push_back(UdfData(6));
    next_udf_inputs.push_back(UdfData(7));
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::Broadcast);
    splitter.push_back(new partitioner::HashData);
    UdfData aggregate("AggregateSlice", next_udf_inputs);
    next_udf_inputs = {aggregate};
    start_index = 8;
  }

  inputs.insert(inputs.end(), data.begin(), data.end());
  for (size_t i = start_index; i < inputs.size(); i++) {
    if (dynamic_cast<WrapperData<Tensor>*>(inputs[i]) != nullptr
      || dynamic_cast<WrapperData<std::vector<Tensor>>*>(inputs[i]) != nullptr) {
      splitter.push_back(new partitioner::HashData);
    } else {
      splitter.push_back(new partitioner::Broadcast);
    }
    next_udf_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, next_udf_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::MergedHashPush(const std::vector<std::string>& var_names,
                            const std::vector<Tensor>& ids,
                            const std::vector<float>& save_ratios,
                            const std::string& updater,
                            const std::vector<Data*>& data,
                            const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, true, false);
  size_t start_index = 5;
  std::vector<std::vector<std::unique_ptr<Data>>>* outputs =
    new std::vector<std::vector<std::unique_ptr<Data>>>;
  std::vector<MergedPartitioner*> splitter = {
    new partitioner::MergedHashId,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast
  };
  std::vector<MergedPartitioner*> combiner = {};
  std::vector<UdfData> next_udf_inputs = {
    UdfData("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4))
  };

  if (sync_mode_ &&
      updater != "AssignUpdater" &&
      updater != "AssignAddUpdater" &&
      updater != "AssignSubUpdater" &&
      updater != "MovingAverageUpdater") {
    inputs.push_back(Args(token_)[0]);
    inputs.push_back(Args(worker_count_)[0]);
    next_udf_inputs.push_back(UdfData(5));
    next_udf_inputs.push_back(UdfData(6));
    next_udf_inputs.push_back(UdfData(7));
    splitter.push_back(new partitioner::MergedBroadcast);
    splitter.push_back(new partitioner::MergedBroadcast);
    splitter.push_back(new partitioner::MergedHashData);
    UdfData aggregate("AggregateSlice", next_udf_inputs);
    next_udf_inputs = {aggregate};
    start_index = 8;
  }

  inputs.insert(inputs.end(), data.begin(), data.end());
  for (size_t i = start_index; i < inputs.size(); i++) {
    if (dynamic_cast<WrapperData<Tensor>*>(inputs[i]) != nullptr
      || dynamic_cast<WrapperData<std::vector<Tensor>>*>(inputs[i]) != nullptr) {
      splitter.push_back(new partitioner::MergedHashData);
    } else {
      splitter.push_back(new partitioner::MergedBroadcast);
    }
    next_udf_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, next_udf_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::vector<std::unique_ptr<Data>>>> deleter(outputs);
    cb(st);
  };

  Process(udf, var_names, inputs, splitter,
          combiner, outputs, realcb);
}

void Client::MergedHashStatis(const std::vector<std::string>& var_names,
                              const std::vector<Tensor>& ids,
                              const std::vector<float>& save_ratios,
                              const std::vector<Tensor>& clicks,
                              const Tensor& global_step,
                              const Tensor& statis_decay,
                              const Tensor& statis_decay_period,
                              const std::string& statis_type,
                              std::vector<Tensor>* result,
                              const Client::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, clicks, global_step, statis_decay, statis_decay_period, statis_type, false, true);
  std::vector<std::vector<std::unique_ptr<Data>>>* outputs =
    new std::vector<std::vector<std::unique_ptr<Data>>>;
  std::vector<MergedPartitioner*> splitter = {
    new partitioner::MergedHashId,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedHashData,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast,
    new partitioner::MergedBroadcast
  };
  std::vector<MergedPartitioner*> combiner = {
    new partitioner::MergedHashData
  };
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(8), UdfData(9));
  UdfData udf_chain("StatisSlice", udf, UdfData(3), UdfData(4), UdfData(5), UdfData(6), UdfData(7));
  Callback realcb = [cb, result, outputs, var_names](const Status& st) {
    std::unique_ptr<std::vector<std::vector<std::unique_ptr<Data>>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on MergedHashStatis"));
      return;
    }

    std::vector<std::unique_ptr<Data>>& output_vec = (*outputs)[0];
    if (output_vec.size() != var_names.size()) {
      cb(Status::ArgumentError("Output[0] Size Should be the Same with Variable Number"));
      return;
    }
    for (auto& output : output_vec) {
      WrapperData<Tensor>* output_ptr = dynamic_cast<WrapperData<Tensor>*>(output.get());
      if (output_ptr == nullptr) {
        cb(Status::ArgumentError("Output[0] should be tensor vector"));
        return;
      }
      result->push_back(output_ptr->Internal());
    }
    cb(Status::Ok());
  };

  Process(udf_chain, var_names, inputs, splitter,
          combiner, outputs, realcb);
}


// REDUNDANCY: add sparse pull/push with parity
void Client::IndexInitializerWithParity(const std::string& variable_name,
                              Initializer* init,
                              const Client::Callback& cb) {
  IndexInitializer(variable_name, init, cb);
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  // calculate number of elements in sparse table
  auto total_size = 1;
  for (auto dim : info.shape) total_size *= dim;

  // iterate through each batch
  for (auto batch_start_index = 0; batch_start_index < total_size; batch_start_index += INIT_BATCH_NUM_CHUNKS) {
    auto num_elements_in_batch = std::min(INIT_BATCH_NUM_CHUNKS * PARITY_K, total_size - batch_start_index);
    // Create tensor of ids corresponding to batch
    std::vector<size_t> shape_vec;
    shape_vec.push_back(num_elements_in_batch);
    TensorShape new_shape(shape_vec);
    Tensor *client_ids = new Tensor(types::kInt64, new_shape, new ps::initializer::NoneInitializer());
    for (auto i = 0; i < num_elements_in_batch; i ++) {
      *(client_ids->Raw<size_t >(i)) = i + batch_start_index;
    }

    Tensor* init_values = new Tensor;

    auto empty_cb = [](const Status& st) {};
    // Pull the corresponding values
    SparsePullWithParity(variable_name, *client_ids, init_values, empty_cb);
    // todo: is this async?

    // Calculate parities
    Tensor *parity_ids = new Tensor;
    Tensor *parity_diff = new Tensor;
    pu.MapClientToServerTensorWithParity(*client_ids, *init_values, parity_ids, parity_diff);
    SparsePush(variable_name, *parity_ids, "AssignUpdater", Args(parity_diff), empty_cb);
  }
}

void Client::SparsePullWithParity(const std::string& variable_name,
                          const Tensor& ids,
                          Tensor* result,
                          const Callback& cb) {
  Tensor new_ids;
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
  pu.MapClientToServerTensor(ids, &new_ids);
  SparsePull(variable_name, new_ids, result, cb);
}

void Client::SparsePushWithParity(const std::string& variable_name,
                          const Tensor& ids,
                          const std::string& updater,
                          const std::vector<Data*>& data,
                          const Callback& cb) {
  Tensor new_ids;
  Tensor new_data_tensor;
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  if (updater == "AssignAddUpdater" || updater == "AssignSubUpdater") {
    // case 1: assign add/sub, we can directly update parity with one round of communication
    WrapperData<Tensor>* data_ptr =
            dynamic_cast<WrapperData<Tensor>*>(data[0]);
    if (data_ptr == nullptr) {
      cb(Status::ArgumentError("data[0] should be tensor"));
      return;
    }
    pu.MapClientToServerTensorWithParity(ids, data_ptr->Internal(), &new_ids, &new_data_tensor);
    SparsePush(variable_name, new_ids, updater, Args(new_data_tensor), cb);
  } else if (updater == "MomentumUpdater") {
    // case 2: handle momentum updater.
    // todo other updaters might also follow the same linear pattern.
    WrapperData<Tensor>* data_ptr =
            dynamic_cast<WrapperData<Tensor>*>(data[0]);
    if (data_ptr == nullptr) {
      cb(Status::ArgumentError("data[0] should be tensor"));
      return;
    }
    std::vector<Data*> new_data(data);
    pu.MapClientToServerTensorWithParity(ids, data_ptr->Internal(), &new_ids, &new_data_tensor);
    // replace the first entry (grad vec) in data with the new gradient vectors, keeping other components the same
    new_data[0] = Args(new_data_tensor)[0];
    SparsePush(variable_name, new_ids, updater, new_data, cb);
  }
  else {
    // case 2: other operators. need to obtain diff first
    // todo fix this total trash
    // todo not really sure if we need this
    /*
    Tensor before_result;
    Tensor after_result;
    auto empty_cb = [ctx, done](const ps::Status& st) {};
    SparsePull(variable_name, ids, &before_result, empty_cb);

    SparsePush(variable_name, ids, )

    SparsePull(variable_name, ids, &after_result, empty_cb);
    */
  }
}

} //namespace client
} //namespace ps
