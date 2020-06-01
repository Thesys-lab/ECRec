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
#include <cstdlib>
#include <ps-plus/ps-plus/common/initializer/constant_initializer.h>

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

void Client::IndexInitializerWithoutParity(const std::string& variable_name,
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

void Client::SparsePullWithoutParity(const std::string& variable_name,
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

void Client::SparsePushWithoutParity(const std::string& variable_name,
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
void Client::IndexInitializer(const std::string& variable_name,
                                   Initializer* init,
                                   const Callback& cb) {
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) == VARIABLE_NAMES_WITH_PARITY.end()) {
    IndexInitializerWithoutParity(variable_name, init, cb);
    return ;
  }

  // first only initialize the variables without values
  bool init_done = false;
  auto init_cb = [&init_done](const Status& st) {
      init_done = true;
  };
  // TODO: fix this part
  IndexInitializerWithoutParity(variable_name, init,  cb);
  return ;

  IndexInitializerWithoutParity(variable_name, new initializer::ConstantInitializer(0),  init_cb);

  while (!init_done) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }


    VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  // initialize an array recording status for each batch
  size_t batch_count = (size_t)info.shape[0] / INIT_BATCH_NUM_CHUNKS;
  if (info.shape[0] % INIT_BATCH_NUM_CHUNKS != 0) batch_count += 1;
  std::vector<bool> each_batch_ready;
  for (auto i = 0; i < batch_count; i ++) each_batch_ready.push_back(false);

  // iterate through each batch
  auto batch_num = 0;
  for (auto batch_start_index = 0; batch_start_index < info.shape[0]; batch_start_index += INIT_BATCH_NUM_CHUNKS) {
    auto num_rows_in_batch = std::min(INIT_BATCH_NUM_CHUNKS * PARITY_K, size_t(info.shape[0] - batch_start_index));

    // Create tensor of ids corresponding to batch
    TensorShape ids_shape(std::vector<size_t>({num_rows_in_batch}));
    TensorShape values_shape(std::vector<size_t>({num_rows_in_batch, (size_t)info.shape[1]}));

    // init tensor for client_ids
    Tensor *client_ids = new Tensor(types::kInt64, ids_shape, new ps::initializer::NoneInitializer());
    for (auto i = 0; i < num_rows_in_batch; i ++) {
      *(client_ids->Raw<size_t >(i)) = i + batch_start_index;
    }

    // init tensor for init values
    Tensor* init_values = new Tensor(info.datatype, values_shape, init);

    // Pull the corresponding values
    auto reduce_count_cb = [&each_batch_ready, batch_num, client_ids, variable_name, this] (const Status& st) mutable {
      each_batch_ready[batch_num] = true;
    };
    // Calculate parities
    Tensor *server_ids = new Tensor;
    Tensor server_values;
    pu.MapClientToServerTensorWithParity(*client_ids, *init_values, server_ids, &server_values, true);
    std::vector<Tensor> server_values_vector = {server_values};
    SparsePushWithoutParity(variable_name, *server_ids, "AssignUpdater", Args(server_values_vector), reduce_count_cb);
    // test
    batch_num += 1;
  }

  auto ready = false;

  while (!ready) {
    ready = true;
    for (auto i = 0; i < batch_count; i ++) {
      if (!each_batch_ready[i]) {
        ready = false;
        break;
      }
    }
    std::this_thread::sleep_for (std::chrono::seconds(1));
  }


  Tensor client_ids(types::kInt64, TensorShape(std::vector<size_t>({1})), new initializer::NoneInitializer());
  *(client_ids.Raw<size_t>()) = 0;
  cb(Status::Ok());
}

void Client::SparsePull(const std::string& variable_name,
                             const Tensor& ids,
                             Tensor* result,
                             const Callback& cb) {
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) == VARIABLE_NAMES_WITH_PARITY.end()) {
    SparsePullWithoutParity(variable_name, ids, result, cb);
    return ;
  }
  Tensor new_ids;
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
  pu.MapClientToServerTensor(ids, &new_ids);

  if (SIMULATED_FAILED_SERVERS.empty()){
    SparsePullWithoutParity(variable_name, new_ids, result, cb);
    return ;
  }

  // failed server simulation
  std::vector<size_t> ids_on_failed;
  std::vector<size_t> ids_not_on_failed;
  for (auto i = 0; i < new_ids.Shape().NumElements(); i ++) {
    auto server_id = *(new_ids.Raw<size_t>(i));
    auto server = pu.FindServer(server_id);
    if (SIMULATED_FAILED_SERVERS.find(server) == SIMULATED_FAILED_SERVERS.end()) {
      ids_not_on_failed.push_back(server_id);
    } else {
      ids_on_failed.push_back(server_id);
    }
  }
  Tensor friend_ids;
  Tensor pull_result;
  TensorShape ids_on_failed_shape({ids_on_failed.size()});
  Tensor ids_on_failed_tensor(new_ids.Type(), ids_on_failed_shape, new initializer::NoneInitializer());

  memcpy(ids_on_failed_tensor.Raw<void>(), ids_on_failed.data(), SizeOfType(ids_on_failed_tensor.Type()) * ids_on_failed.size());

  TensorShape sent_to_servers_shape({ids_not_on_failed.size() + friend_ids.Shape().NumElements()});
  Tensor sent_to_servers(new_ids.Type(), sent_to_servers_shape, new initializer::NoneInitializer());
  memcpy(sent_to_servers.Raw<void>(),
         friend_ids.Raw<void>(),
         SizeOfType(friend_ids.Type()) * friend_ids.Shape().NumElements());
  memcpy(sent_to_servers.Raw<void>(friend_ids.Shape().NumElements()),
         ids_not_on_failed.data(),
         SizeOfType(sent_to_servers.Type()) * ids_not_on_failed.size());


  pu.FindFriendIds(ids_on_failed_tensor, &friend_ids, SIMULATED_FAILED_SERVERS);

  std::condition_variable cv;
  std::mutex mtx;
  bool ready;

  auto result_cb = [&] (const Status& st) mutable {
      go(&mtx, &cv, &ready);
  };
  SparsePullWithoutParity(variable_name, sent_to_servers, &pull_result, result_cb);
  wait(&mtx, &cv, &ready);

  Tensor recovered_values(pull_result.Type(), ids_on_failed_tensor.Shape(), new initializer::NoneInitializer());
  pu.RecoverServerValues(ids_on_failed_tensor, friend_ids, pull_result, &recovered_values);
  // create a map from id to value to reorder output.
  std::unordered_map<size_t, float> map;
  // first add the recovered result
  for (size_t i = 0; i < recovered_values.Shape().NumElements(); i ++) {
    map[ids_on_failed[i]] = *(recovered_values.Raw<float>(i));
  }
  // then add the other values not on any failed server
  // skip the first friend_ids size entries of pull results
  for (size_t i = friend_ids.Shape().NumElements(); i < pull_result.Shape().NumElements(); i ++) {
    auto ind = i - friend_ids.Shape().NumElements();
    map[ids_not_on_failed[ind]] = *(pull_result.Raw<float>(i));
  }

  // recover target result
  TensorShape result_shape({ids.Shape().NumElements()});
  *result = Tensor(pull_result.Type(), result_shape, new initializer::NoneInitializer());
  for (size_t i = 0; i < ids.Shape().NumElements(); i ++) {
    *(result->Raw<float>(i)) = map[*(new_ids.Raw<size_t>(i))];
  }
  cb(Status::Ok());
}

void Client::SparsePush(const std::string& variable_name,
                             const Tensor& ids,
                             const std::string& updater,
                             const std::vector<Data*>& data,
                             const Callback& cb) {
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) == VARIABLE_NAMES_WITH_PARITY.end()) {
    SparsePushWithoutParity(variable_name, ids, updater, data, cb);
    return ;
  }
  Tensor new_data_tensor;
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  if (updater == "AssignAddUpdater" || updater == "AssignSubUpdater") {
    // case 1: assign add/sub, we can directly update parity with one round of communication
    // todo: need to redo this

  } else if (updater == "MomentumUpdater") {
    // case 2: handle momentum updater.
    // todo other updaters might also follow the same linear pattern.
    if (!SERVER_PARITY_UPDATE) {
      std::vector<Tensor> parity_ids;
      pu.MapClientToParityIds(ids, parity_ids);
      WrapperData<std::vector<Tensor>>* data_vec_ptr =
              dynamic_cast<WrapperData<std::vector<Tensor>>*>(data[0]);
      if (data_vec_ptr == nullptr) {
        cb(Status::ArgumentError("data[0] should be tensor"));
        return;
      }
      auto data_vec = data_vec_ptr->Internal();
      SparsePushWithoutParity(variable_name, ids, updater, data, cb);
      for (const auto& parity_ids_tensor : parity_ids) {
        std::vector<ps::Tensor> new_data_vec;
        for (size_t i = 0; i < data_vec.size(); i ++) {
          new_data_vec.push_back(data_vec[i].Clone());
        }
        auto lr_vec = dynamic_cast<WrapperData<std::vector<double>>*>(data[1])->Internal();
        auto momentum_vec = dynamic_cast<WrapperData<std::vector<double>>*>(data[2])->Internal();
        auto use_nesterov_vec = dynamic_cast<WrapperData<std::vector<bool>>*>(data[3])->Internal();

        auto new_data = Args(new_data_vec, lr_vec, momentum_vec, use_nesterov_vec);
        auto empty_cb = [] (const Status& st){};
        SparsePushWithoutParity(variable_name, parity_ids_tensor, updater, new_data, empty_cb);
      }
    } else {
      std::vector<Data*> new_data(data);
      //std::vector<VariableInfo> infos({info});
      //new_data.push_back(Args(infos)[0]);
      SparsePushWithoutParity(variable_name, ids, "MomentumServerUpdater", new_data, cb);
    }
  }
  else {
    // case 2: other operators. need to obtain diff first
    // todo fix this total trash
    // todo not really sure if we need this
  }
}

} //namespace client
} //namespace ps
