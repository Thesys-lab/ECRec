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
#include <ps-plus/common/file_system.h>

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

  // record sparse update statistics
  WrapperData<std::vector<Tensor>> *data_vec_ptr =
            dynamic_cast<WrapperData<std::vector<Tensor>> *>(data[0]);
  std::vector<Tensor> data_vec = data_vec_ptr->Internal();
  size_t ids_n = ids.Shape().NumElements();
  size_t data_vec_n = data_vec[0].Shape().NumElements();
  size_t num_floats = data_vec.size() * ids.Shape().NumElements();
  std::string log = "(Tianyu) ids_n=" + std::to_string(ids_n) 
          + ", data_vec_n=" + std::to_string(data_vec_n)
          + ", num_floats=" + std::to_string(num_floats) + "\n";
  std::unique_ptr<ps::FileSystem::WriteStream> s;
  FileSystem::OpenWriteStreamAny("/xdl_data/sparse_log", &s, true);
  s->WriteStr(log);

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
  IndexInitializerWithoutParity(variable_name, init, cb);
  return ;
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
  pu.MapClientToServerIds(ids, &new_ids);

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
  QuickMemcpy(ids_on_failed_tensor.Raw<void>(), ids_on_failed.data(), SizeOfType(ids_on_failed_tensor.Type()) * ids_on_failed.size());
  pu.FindFriendIds(ids_on_failed_tensor, &friend_ids, SIMULATED_FAILED_SERVERS);

  TensorShape sent_to_servers_shape({ids_not_on_failed.size() + friend_ids.Shape().NumElements()});
  Tensor sent_to_servers(new_ids.Type(), sent_to_servers_shape, new initializer::NoneInitializer());
  QuickMemcpy(sent_to_servers.Raw<void>(),
         friend_ids.Raw<void>(),
         SizeOfType(friend_ids.Type()) * friend_ids.Shape().NumElements());
  QuickMemcpy(sent_to_servers.Raw<size_t>(friend_ids.Shape().NumElements()),
         ids_not_on_failed.data(),
         SizeOfType(sent_to_servers.Type()) * ids_not_on_failed.size());

  std::condition_variable cv;
  std::mutex mtx;
  bool ready = false;

  auto result_cb = [&] (const Status& st) mutable {
      go(&mtx, &cv, &ready);
  };
  SparsePullWithoutParity(variable_name, sent_to_servers, &pull_result, result_cb);

  wait(&mtx, &cv, &ready);

  auto width = pull_result.Shape().Dims()[1];
  std::vector<size_t> recovered_values_shape_vec({ids_on_failed.size(), width});
  Tensor recovered_values(pull_result.Type(), TensorShape(recovered_values_shape_vec), new initializer::NoneInitializer());
  pu.RecoverServerValues(ids_on_failed_tensor, friend_ids, pull_result, &recovered_values);

  // recover target result
  std::vector<size_t> result_shape_vec({ids.Shape().NumElements(), width});
  *result = Tensor(types::kFloat, TensorShape(result_shape_vec), new initializer::NoneInitializer());
  size_t failed_index = 0;
  for (size_t i = 0; i < ids.Shape().NumElements(); i ++) {
    auto server_id = *(new_ids.Raw<size_t>(i));
    if (failed_index != ids_on_failed.size() && server_id == ids_on_failed[failed_index]) {
      QuickMemcpy(result->Raw<float>(i), recovered_values.Raw<float>(failed_index), sizeof(float) * width);
      failed_index ++;
    } else {
      auto non_failed_index = i - failed_index;
      QuickMemcpy(result->Raw<float>(i), pull_result.Raw<float>(non_failed_index), sizeof(float) * width);
    }
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
    Tensor new_ids;
    auto empty_cb = [&] (const Status& st){};
    pu.MapClientToServerIds(ids, &new_ids);

    auto original_beg = data.begin();
    auto original_end = data.begin() + 4;
    std::vector<Data*> original_data(original_beg, original_end);

    if (SERVER_PARITY_UPDATE) {
      SparsePushWithoutParity(variable_name, new_ids, "MomentumServerUpdater", original_data, cb);
    } else {
      SparsePushWithoutParity(variable_name, new_ids, updater, original_data, cb);
      std::vector<Tensor> parity_ids;
      pu.MapClientToParityIds(ids, parity_ids);
      if (data.size() == 0) {
        cb(Status::ArgumentError("data length should be nonzero"));
        return;
      }

      for (size_t i = 0; i < parity_ids.size(); i ++) {
        const auto& parity_ids_tensor = parity_ids[i];
        auto par_beg = data.begin() + i * 4 + 4;
        auto par_end = data.begin() + i * 4 + 8;
        std::vector<Data*> new_data(par_beg, par_end);
        SparsePushWithoutParity(variable_name, parity_ids_tensor, updater, new_data, empty_cb);
      }

    }
  } else if (updater == "AdagradUpdater") {
    // case 2: handle momentum updater.
    // todo other updaters might also follow the same linear pattern.
    Tensor new_ids;
    auto empty_cb = [&](const Status &st) {};
    pu.MapClientToServerIds(ids, &new_ids);
    std::vector<Tensor> parity_ids;
    std::vector<Tensor> chunk_indices;
    pu.MapClientToParityIds(ids, parity_ids);
    WrapperData<std::vector<Tensor>> *data_vec_ptr =
            dynamic_cast<WrapperData<std::vector<Tensor>> *>(data[0]);
    if (data_vec_ptr == nullptr) {
      cb(Status::ArgumentError("data[0] should be tensor"));
      return;
    }
    auto data_vec = data_vec_ptr->Internal();
    auto lr_vec = dynamic_cast<WrapperData<std::vector<double>> *>(data[1])->Internal();
    auto momentum_vec = dynamic_cast<WrapperData<std::vector<double>> *>(data[2])->Internal();
    auto use_nesterov_vec = dynamic_cast<WrapperData<std::vector<bool>> *>(data[3])->Internal();

    SparsePushWithoutParity(variable_name, new_ids, "AdagradUpdaterLowPrec", data, cb);
    for (const auto &parity_ids_tensor : parity_ids) {

      std::vector<ps::Tensor> new_data_vec;
      for (size_t i = 0; i < data_vec.size(); i++) {
        auto data_tens = data_vec[i].Clone();
        size_t num_elements = ids.Shape().NumElements();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
              auto chunk_index = *(ids.Raw<size_t >(i)) % PARITY_K;
              int* bitwise_data_ptr = (int*)(data_tens.Raw<float>(i));
              (*bitwise_data_ptr) &= 0xfffffffc;
              (*bitwise_data_ptr) |= chunk_index;
            }
        });
        new_data_vec.push_back(data_tens);
      }

      auto new_data = Args(new_data_vec, lr_vec, momentum_vec, use_nesterov_vec);

      std::thread t1(&Client::SparsePushWithoutParity, this, variable_name, parity_ids_tensor,
                     "AdagradUpdater", new_data, empty_cb);
      t1.detach();
    }
  } else {
    // case 2: other operators. need to obtain diff first
    // todo not really sure if we need this
  }
}

} //namespace client
} //namespace ps
