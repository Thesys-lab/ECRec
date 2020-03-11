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

#include "ps-plus/client/local_client.h"
#include "ps-plus/client/partitioner/dense.h"
#include "ps-plus/client/partitioner/logic.h"
#include "ps-plus/client/partitioner/sparse.h"
#include "ps-plus/client/partitioner/broadcast.h"
#include "ps-plus/client/partitioner/index.h"
#include "ps-plus/client/partitioner/hash.h"

#include <iostream>

namespace ps {
namespace client {

#define RETURN_ASYNC(STATUS) do { cb(STATUS); return; } while (0)

#define CHECK_ASYNC(STATUS) do {                                                                                \
    Status st_ = STATUS;                                                                                        \
    if (!st_.IsOk()) {                                                                                          \
        st_.Msg() += "\nCHECKED BY [" #STATUS "] @ FILE[" __FILE__ "] LINE[" + std::to_string(__LINE__) + "]";  \
        RETURN_ASYNC(st_);                                                                                      \
    }                                                                                                           \
} while (0)

void LocalClient::IndexInitializerWithoutParity(const std::string& variable_name,
                                   Initializer* init, 
                                   const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));

  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  std::vector<Data*> inputs = Args(
      info.datatype, TensorShape(dims), (size_t)0, 
      std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IndexVariableInitializer", 
              UdfData(0), 
              UdfData(1), 
              UdfData(2), 
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::IdentityInitializer(const std::string& variable_name, 
                                      const Tensor& init, 
                                      const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  std::vector<Data*> inputs = Args(
      info.datatype, 
      TensorShape(dims), 
      (size_t)0, 
      init);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IdentityIndexVariableInitializer", 
              UdfData(0), 
              UdfData(1), 
              UdfData(2), 
              UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::HashInitializer(const std::string& variable_name, 
                                  Initializer* init,
                                  const LocalClient::Callback& cb) {
  VariableInfo info;
  CHECK_ASYNC(local_server_->GetVariableInfo(variable_name, &info));
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  size_t k = info.shape[0];
  dims[0] = k + 10 * sqrt(k) + 10;
  std::string extra_info;
  for (auto& arg : info.args) {
    extra_info += arg.first + "=" + arg.second + "&";
  }
  if (!extra_info.empty()) { extra_info.pop_back(); }
  std::vector<Data*> inputs = Args(
      info.datatype, 
      TensorShape(dims), 
      extra_info,
      std::unique_ptr<Initializer>(init));
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("HashVariableInitializer", UdfData(0), UdfData(1), UdfData(2), UdfData(3));
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, '^' + variable_name, inputs, outputs, realcb);
}

void LocalClient::IsInitialized(const std::string& variable_name, 
                                bool* inited, 
                                const LocalClient::Callback& cb) {
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("IsInitialized");
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

  Process(udf, '^' + variable_name, {}, outputs, realcb);
}

void LocalClient::DensePull(const std::string& variable_name, 
                            Tensor* result, 
                            const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(false);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildDenseSlice", UdfData(0));
  UdfData udf_chain("SliceToTensor", udf);
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

    WrapperData<std::vector<Tensor>>* output_ptr = 
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor vector"));
      return;
    }

    if (output_ptr->Internal().size() != 1) {
      cb(Status::ArgumentError("Output[0] size should be 1"));
      return;
    }

    *result = output_ptr->Internal()[0];
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::DensePush(const std::string& variable_name, 
                            const std::string& updater, 
                            const std::vector<Data*>& data, 
                            const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(true);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {UdfData("BuildDenseSlice", UdfData(0))};
  for (size_t i = 1; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::SparsePullWithoutParity(const std::string& variable_name,
                             const Tensor& ids, 
                             Tensor* result, 
                             const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, false);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildSparseSlice", UdfData(0), UdfData(1));
  UdfData udf_chain("SliceToTensor", udf);
  Callback realcb = [this, cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on SparsePull"));
      return;
    }

    WrapperData<std::vector<Tensor>>* output_ptr = 
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor vector"));
      return;
    }

    if (output_ptr->Internal().size() != 1) {
      cb(Status::ArgumentError("Output[0] size should be 1"));
      return;
    }

    *result = output_ptr->Internal()[0];
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::SparsePushWithoutParity(const std::string& variable_name,
                             const Tensor& ids, 
                             const std::string& updater, 
                             const std::vector<Data*>& data, 
                             const LocalClient::Callback& cb) {
  std::vector<Data*> inputs = Args(ids, true);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {
    UdfData("BuildSparseSlice", UdfData(0), UdfData(1))
  };
  for (size_t i = 2; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::HashPull(const std::string& variable_name, 
                           const Tensor& ids,
                           const float& save_ratio,
                           Tensor* result, 
                           const LocalClient::Callback& cb) {
  std::vector<Tensor> ids_vec = {ids};
  std::vector<std::string> name_vec = {variable_name};
  std::vector<float> save_ratio_vec = {save_ratio};  
  std::vector<Data*> inputs = Args(ids_vec, name_vec, save_ratio_vec, false, true);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4));
  UdfData udf_chain("SliceToTensor", udf);
  Callback realcb = [this, cb, result, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on HashPull"));
      return;
    }

    WrapperData<std::vector<Tensor>>* output_ptr = 
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor vector"));
      return;
    }

    if (output_ptr->Internal().size() != 1) {
      cb(Status::ArgumentError("Output[0] size should be 1"));
      return;
    }

    *result = output_ptr->Internal()[0];
    cb(Status::Ok());
  };

  Process(udf_chain, variable_name, inputs, outputs, realcb);
}

void LocalClient::MergedHashPull(const std::vector<std::string>& var_names, 
                                 const std::vector<Tensor>& ids,
                                 const std::vector<float>& save_ratios,
                                 std::vector<Tensor>* result, 
                                 const Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, false, true);
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4));
  UdfData udf_chain("SliceToTensor", udf);
  Callback realcb = [this, cb, result, outputs, var_names](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on HashPull"));
      return;
    }

    WrapperData<std::vector<Tensor>>* output_ptr = 
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor vector"));
      return;
    }

    if (output_ptr->Internal().size() != var_names.size()) {
      cb(Status::ArgumentError("Output[0] Size Should be the Same with Variable Number"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, "^hash_variable", inputs, outputs, realcb);
}

void LocalClient::HashPush(const std::string& variable_name, 
                           const Tensor& ids,
                           const float& save_ratio,
                           const bool& insertable,
                           const std::string& updater,
                           const std::vector<Data*>& data, 
                           const LocalClient::Callback& cb) {
  std::vector<Tensor> ids_vec = {ids};
  std::vector<std::string> name_vec = {variable_name};
  std::vector<float> save_ratio_vec = {save_ratio};  
  std::vector<Data*> inputs = Args(ids_vec, name_vec, save_ratio_vec, true, insertable);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {
    UdfData("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4))
  };
  for (size_t i = 5; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, variable_name, inputs, outputs, realcb);
}

void LocalClient::MergedHashPush(const std::vector<std::string>& var_names,
                                 const std::vector<Tensor>& ids,
                                 const std::vector<float>& save_ratios,
                                 const std::string& updater,
                                 const std::vector<Data*>& data,
                                 const Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, true, false);
  inputs.insert(inputs.end(), data.begin(), data.end());
  std::vector<std::unique_ptr<Data>>* outputs = 
    new std::vector<std::unique_ptr<Data>>;
  std::vector<UdfData> updater_inputs = {
    UdfData("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4))
  };
  for (size_t i = 5; i < inputs.size(); i++) {
    updater_inputs.push_back(UdfData(i));
  }

  UdfData udf(updater, updater_inputs);
  Callback realcb = [cb, outputs](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    cb(st);
  };

  Process(udf, "^hash_variable", inputs, outputs, realcb);
}

void LocalClient::MergedHashStatis(const std::vector<std::string>& var_names,
                                   const std::vector<Tensor>& ids,
                                   const std::vector<float>& save_ratios,
                                   const std::vector<Tensor>& clicks,
                                   const Tensor& global_step,
                                   const Tensor& statis_decay,
                                   const Tensor& statis_decay_period,
                                   const std::string& statis_type,
                                   std::vector<Tensor>* result,
                                   const Callback& cb) {
  std::vector<Data*> inputs = Args(ids, var_names, save_ratios, false, true);
  std::vector<std::unique_ptr<Data>>* outputs =
    new std::vector<std::unique_ptr<Data>>;
  UdfData udf("BuildHashSlice", UdfData(0), UdfData(1), UdfData(2), UdfData(3), UdfData(4));
  UdfData udf_chain("SliceToTensor", udf);
  Callback realcb = [this, cb, result, outputs, var_names](const Status& st) {
    std::unique_ptr<std::vector<std::unique_ptr<Data>>> deleter(outputs);
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    if (outputs->size() != 1) {
      cb(Status::ArgumentError("Output Size Should be 1 on MergedHashStatis"));
      return;
    }

    WrapperData<std::vector<Tensor>>* output_ptr =
      dynamic_cast<WrapperData<std::vector<Tensor>>*>((*outputs)[0].get());
    if (output_ptr == nullptr) {
      cb(Status::ArgumentError("Output[0] should be tensor vector"));
      return;
    }

    if (output_ptr->Internal().size() != var_names.size()) {
      cb(Status::ArgumentError("Output[0] Size Should be the Same with Variable Number"));
      return;
    }

    *result = output_ptr->Internal();
    cb(Status::Ok());
  };

  Process(udf_chain, "^hash_variable", inputs, outputs, realcb);
}

void LocalClient::Process(const UdfChain& udf, 
                          const std::string& var_name,
                          const std::vector<Data*>& datas,
                          std::vector<std::unique_ptr<Data>>* results,
                          const LocalClient::Callback& cb) {
  std::vector<Data*> outputs;
  Status st = local_server_->Process(udf.hash(), var_name, datas, &outputs);
  if (st.Code() == Status::kUdfNotRegistered) {
    st = local_server_->RegisterUdfChain(udf.BuildChainRegister());
    if (!st.IsOk()) {
      cb(st);
      return;
    }

    st = local_server_->Process(udf.hash(), var_name, datas, &outputs);
  }

  results->reserve(outputs.size());
  for (Data* data: outputs) results->push_back(std::unique_ptr<Data>(data));
  cb(st);
  for (Data* data: datas) delete data;
}

// REDUNDANCY: add sparse pull/push with parity
void LocalClient::IndexInitializer(const std::string& variable_name,
                              Initializer* init,
                              const Callback& cb) {
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) == VARIABLE_NAMES_WITH_PARITY.end()) {
    IndexInitializerWithoutParity(variable_name, init, cb);
    return ;
  }

  // first only initialize the variables without values
  IndexInitializerWithoutParity(variable_name, new initializer::NoneInitializer(), cb);
  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(variable_name, &info));
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  auto num_cols = info.shape[1];

  // iterate through each batch
  for (auto batch_start_index = 0; batch_start_index < info.shape[0]; batch_start_index += INIT_BATCH_NUM_CHUNKS) {
    auto num_rows_in_batch = std::min(INIT_BATCH_NUM_CHUNKS * PARITY_K, int(info.shape[0] - batch_start_index));
    // Create tensor of ids corresponding to batch
    std::vector<size_t> ids_shape_vec;
    std::vector<size_t> values_shape_vec;
    ids_shape_vec.push_back(num_rows_in_batch);
    values_shape_vec.push_back(num_rows_in_batch);
    values_shape_vec.push_back(num_rows_in_batch);
    TensorShape ids_shape(ids_shape_vec);
    TensorShape values_shape(values_shape_vec);

    // init tensor for client_ids
    Tensor *client_ids = new Tensor(types::kInt64, ids_shape, new ps::initializer::NoneInitializer());
    for (auto i = 0; i < num_rows_in_batch; i ++) {
      *(client_ids->Raw<size_t >(i)) = i + batch_start_index;
    }

    // init tensor for init values
    Tensor* init_values = new Tensor(info.datatype, values_shape, init);

    auto empty_cb = [](const Status& st) {};
    // Pull the corresponding values

    // Calculate parities
    Tensor *server_ids = new Tensor;
    Tensor *server_values = new Tensor;
    pu.MapClientToServerTensorWithParity(*client_ids, *init_values, server_ids, server_values, true);
    SparsePushWithoutParity(variable_name, *server_ids, "AssignUpdater", Args(server_values), empty_cb);
  }
}

void LocalClient::SparsePull(const std::string& variable_name,
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
  SparsePullWithoutParity(variable_name, new_ids, result, cb);
}

void LocalClient::SparsePush(const std::string& variable_name,
                        const Tensor& ids,
                        const std::string& updater,
                        const std::vector<Data*>& data,
                        const Callback& cb) {
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) == VARIABLE_NAMES_WITH_PARITY.end()) {
    SparsePushWithoutParity(variable_name, ids, updater, data, cb);
    return ;
  }
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
    SparsePushWithoutParity(variable_name, new_ids, updater, Args(new_data_tensor), cb);
  } else if (updater == "MomentumUpdater") {
    // case 2: handle momentum updater.
    // todo other updaters might also follow the same linear pattern.
    WrapperData<std::vector<Tensor>>* data_vec_ptr =
            dynamic_cast<WrapperData<std::vector<Tensor>>*>(data[0]);
    if (data_vec_ptr == nullptr) {
      cb(Status::ArgumentError("data[0] should be tensor"));
      return;
    }
    auto data_vec = data_vec_ptr->Internal();
    std::vector<Tensor> new_data_vec(data_vec.size());
    std::vector<Data*> new_data(data);
    for (auto i = 0; i < data_vec.size(); i ++) {
      pu.MapClientToServerTensorWithParity(ids, data_vec[i], &new_ids, &(new_data_vec[i]));
    }
    // replace the first entry (grad vec) in data with the new gradient vectors, keeping other components the same
    new_data[0] = Args(new_data_vec)[0];
    SparsePushWithoutParity(variable_name, new_ids, updater, new_data, cb);
  }
  else {
    // case 2: other operators. need to obtain diff first
    // todo fix this total trash
    // todo not really sure if we need this
  }
}


} //namespace client
} //namespace ps

