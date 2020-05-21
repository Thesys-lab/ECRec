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
#include "ps-plus/ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/ps-plus/common/initializer/constant_initializer.h"

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
  if (VARIABLE_NAMES_WITH_PARITY.find(variable_name) != VARIABLE_NAMES_WITH_PARITY.end()) {
    BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
    pu.AdaptVariableInfoToServerSpace(&info);
  }
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
  bool init_done = false;
  auto init_cb = [&init_done](const Status& st) {
      init_done = true;
  };
  IndexInitializerWithoutParity(variable_name, new initializer::NoneInitializer(), init_cb);

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
  cb(Status::Ok());
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
  auto result_cb = [&] (const Status& st) mutable {
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
      cb(st);
  };
  SparsePullWithoutParity(variable_name, sent_to_servers, &pull_result, result_cb);
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
    // todo: need to redo this

  } else if (updater == "MomentumUpdater") {
    // case 2: handle momentum updater.
    // todo other updaters might also follow the same linear pattern.
    pu.MapClientToServerIds(ids, &new_ids);

    WrapperData<std::vector<Tensor>>* data_vec_ptr =
            dynamic_cast<WrapperData<std::vector<Tensor>>*>(data[0]);
    if (data_vec_ptr == nullptr) {
      cb(Status::ArgumentError("data[0] should be tensor"));
      return;
    }
    auto data_vec = data_vec_ptr->Internal();
    for (size_t i = 0; i < data_vec.size(); i ++) {
      data_vec[i].Multiply<float>(1 + PARITY_N - PARITY_K);
    }

    std::vector<Data*> new_data(data);
    new_data[0] = Args(data_vec)[0];

    SparsePushWithoutParity(variable_name, new_ids, updater, new_data, cb);
  }
  else {
    // case 2: other operators. need to obtain diff first
    // todo fix this total trash
    // todo not really sure if we need this
  }
}

void LocalClient::PrintFirstChunk(const Tensor &ids, const std::string& variable_name) {
  VariableInfo info;
  GetVariableInfo(variable_name, &info);
  BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);

  auto original_id = *(ids.Raw<size_t>(0));

  auto friend_id = original_id - 1;
  if (original_id % 2 == 0) friend_id = original_id + 1;

  size_t original_server_id;
  size_t friend_server_id;
  size_t parity_ids[PARITY_N - PARITY_K];
  pu.MapClientIdToServerId(original_id, &original_server_id, parity_ids);
  pu.MapClientIdToServerId(friend_id, &friend_server_id, parity_ids);

  std::vector<size_t> shape({4});
  Tensor length_4_tensor = Tensor(ids.Type(), TensorShape(shape), new initializer::NoneInitializer());
  *(length_4_tensor.Raw<size_t >(0)) = original_server_id;
  *(length_4_tensor.Raw<size_t >(1)) =friend_server_id;
  *(length_4_tensor.Raw<size_t >(2)) = parity_ids[0];
  *(length_4_tensor.Raw<size_t >(3)) = parity_ids[1];
  Tensor *test_result = new Tensor;
  auto empty_cb = [&test_result, variable_name, original_server_id, friend_server_id, parity_ids, original_id](const Status& st) {
  printf("printing one group for var %s and ids %lu %lu %lu %lu original_id %lu\n", variable_name.c_str(), original_server_id, friend_server_id, parity_ids[0], parity_ids[1], original_id);
    for (auto row = 0; row < test_result->Shape().Dims()[0]; row ++) {
      for (auto col = 0; col < test_result->Shape().Dims()[1]; col ++) {
        printf("%f ", *(test_result->Raw<float>(row) + col));
      }
      printf("\n");
    }};
  SparsePullWithoutParity(variable_name, length_4_tensor, test_result, empty_cb);
}

} //namespace client
} //namespace ps

