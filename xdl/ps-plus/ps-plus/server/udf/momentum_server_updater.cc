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

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/server/udf/momentum_map_range_updater.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/client/client.h"
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/server/mlc_utils.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;
class MomentumServerUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double>, vector<bool>> {
public:
  virtual Status SimpleRun(
          UdfContext* ctx,
          const vector<Slices>& sslices,
          const vector<Tensor>& grad_tensors,
          const vector<double>& learning_rates,
          const vector<double>& momentums,
          const vector<bool>& use_nesterovs
          ) const {
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != momentums.size() || sslices.size() != use_nesterovs.size()) {
      return Status::ArgumentError("MomentumUpdater: slices and other size not match");
    }

    LOG(INFO) << "Tianyu: sslices.size()=" << sslices.size();

    // wait for update allowed
    while (!MomentumMapRangeUpdater::update_allowed) {
      MomentumMapRangeUpdater::wait_update_allowed();
    }

    MomentumMapRangeUpdater::ongoing_update_count_mtx.lock();
    MomentumMapRangeUpdater::ongoing_udpate_count += 1;
    MomentumMapRangeUpdater::ongoing_update_count_mtx.unlock();

    // access freq stats
    // std::unique_ptr<FileSystem::WriteStream> s_freq_stats;
    // std::string freq_stats_path = "/xdl_data/freq_stats/emb1";
    // PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(freq_stats_path, &s_freq_stats, true));

    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      std::unique_ptr<QRWLocker> locker;
      locker.reset(new QRWLocker(slices.variable->VariableLock(), QRWLocker::kSimpleRead));
      if (!slices.writable) {
        MomentumMapRangeUpdater::ongoing_update_count_mtx.lock();
        MomentumMapRangeUpdater::ongoing_udpate_count -= 1;
        MomentumMapRangeUpdater::ongoing_update_count_mtx.unlock();
        return Status::ArgumentError("slice is not writable");
      }
      double learning_rate = learning_rates[si];
      double momentum = momentums[si];
      bool use_nesterov = use_nesterovs[si];
      const Tensor& grad_tensor = grad_tensors[si];

      // LOG(INFO) << "Tianyu: grad_tensor shape: " << grad_tensor.Shape().ToString();
      
      WrapperData<size_t>* offset = dynamic_cast<WrapperData<size_t>*>(slices.variable->GetSlicer());
      int64_t min_id = offset->Internal();

      auto client = CheckpointUtils::GetTempClient();\
      Tensor* data_tensor = slices.variable->GetData();
      Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accumulation", data_tensor->Type(), []{ return new initializer::ConstantInitializer(0); });
      if (grad_tensor.Type() != data_tensor->Type()) {
        MomentumMapRangeUpdater::ongoing_update_count_mtx.lock();
        MomentumMapRangeUpdater::ongoing_udpate_count -= 1;
        MomentumMapRangeUpdater::ongoing_update_count_mtx.unlock();
        return Status::ArgumentError("grad should has same datatype with variable");
      }

      // LOG(INFO) << "Tianyu: data_tensor shape: " << data_tensor->Shape().ToString();
      // LOG(INFO) << "Tianyu: acc_tensor shape: " << acc_tensor->Shape().ToString();
      
      //Create id tensors
      std::vector<size_t> id_shape_vec({slices.slice_id.size()});
      TensorShape id_shape(id_shape_vec);
      // Tensor ids(types::kInt64, id_shape, new initializer::NoneInitializer());

      //Crete diff tensor

      std::vector<size_t> diff_shape_vec({slices.slice_id.size(), slices.slice_size});
      TensorShape diff_shape(diff_shape_vec);
      // Tensor diff_tens(types::kFloat, diff_shape, new initializer::NoneInitializer());

      bool use_map = (MomentumMapRangeUpdater::map_range_start != MomentumMapRangeUpdater::map_range_end);

      // TODO: this is assuming there is only ONE embedding table
      if (MomentumMapRangeUpdater::temp_map == nullptr) {
        std::vector<size_t> map_shape_vec({data_tensor->Shape().Dims()});
        map_shape_vec[0] = map_shape_vec[0] / RECOVERY_NUM_LOCKS + 1;
        TensorShape map_shape(map_shape_vec);
        MomentumMapRangeUpdater::temp_map = new Tensor(data_tensor->Type(), map_shape, new initializer::NoneInitializer);
        MomentumMapRangeUpdater::acc_temp_map = new Tensor(data_tensor->Type(), map_shape, new initializer::NoneInitializer);
        // set pointer to original tensors
        MomentumMapRangeUpdater::original = data_tensor;
        MomentumMapRangeUpdater::original_acc = acc_tensor;
      }

      // create ckpt data & acc tensors
      Tensor ckpt_data_tensor(types::kFloat, diff_shape, new initializer::NoneInitializer());
      Tensor ckpt_acc_tensor(types::kFloat, diff_shape, new initializer::NoneInitializer());

      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
          // T* diff_ptr = diff_tens.Raw<T>();
          T* ckpt_data_ptr = ckpt_data_tensor.Raw<T>();
          T* ckpt_acc_ptr = ckpt_acc_tensor.Raw<T>();

          for (size_t i = r.begin; i < r.end; i++) {
            int64_t slice = slices.slice_id[i];
            if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
              continue;
            }

            // // log access
            // auto time = std::chrono::system_clock::now();
            // std::time_t now = std::chrono::system_clock::to_time_t(time);
            // s_freq_stats.get()->WriteRaw(now);
            // s_freq_stats.get()->WriteRaw(slice);

            if (use_map && slice >= MomentumMapRangeUpdater::map_range_start && slice < MomentumMapRangeUpdater::map_range_end) {
              float* data = MomentumMapRangeUpdater::temp_map->Raw<float>(slice - MomentumMapRangeUpdater::map_range_start);
              float* acc = MomentumMapRangeUpdater::acc_temp_map->Raw<float>(slice - MomentumMapRangeUpdater::map_range_start);
              float* grad = grad_tensor.Raw<float>(i);
              if (use_nesterov) {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  float diff = *grad * learning_rate + *acc * momentum * learning_rate;
                  *data -= diff;
                  // *diff_ptr = diff;
                  data++; acc++; grad++;
                }
              } else {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  float diff = *acc * learning_rate;
                  *data -= diff;
                  // *diff_ptr = diff;
                  data++; acc++; grad++;          }
              }
            } else {
              // LOG(INFO) << "Tianyu: not using MapRangeUpdater";

              T* data = data_tensor->Raw<T>(slice);
              auto id = slice + min_id;
              T* acc = acc_tensor->Raw<T>(slice);
              T* grad = grad_tensor.Raw<T>(i);
              // *(ids.Raw<size_t>(i)) = id;
              if (use_nesterov) {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  T diff = *grad * learning_rate + *acc * momentum * learning_rate;
                  *data -= diff;
                  // *diff_ptr = diff;

                  // update ckpt tensors
                  *ckpt_data_ptr = *data;
                  *ckpt_acc_ptr = *acc;
                  ckpt_data_ptr++; ckpt_acc_ptr++;

                  // data++; acc++; grad++; diff_ptr++;
                  data++; acc++; grad++;
                }
              } else {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  T diff = *acc * learning_rate;
                  *data -= diff;
                  // *diff_ptr = diff;

                  // update ckpt tensors
                  *ckpt_data_ptr = *data;
                  *ckpt_acc_ptr = *acc;
                  ckpt_data_ptr++; ckpt_acc_ptr++;

                  // data++; acc++; grad++; diff_ptr++;
                  data++; acc++; grad++;
                }
              }
            }
          }
          return Status::Ok();
      }));

      // auto empty_cb = [](const Status &st) {
      // };
      // std::vector<Tensor> diffs;
      // diffs.push_back(diff_tens);

      VariableInfo info;
      client->GetVariableInfo(slices.variable->GetName(), &info);

      bool writeThis = getNext(INTERVAL);
      LOG(INFO) << "Tianyu: writeThis=" << writeThis;
      if (writeThis) {
        // real-time ckpt
        std::unique_ptr<FileSystem::WriteStream> s;
        std::string checkpoint = "/xdl_data/ckpt_test";
        PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(checkpoint + '/' + 
          CheckpointUtils::VariableNameToFileName(info.name, 0), &s, false));

        PS_CHECK_STATUS(CheckpointUtils::SaveTensor(s.get(), ckpt_data_tensor));
        PS_CHECK_STATUS(CheckpointUtils::SaveTensor(s.get(), ckpt_acc_tensor));

      }
      

      // VariableInfoCollection from = {.infos = {info}};
      // CheckpointUtils ckpt(from);
      // CheckpointUtils::VariableStruct vs;
      // std::unique_ptr<Variable> var_ptr;
      // var_ptr.reset(slices.variable);
      // ckpt.VariableToStruct(var_ptr, &vs);
      // std::string ckpt_path = "/mydata/ckpt_test";
      // ckpt.SaveVariable(ckpt_path, info.name, 0, &vs);
      // LOG(INFO) << "Tianyu: ckpt tensors saved!";


      // parity update
      // BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
      // std::vector<Tensor> parity_ids;
      // pu.MapServerToParityIds(ids, parity_ids);

      // auto varname = slices.variable->GetName();
      // for (auto pids : parity_ids) {
      //   ThreadPool::Global()->Schedule([varname, pids, diffs, empty_cb, client] {
      //       client->SparsePushWithoutParity(varname,
      //                                                   pids,
      //                                                   "AssignSubUpdater",
      //                                                   client->Args(diffs),
      //                                                   empty_cb);
      //   });

      //   ThreadPool::Global()->Schedule([varname, pids, diffs, empty_cb, client] {
      //       client->SparsePushWithoutParity(varname,
      //                                       pids,
      //                                       "AssignSubUpdater",
      //                                       client->Args(diffs),
      //                                       empty_cb);
      //   });

      // }

    }
    MomentumMapRangeUpdater::ongoing_update_count_mtx.lock();
    MomentumMapRangeUpdater::ongoing_udpate_count -= 1;
    MomentumMapRangeUpdater::ongoing_update_count_mtx.unlock();
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(MomentumServerUpdater, MomentumServerUpdater);

}
}
}

