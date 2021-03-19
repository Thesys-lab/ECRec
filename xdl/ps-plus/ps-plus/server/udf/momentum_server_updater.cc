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

    // wait for update allowed
    while (!MomentumMapRangeUpdater::update_allowed) {
      MomentumMapRangeUpdater::wait_update_allowed();
    }

    MomentumMapRangeUpdater::ongoing_update_count_mtx.lock();
    MomentumMapRangeUpdater::ongoing_udpate_count += 1;
    MomentumMapRangeUpdater::ongoing_update_count_mtx.unlock();

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

      //Create id tensors
      std::vector<size_t> id_shape_vec({slices.slice_id.size()});
      TensorShape id_shape(id_shape_vec);
      Tensor ids(types::kInt64, id_shape, new initializer::NoneInitializer());

      //Crete diff tensor

      std::vector<size_t> diff_shape_vec({slices.slice_id.size(), slices.slice_size});
      TensorShape diff_shape(diff_shape_vec);
      Tensor diff_tens(types::kFloat, diff_shape, new initializer::NoneInitializer());

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

      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
          T* diff_ptr = diff_tens.Raw<T>();
          for (size_t i = r.begin; i < r.end; i++) {
            int64_t slice = slices.slice_id[i];
            if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
              continue;
            }
            if (use_map && slice >= MomentumMapRangeUpdater::map_range_start && slice < MomentumMapRangeUpdater::map_range_end) {
              float* data = MomentumMapRangeUpdater::temp_map->Raw<float>(slice - MomentumMapRangeUpdater::map_range_start);
              float* acc = MomentumMapRangeUpdater::acc_temp_map->Raw<float>(slice - MomentumMapRangeUpdater::map_range_start);
              float* grad = grad_tensor.Raw<float>(i);
              if (use_nesterov) {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  float diff = *grad * learning_rate + *acc * momentum * learning_rate;
                  *data -= diff;
                  *diff_ptr = diff;
                  data++; acc++; grad++; diff_ptr++;
                }
              } else {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  float diff = *acc * learning_rate;
                  *data -= diff;
                  *diff_ptr = diff;
                  data++; acc++; grad++; diff_ptr++;                }
              }
            } else {
              T* data = data_tensor->Raw<T>(slice);
              auto id = slice + min_id;
              T* acc = acc_tensor->Raw<T>(slice);
              T* grad = grad_tensor.Raw<T>(i);
              *(ids.Raw<size_t>(i)) = id;
              if (use_nesterov) {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  T diff = *grad * learning_rate + *acc * momentum * learning_rate;
                  *data -= diff;
                  *diff_ptr = diff;
                  data++; acc++; grad++; diff_ptr++;
                }
              } else {
                for (size_t j = 0; j < slices.slice_size; j++) {
                  *acc = *acc * momentum + *grad;
                  T diff = *acc * learning_rate;
                  *data -= diff;
                  *diff_ptr = diff;
                  data++; acc++; grad++; diff_ptr++;
                }
              }
            }
          }
          return Status::Ok();
      }));

      auto empty_cb = [](const Status &st) {
      };
      std::vector<Tensor> diffs;
      diffs.push_back(diff_tens);

      LOG(INFO) << "Tianyu: HERE!!";

      VariableInfo info;
      client->GetVariableInfo(slices.variable->GetName(), &info);
      BaseParityScheme pu(&info, PARITY_N, PARITY_K, CLIENT_PARITY_FUNC);
      std::vector<Tensor> parity_ids;
      pu.MapServerToParityIds(ids, parity_ids);

      auto varname = slices.variable->GetName();
      for (auto pids : parity_ids) {
        ThreadPool::Global()->Schedule([varname, pids, diffs, empty_cb, client] {
            client->SparsePushWithoutParity(varname,
                                                        pids,
                                                        "AssignSubUpdater",
                                                        client->Args(diffs),
                                                        empty_cb);
        });

        ThreadPool::Global()->Schedule([varname, pids, diffs, empty_cb, client] {
            client->SparsePushWithoutParity(varname,
                                            pids,
                                            "AssignSubUpdater",
                                            client->Args(diffs),
                                            empty_cb);
        });

      }

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

