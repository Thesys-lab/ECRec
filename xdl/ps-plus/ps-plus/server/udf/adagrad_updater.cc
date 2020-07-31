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
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;
uint32_t update_counter = 0;
float multiplier = 0.00001;
uint32_t acc_max = 65536;
uint32_t multiplier_multiplication = 4;
std::mutex mut;
class AdagradUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& initial_accumulator_values) const {
    mut.lock();
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != initial_accumulator_values.size()) {
      return Status::ArgumentError("AdagradUpdater: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      if (!slices.writable) {
        return Status::ArgumentError("slice is not writable");
      }
      double learning_rate = learning_rates[si];
      double initial_accumulator_value = initial_accumulator_values[si];
      Tensor* data_tensor = slices.variable->GetData();
      Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("adagrad_accumulation", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
      Tensor* acc_tensor_compressed = slices.variable->GetVariableLikeSlot("adagrad_accumulation_lowpre", types::kInt16, [=]{ return new initializer::ConstantInitializer(std::floor(initial_accumulator_value / multiplier)); });
      const Tensor& grad_tensor = grad_tensors[si];
      if (grad_tensor.Type() != data_tensor->Type()) {
        return Status::ArgumentError("grad should has same datatype with variable");
      }
      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                for (size_t i = r.begin; i < r.end; i++) {
                  int64_t slice = slices.slice_id[i];
                  if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                    continue;
                  }
                  T* grad = grad_tensor.Raw<T>(i);
                  T* acc = acc_tensor->Raw<T>(slice);
                  uint16_t *acc_comp = acc_tensor_compressed->Raw<uint16_t>(slice);
                  T* data = data_tensor->Raw<T>(slice);
                  //printf("Gradient: ");
                  for (size_t j = 0; j < slices.slice_size; j++) {
                    //if (*grad > 0.000000001 || *grad < - 0.000000001) {
                      //printf("%f ", *grad);
                    //}
                    auto diff = *grad * *grad;
                    *acc += diff;
                    while (true) {
                      auto diff_with_multiplier = std::floor(diff/multiplier);
                      uint32_t r = *acc_comp + diff_with_multiplier;
                      if (r  < acc_max) {
                        *acc_comp = r;
                        break;
                      } else {
                        multiplier *= multiplier_multiplication;
                        for (uint32_t j = 0; j < acc_tensor_compressed->Shape().NumElements(); j ++) {
                          *(acc_tensor_compressed->Raw<uint16_t>() + j) /= multiplier_multiplication;
                        }
                      }
                    }
                    *data -= *grad * learning_rate / sqrt(*acc);
                    data++;grad++;acc++;acc_comp++;
                  }
                  //printf("\n");

                }
                return Status::Ok();
              }));

      if (slices.variable->GetName() == "emb1") {
        printf("(%f,%f,%u,%f)\n", *(acc_tensor->Raw<float>()), *(acc_tensor_compressed->Raw<uint16_t>()) * multiplier, *(acc_tensor_compressed->Raw<uint16_t>()), multiplier);

        /*if (update_counter % 1000 == 0) {
          printf("Print all accumulation round %lu: ", update_counter);
          for (uint32_t i = 0; i < acc_tensor->Shape().NumElements(); i ++) {
            printf("(%f,%f,%u,%f)", *(acc_tensor->Raw<float>() + i), *(acc_tensor_compressed->Raw<uint16_t>() + i) * multiplier, *(acc_tensor_compressed->Raw<uint16_t>() + i), multiplier);
          }
          printf("\n");
        }
        update_counter ++;
         */:wq

      }
    }
    mut.unlock();
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdagradUpdater, AdagradUpdater);

}
}
}
