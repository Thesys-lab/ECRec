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
#include "ps-plus/common/base_parity_utils.h"
namespace ps {
namespace server {
namespace udf {

using std::vector;

float multiplier = 0.00001;
uint32_t acc_max = 1 << 8;
uint32_t multiplier_multiplication = 2;
std::mutex mut;

class AdagradUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& initial_accumulator_values) const {
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
      size_t low_freq_threshold = data_tensor->Shape().Dims()[0] * (1 - HIGH_FREQ_PERCENTAGE);
      Tensor* data_low_prec = slices.variable->GetVariableLikeSlot("adagrad_low_prec", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
      Tensor* acc_low_prec = slices.variable->GetVariableLikeSlot("adagrad_accumulation_low_prec", types::kInt16, [=]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
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
                  if (slice > low_freq_threshold) {
                    T* grad = grad_tensor.Raw<T>(i);
                    T* acc = acc_tensor->Raw<T>(slice);
                    T* data = data_tensor->Raw<T>(slice);
                    for (size_t j = 0; j < slices.slice_size; j++) {
                      *acc += *grad * *grad;
                      *data -= *grad * learning_rate / sqrt(*acc);
                      data++;grad++;acc++;
                    }
                  } else {
                    T* grad = grad_tensor.Raw<T>(i);
                    T* acc = acc_tensor->Raw<T>(slice);
                    T* data = data_tensor->Raw<T>(slice);
                    uint16_t* acc_lp = acc_low_prec->Raw<uint16_t >(slice);
                    T* data_lp = data_low_prec->Raw<T>(slice);

                    for (size_t j = 0; j < slices.slice_size; j++) {
                      auto diff = *grad * *grad;
                      while (true) {
                        auto diff_with_multiplier = std::floor(diff/multiplier);
                        uint32_t r = *acc_lp + diff_with_multiplier;
                        if (r  < acc_max) {
                          *acc_lp = r;
                          break;
                        } else {
                          mut.lock();
                          multiplier *= multiplier_multiplication;
                          for (uint32_t j = 0; j < acc_low_prec->Shape().NumElements(); j ++) {
                            *(acc_low_prec->Raw<uint16_t>() + j) /= multiplier_multiplication;
                          }
                          mut.unlock();
                        }
                      }
                      *acc += diff;
                      *data -= *grad * learning_rate / sqrt(*acc);
                      *data_lp -= *grad * learning_rate / sqrt(*acc_lp * multiplier);
                      data++;grad++;acc++;
                    }
                  }

                }
                return Status::Ok();
              }));
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdagradUpdater, AdagradUpdater);

}
}
}
