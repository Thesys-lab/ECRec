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
uint32_t update_counter = 0;
float multiplier = 0.001;
uint32_t acc_max = 1 << 8;
uint32_t multiplier_multiplication = 2;
std::mutex mut;
class AdagradUpdaterLowPrec : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double> > {
public:
  virtual Status SimpleRun(
          UdfContext* ctx,
          const vector<Slices>& sslices,
          const vector<Tensor>& grad_tensors,
          const vector<double>& learning_rates,
          const vector<double>& initial_accumulator_values) const {
    mut.lock();
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != initial_accumulator_values.size()) {
      return Status::ArgumentError("AdagradUpdaterLowPrec: slices and other size not match");
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
      Tensor* acc_tensor_compressed = slices.variable->GetVariableLikeSlot("adagrad_accumulation_lowpre", types::kInt16, [=]{ return new initializer::ConstantInitializer(std::round(initial_accumulator_value / multiplier)); });
      size_t low_freq_threshold = data_tensor->Shape().Dims()[0] * (1 - HIGH_FREQ_PERCENTAGE);
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
              uint16_t *acc_comp = acc_tensor_compressed->Raw<uint16_t>(slice);
              T* data = data_tensor->Raw<T>(slice);
              for (size_t j = 0; j < slices.slice_size; j++) {
                auto diff = *grad * *grad;
                *acc += diff;
                while (true) {
                  auto diff_with_multiplier = std::round(diff/multiplier);
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
            }
          }
          return Status::Ok();
      }));

      size_t bucket1 = 0;
      size_t bucket2p5 = 0;
      size_t bucket5 = 0;
      size_t bucket10 = 0;
      size_t bucket20 = 0;
      size_t bucket30 = 0;
      size_t bucket50 = 0;
      size_t bucket100 = 0;
      if (update_counter % 25 == 0) {
        printf("Print all accumulation round %lu: ", update_counter);
        for (uint32_t i = 0; i < acc_tensor->Shape().NumElements() * (1 - HIGH_FREQ_PERCENTAGE); i ++) {
          auto off = std::abs((*(acc_tensor->Raw<float>() + i) - *(acc_tensor_compressed->Raw<uint16_t>() + i) * multiplier)/(*(acc_tensor->Raw<float>() + i)));
          if (off < 0.01) {
            bucket1 ++;
          } else if (off < 0.025) {
            bucket2p5 ++;
          } else if (off < 0.05) {
            bucket5 ++;
          } else if (off < 0.10) {
            bucket10 ++;
          } else if (off < 0.20) {
            bucket20 ++;
          } else if (off < 0.30) {
            bucket30 ++;
          } else if (off < 0.50) {
            bucket50 ++;
          } else {
            bucket100 ++;
          }
        }
        printf("< %1 %f\n", (float)bucket1/acc_tensor->Shape().NumElements());
        printf("< %2.5 %f\n", (float)bucket2p5/acc_tensor->Shape().NumElements());
        printf("< %5 %f\n", (float)bucket5/acc_tensor->Shape().NumElements());
        printf("< %10 %f\n", (float)bucket10/acc_tensor->Shape().NumElements());
        printf("< %30 %f\n", (float)bucket30/acc_tensor->Shape().NumElements());
        printf("< %50 %f\n", (float)bucket50/acc_tensor->Shape().NumElements());
        printf("< %100 %f\n", (float)bucket100/acc_tensor->Shape().NumElements());
        for (uint32_t j = 0; j < 20; j ++) {
          printf("%f,%f,%f,%u ", *(acc_tensor->Raw<float>() + j), *(acc_tensor_compressed->Raw<uint16_t>() + j) * multiplier, multiplier, *(acc_tensor_compressed->Raw<uint16_t>() + j));
        }
        printf("\n");

        uint32_t init_count = 0;
        for (uint32_t j = 0; j < acc_tensor->Shape().NumElements() *  (1 - HIGH_FREQ_PERCENTAGE); j ++) {
          float val = *(acc_tensor->Raw<float>() + j);
          if (val < initial_accumulator_value + 0.000001 && val > initial_accumulator_value - 0.00000001) {
            init_count ++;
          }
        }
        printf("Acc values total %u zeroes %u init percentage %f\n", acc_tensor->Shape().NumElements(), init_count, (float)init_count/acc_tensor->Shape().NumElements());
      }
      uint32_t zero_count = 0;
      for (uint32_t k = 0; k < grad_tensor.Shape().NumElements(); k++) {
        float val = *(grad_tensor.Raw<float>() + k);
        if (val < 0.0001 && val > -0.0001) {
          zero_count ++;
        }
      }
      printf("Update on variable: %s with total %u zeroes %u zero percentage %f\n",
             slices.variable->GetName().c_str(), grad_tensor.Shape().NumElements(), zero_count,
             (float) zero_count / grad_tensor.Shape().NumElements());
      update_counter ++;
      printf("update counter: %lu\n", update_counter);
    }
    mut.unlock();
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdagradUpdaterLowPrec, AdagradUpdaterLowPrec);

}
}
}
