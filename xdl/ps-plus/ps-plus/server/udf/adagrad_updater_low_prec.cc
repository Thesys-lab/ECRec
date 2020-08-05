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
size_t update_counter = 0;
std::mutex mut;
class AdagradUpdaterLowPrec : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double> > {
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
      size_t bucket1 = 0;
      size_t bucket2p5 = 0;
      size_t bucket5 = 0;
      size_t bucket10 = 0;
      size_t bucket20 = 0;
      size_t bucket30 = 0;
      size_t bucket50 = 0;
      size_t bucket100 = 0;
      size_t total = 0;
      if (update_counter % 50 == 0) {
        printf("Print all accumulation round %lu: ", update_counter);
        for (uint32_t i = 0; i < low_freq_threshold; i ++) {
          auto off = std::abs((*(acc_tensor->Raw<float>() + i) - *(acc_low_prec->Raw<uint16_t>() + i) * multiplier)/(*(acc_tensor->Raw<float>() + i)));
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
          total ++;
        }
        printf("< %1 %f\n", (float)bucket1/total);
        printf("< %2.5 %f\n", (float)bucket2p5/total);
        printf("< %5 %f\n", (float)bucket5/total);
        printf("< %10 %f\n", (float)bucket10/total);
        printf("< %20 %f\n", (float)bucket20/total);
        printf("< %30 %f\n", (float)bucket30/total);
        printf("< %50 %f\n", (float)bucket50/total);
        printf("< %100 %f\n", (float)bucket100/total);
        for (uint32_t j = 0; j < 50; j ++) {
          printf("%f,%f,%f,%u ", *(acc_tensor->Raw<float>() + j), *(acc_low_prec->Raw<uint16_t>() + j) * multiplier, multiplier, *(acc_low_prec->Raw<uint16_t>() + j));
        }
        printf("\n");

        for (uint32_t j = 0; j < 1000; j ++) {
          printf("%f,", *(acc_tensor->Raw<float>() + j));
        }
      }
      update_counter ++;
      printf("update counter: %lu\n", update_counter);
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdagradUpdaterLowPrec, AdagradUpdaterLowPrec);
}
}
}
