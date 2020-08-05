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
      std::vector<Tensor*> acc_tensors;
      for (size_t chunk_ind = 0; chunk_ind < PARITY_K; chunk_ind ++) {
        acc_tensors.push_back(slices.variable->GetVariableLikeSlot(
                "adagrad_accumulation_" + std::to_string(chunk_ind),
                data_tensor->Type(),
                [=]{ return new initializer::ConstantInitializer(initial_accumulator_value);})
        );
      }
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

            uint32_t grad_bitwise = *((uint32_t*)(&grad));
            size_t chunk_ind = grad_bitwise & 3;
            T* acc = acc_tensors[chunk_ind]->Raw<T>(slice);
            T* data = data_tensor->Raw<T>(slice);
            for (size_t j = 0; j < slices.slice_size; j++) {
              *acc += *grad * *grad;
              *data -= *grad * learning_rate / sqrt(*acc);
              data++;grad++;acc++;
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
