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

// REDUNDANCY: add utility set for redundancy.

#ifndef PS_COMMON_PARITY_UTILS_H_
#define PS_COMMON_PARITY_UTILS_H_

#include <string>
#include <vector>
#include <stdlib.h>

#include "ps-plus/message/variable_info.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer.h"
#include "ps-plus/ps-plus/common/initializer/none_initializer.h"
#include "tbb/parallel_for.h"

#define PARITY_N 4
#define PARITY_K 2
const int parity_func[PARITY_N - PARITY_K][PARITY_K] = {{1, 1}, {1, 2}};

namespace ps {

class ParityUtils {
public:
  ParityUtils(VariableInfo *variableInfo) {
    _max_part_size = 0;
    auto total_size = 0u;
    _server_start_ids.push_back(0);
    for (auto part : variableInfo->parts) {
      total_size += part.size;
      _max_part_size = std::max(_max_part_size, part.size);
      _server_start_ids.push_back(total_size);
    }
    _single_server_size = _max_part_size / PARITY_K * PARITY_N;
  }

  void MapClientToServerTensor(const Tensor &ids, Tensor *result_ids) {
    // get new shape, with double elements including the parities
    auto num_elements = ids.Shape().NumElements();
    std::vector<size_t> shape_vec;
    shape_vec.push_back(num_elements);
    TensorShape new_shape(shape_vec);

    *result_ids = Tensor(ids.Type(), new_shape, new ps::initializer::NoneInitializer());

    // for id at the ith position, place the corresponding server_id at the ith position,
    // and the corresponding parity_id at the i + num_elements position.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t>& r) {
       for (size_t i = r.begin(); i < r.end(); i ++) {
         this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), result_ids->Raw<size_t>(i), nullptr);      
       }
    });
  }

  void MapClientToServerTensorWithParity(const Tensor &ids, Tensor *result_ids) {
    // get new shape, with double elements including the parities
    auto num_elements = ids.Shape().NumElements();
    std::vector<size_t> shape_vec;
    shape_vec.push_back(num_elements * (1 + PARITY_N - PARITY_K));
    TensorShape new_shape(shape_vec);

    *result_ids = Tensor(ids.Type(), new_shape, new ps::initializer::NoneInitializer());

    // for id at the ith position, place the corresponding server_id at the ith position,
    // and the corresponding parity_id at the i + num_elements position.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t>& r) {
       for (size_t i = r.begin(); i < r.end(); i ++) {
         std::vector<size_t> parity_ids;
         this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), result_ids->Raw<size_t>(i), &parity_ids);
         for (auto j = 0; j < PARITY_N - PARITY_K; j ++) {
           *(result_ids->Raw<size_t>(num_elements + (PARITY_N - PARITY_K) * i + j)) = parity_ids[j];
         }
       }
    });
  }

  const void MapClientIdToServerId(size_t client_id, size_t* server_id, std::vector<size_t>* parity_ids) {
    auto chunk_number = client_id / PARITY_K;
    auto chunk_offset = client_id % PARITY_K;
    auto horizontal_start = chunk_number * PARITY_N;
    auto horizontal_id = horizontal_start + chunk_offset;
    *server_id = HorizontalToVerticalId(horizontal_id);
    if (parity_ids) {
      for (auto i = PARITY_K; i < PARITY_N; i ++) {
        parity_ids->push_back(HorizontalToVerticalId(horizontal_start + i));
      }  
    }
  }

private:
  size_t HorizontalToVerticalId(size_t horizontal_id) {
    auto server = horizontal_id % _num_servers;
    auto offset = horizontal_id / _num_servers;
    return offset + server * _single_server_size;
  }


  std::vector<size_t> _server_start_ids;
  size_t _max_part_size;
  size_t _single_server_size;
  size_t _num_servers;
};
} //ps

#endif  // PS_COMMON_NET_UTILS_H
