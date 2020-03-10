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

#include "ps-plus/ps-plus/message/variable_info.h"
#include "ps-plus/ps-plus/common/tensor.h"
#include "ps-plus/ps-plus/common/initializer.h"
#include "ps-plus/ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/ps-plus/common/types.h"
#include "ps-plus/ps-plus/client/client.h"
#include "tbb/parallel_for.h"

#define PARITY_N 4
#define PARITY_K 2
#define INIT_BATCH_NUM_CHUNKS 1 << 16

const int parity_func[PARITY_N - PARITY_K][PARITY_K] = {{1, 1}, {1, 2}};

namespace ps {

class ParityUtils {
public:
  ParityUtils(VariableInfo *variableInfo) {
    _variable_name = variableInfo->name;
    _max_part_size = 0;
    _total_size = 0u;
    _num_servers = 0;
    for (auto part : variableInfo->parts) {
      _total_size += part.size;
      _num_servers ++;
      _max_part_size = std::max(_max_part_size, part.size);
      _server_start_ids.push_back(_total_size);
    }
    _single_server_size = _max_part_size * PARITY_N / PARITY_K;
    if ((_max_part_size * PARITY_N) % PARITY_K != 0) _single_server_size ++;
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

  // todo: add possible parallelism
  void MapClientToServerTensorWithParity(const Tensor &ids, const Tensor &diff, Tensor *result_ids, Tensor *result_diff,
          bool include_original_ids = false) {
    // Step 1: get number of elements in ids
    auto num_elements = ids.Shape().NumElements();

    // Step 2: create a mapping parity id to result diff
    std::unordered_map<size_t, double> parity_id_to_result_diff;

    // Create a vector to place original ids and original diffs
    std::vector<size_t> original_ids;

    // Step 3: for id at the ith position, place the corresponding server id into map, with the corresponding diff
    for (size_t i = 0; i < num_elements; i ++) {
      std::vector<size_t> parity_ids;
      size_t result_id;
      this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), &result_id, &parity_ids);
      // store corresponding server_id if include_original_ids true
      if (include_original_ids) original_ids.push_back(result_id);

      for (auto j = 0; j < PARITY_N - PARITY_K; j ++) {
        auto parity_id = parity_ids[j];
        if (parity_id_to_result_diff.find(parity_id) == parity_id_to_result_diff.end()) {
          // key not present. then find the ith element in diff, and multiply by the corresponding factor
          parity_id_to_result_diff[parity_id] = *(diff.Raw<double>(i)) * parity_func[j][i % PARITY_K];
        } else {
          // key already present. add the corresponding diff
          parity_id_to_result_diff[parity_id] += *(diff.Raw<double>(i)) * parity_func[j][i % PARITY_K];
        }
      }
    }

    // create a 1d shape with number of update parities
    std::vector<size_t> shape_vec;
    if (include_original_ids) {
      shape_vec.push_back(parity_id_to_result_diff.size() + num_elements);
    } else {
      shape_vec.push_back(parity_id_to_result_diff.size());
    }
    TensorShape new_shape(shape_vec);
    *result_ids = Tensor(ids.Type(), new_shape, new ps::initializer::NoneInitializer());
    *result_diff = Tensor(diff.Type(), new_shape, new ps::initializer::NoneInitializer());

    // Step 4: convert the map to tensor
    auto counter = 0;
    if (include_original_ids) {
      // if this is true, original_ids now have num_elements items. result_ids and result_diff are still empty.
      // first memcpy in the resulting ids/diff for original elements.
      memcpy(result_ids->Raw<void>(), original_ids.data(), SizeOfType(ids.Type()) * num_elements);
      memcpy(result_diff->Raw<void>(), diff.Raw<void>(), SizeOfType(diff.Type()) * num_elements);
      counter = num_elements;
    }

    for (auto & it : parity_id_to_result_diff) {
      *(result_ids->Raw<size_t>(counter)) = it.first;
      *(result_ids->Raw<double>(counter)) = it.second;
      counter ++;
    }
  }

private:
  void MapClientIdToServerId(size_t client_id, size_t* server_id, std::vector<size_t>* parity_ids) {
    auto chunk_number = client_id / PARITY_K;
    auto chunk_offset = client_id % PARITY_K;
    auto horizontal_start = chunk_number * PARITY_N;
    auto horizontal_id = horizontal_start + chunk_offset;
    if (server_id) {
      *server_id = HorizontalToVerticalId(horizontal_id);
    }
    if (parity_ids) {
      for (auto i = PARITY_K; i < PARITY_N; i ++) {
        parity_ids->push_back(HorizontalToVerticalId(horizontal_start + i));
      }
    }
  }

  size_t HorizontalToVerticalId(size_t horizontal_id) {
    auto server = horizontal_id % _num_servers;
    auto offset = horizontal_id / _num_servers;
    return offset + server * _single_server_size;
  }

  std::vector<size_t> _server_start_ids;
  std::string _variable_name;
  size_t _max_part_size;
  size_t _single_server_size;
  size_t _num_servers;
  size_t _total_size;
};
} //ps

#endif  // PS_COMMON_NET_UTILS_H
