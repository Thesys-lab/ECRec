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

#define PARITY_DATA_CHUNK_SIZE 4

namespace ps {

class ParityUtils {
public:
  ParityUtils(VariableInfo *variableInfo) {
    _variable_info = variableInfo;
    _max_part_size = 0;
    auto total_size = 0u;
    _server_start_ids.push_back(0);
    for (auto part : variableInfo->parts) {
      total_size += part.size;
      _max_part_size = std::max(_max_part_size, part.size);
      _server_start_ids.push_back(total_size);
    }
    _single_server_size = _max_part_size / PARITY_DATA_CHUNK_SIZE *
                          (PARITY_DATA_CHUNK_SIZE + 1);
    GenerateParityMapTable();
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
    shape_vec.push_back(num_elements * 2);
    TensorShape new_shape(shape_vec);

    *result_ids = Tensor(ids.Type(), new_shape, new ps::initializer::NoneInitializer());

    // for id at the ith position, place the corresponding server_id at the ith position,
    // and the corresponding parity_id at the i + num_elements position.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t>& r) {
       for (size_t i = r.begin(); i < r.end(); i ++) {
         this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), result_ids->Raw<size_t>(i), result_ids->Raw<size_t>(i + num_elements));
       }
    });
  }

  const void MapClientIdToServerId(size_t client_id, size_t* server_id, size_t* parity_id) {
    size_t server;
    for (server = 0; server < _num_servers; server++) {
      if (_server_start_ids[server] > client_id) break;
    }
    server--;

    if (server == -1) {
      printf("Failed to match\n");
      return;
    }

    auto offset = client_id - _server_start_ids[server];
    *server_id = offset + server * _single_server_size;

    if (parity_id != nullptr) {
      auto block_number = offset / PARITY_DATA_CHUNK_SIZE;
      auto block_row = offset % PARITY_DATA_CHUNK_SIZE;
      auto parity_row = _max_part_size + block_number;
      auto parity_col = _parity_map_table[block_row][server];
      *parity_id = parity_row + parity_col * _single_server_size;
    }
  }

private:
  void GenerateParityMapTable() {
    // table has PARITY_DATA_CHUNK_SIZE columns and num_servers rows
    auto num_servers = _variable_info->parts.size();

    // init counters
    std::vector<size_t> counters;
    for (auto col = 0u; col < num_servers; col++) {
      counters.push_back(0);
    }

    // init empty table
    for (auto row = 0u; row < PARITY_DATA_CHUNK_SIZE; row++) {
      std::vector<size_t> row_vec;
      for (auto col = 0u; col < num_servers; col++) {
        row_vec.push_back(0);
      }
      _parity_map_table.push_back(row_vec);
    }

    // Start assigning blocks to parities sequentially
    auto current_column = 0u;
    for (auto server = 0u; server < num_servers; server++) {
      for (auto data_count = 0u; data_count < PARITY_DATA_CHUNK_SIZE; data_count++) {
        _parity_map_table[counters[current_column]][current_column] = server;
        counters[current_column]++;
        if (current_column == num_servers - 1) current_column = 0;
        else current_column++;
      }
    }
  }

  VariableInfo *_variable_info;
  std::vector<size_t> _server_start_ids;
  size_t _max_part_size;
  size_t _single_server_size;
  size_t _num_servers;
  std::vector<std::vector<size_t>> _parity_map_table;
};
} //ps

#endif  // PS_COMMON_NET_UTILS_H
