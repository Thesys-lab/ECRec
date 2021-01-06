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
#include <unordered_set>

#include "ps-plus/ps-plus/message/variable_info.h"
#include "ps-plus/ps-plus/common/tensor.h"
#include "ps-plus/ps-plus/common/initializer.h"
#include "ps-plus/ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/ps-plus/common/types.h"
#include "ps-plus/ps-plus/client/client.h"
#include "tbb/parallel_for.h"

// define all constants related to parity here
const size_t PARITY_N = 3;
const size_t PARITY_K = 2;
const std::vector<float> CLIENT_PARITY_FUNC = {1, 1, 1, 1};
const size_t INIT_BATCH_NUM_CHUNKS = 1 << 26;
const size_t RECOVERY_NUM_LOCKS = 10;
const size_t RECOVERY_NUM_BATCHES_PER_LOCK = 10;
const std::unordered_set<std::string> VARIABLE_NAMES_WITH_PARITY = {"emb1"};
const std::unordered_set<size_t> SIMULATED_FAILED_SERVERS = {0};
const std::unordered_set<size_t> SIMULATED_RECOVERY_SERVERS = {0};
const bool SERVER_PARITY_UPDATE = true;
const float HIGH_FREQ_PERCENTAGE = 0.01;

namespace ps {
class BaseParityScheme {
public:
  // currently requires parity_k <= 64
  BaseParityScheme(const VariableInfo *variableInfo, size_t parity_n, size_t parity_k,
                   const std::vector<float> parity_func) {
    _parity_n = parity_n;
    _parity_k = parity_k;
    _max_part_size = 0;
    _total_size = 0u;
    _num_servers = 0;
    for (auto part : variableInfo->parts) {
      _total_size += part.size;
      _num_servers++;
      _max_part_size = std::max(_max_part_size, part.size);
      _server_start_ids.push_back(_total_size);
      _servers.push_back(part.server);
    }
    _single_server_size = ConvertClientToServerSize(_max_part_size);

    for (size_t offset_bitmap = 0; offset_bitmap < 1 << parity_n; offset_bitmap ++) {
      size_t one_count = 0;
      for (size_t j = 0; j < parity_n; j++) {
        if ((offset_bitmap & (1 << j)) > 0) one_count ++;
      }
    }
  }

  void MapClientToServerIds(const Tensor &ids, Tensor *result_ids) {
    auto num_elements = ids.Shape().NumElements();
    *result_ids = Tensor(ids.Type(), ids.Shape(), new initializer::NoneInitializer());
    auto total_size = _single_server_size * _num_servers;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          auto client_id = *(ids.Raw<size_t>(i));
          auto chunk_number = client_id / _parity_k;
          auto chunk_offset = client_id % _parity_k;
          auto horizontal_start = chunk_number * _parity_n;
          auto horizontal_id = horizontal_start + chunk_offset;
          auto server_id = HorizontalToVerticalId(horizontal_id);
          *(result_ids->Raw<size_t>(i)) = server_id % total_size;
        }
    });
  }

  void MapClientToParityIds(const Tensor &ids, std::vector<Tensor> &result_ids) {
    for (auto i = 0; i < _parity_n - _parity_k; i++) {
      result_ids.push_back(Tensor(ids.Type(), ids.Shape(), new ps::initializer::NoneInitializer()));
    }

    auto num_elements = ids.Shape().NumElements();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          auto client_id = *(ids.Raw<size_t>(i));

          auto chunk_number = client_id / _parity_k;
          auto chunk_index = client_id % _parity_k;
          auto horizontal_start = chunk_number * _parity_n;
          for (auto j = _parity_k; j < _parity_n; j++) {
            *(result_ids[j - _parity_k].Raw<size_t>(i)) = (HorizontalToVerticalId(horizontal_start + j));
          }
        }
    });
  }

  void MapServerToParityIds(const Tensor &ids, std::vector<Tensor> &result_ids) {
    for (auto i = 0; i < _parity_n - _parity_k; i++) {
      result_ids.push_back(Tensor(ids.Type(), ids.Shape(), new ps::initializer::NoneInitializer()));
    }

    auto num_elements = ids.Shape().NumElements();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          auto server_id = *(ids.Raw<size_t>(i));
          auto horizontal_server_id = VerticalToHorizontalId(server_id);
          auto client_id = _parity_k * (horizontal_server_id / _parity_n) + horizontal_server_id % _parity_n;
          auto chunk_number = client_id / _parity_k;
          auto chunk_index = client_id % _parity_k;
          auto horizontal_start = chunk_number * _parity_n;
          for (auto j = _parity_k; j < _parity_n; j++) {
            *(result_ids[j - _parity_k].Raw<size_t>(i)) = (HorizontalToVerticalId(horizontal_start + j));
          }
        }
    });
  }

  bool FindFriendIds(const Tensor &ids, Tensor *friend_ids, std::unordered_set<size_t> failed_servers) {
    if (failed_servers.size() > _parity_n - _parity_k) {
      // cant recover with too many failed servers
      return false;
    }
    // initialize result tensor
    *friend_ids = Tensor(ids.Type(), TensorShape({ids.Shape().NumElements() * _parity_k}),
                  new initializer::NoneInitializer());
    // translate
    auto num_elements = ids.Shape().NumElements();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          auto this_id = *(ids.Raw<size_t>(i));
          std::vector<size_t> friend_ids_vector;
          MapServerIdToFriends(this_id, failed_servers, &friend_ids_vector);
          for (size_t j = 0; j < _parity_k; j++) {
            *(friend_ids->Raw<size_t>(i * _parity_k + j)) = friend_ids_vector[j];
          }
        }
    });
    return true;
  }


  // requires each entry in ids at index i correspond to its friend ids at i*k,
  // i*k+1,... i*k+k-1;
  bool RecoverServerValues(const Tensor &ids, const Tensor &friend_ids, const Tensor &friend_values,
                           Tensor *result_values) {
    auto num_columns = friend_values.Shape().Dims()[1];
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ids.Shape().NumElements()), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          RecoverSingleServerColumn(friend_ids.Raw<size_t>(i * _parity_k), *(ids.Raw<size_t>(i)),
                                               friend_values.Raw<float>(i * _parity_k), result_values->Raw<float>(i), num_columns);
        }
    });
    return true;
  }

  // using simple padding strategy
  void AdaptVariableInfoToServerSpace(VariableInfo *info) {
    info->shape[0] = _single_server_size * _num_servers;
    for (size_t i = 0; i < info->parts.size(); i ++) {
      info->parts[i].size = _single_server_size;
    }
  }

  void MapServerIdToFriends(size_t server_id, std::unordered_set<size_t> &failed_servers,
                            std::vector<size_t> *friend_ids) {
    auto horizontal_id = VerticalToHorizontalId(server_id);
    auto horizontal_chunk_start = horizontal_id - (horizontal_id % _parity_n);

    for (size_t offset = 0; offset < _parity_n; offset++) {
      auto horizontal_friend_id = horizontal_chunk_start + offset;
      auto this_server = _servers[horizontal_id % _num_servers];
      auto friend_server = _servers[horizontal_friend_id % _num_servers];
      if (failed_servers.find(friend_server) == failed_servers.end() && friend_server != this_server) {
        // ok to add the id
        auto r = HorizontalToVerticalId(horizontal_friend_id);
        friend_ids->push_back(r);
        if (friend_ids->size() == _parity_k) break;
      }
    }
    while (friend_ids->size() < _parity_k) {
      friend_ids->push_back(server_id);
    }
  }

  bool RecoverSingleServerColumn(size_t *friend_server_indices, size_t server_index, float *friend_values,
                                 float *this_server_values, size_t num_columns) {
    // find horizontal ids and offsets
    auto server_offset = VerticalToHorizontalId(server_index) % _parity_n;
    if (server_offset < _parity_k) {
      // iterate through each column in tensor
      for (size_t i = 0; i < _parity_k; i++) {
        this_server_values[i] = 0.0;
        auto friend_server_offset = VerticalToHorizontalId(friend_server_indices[i]) % _parity_n;
        if (friend_server_offset < _parity_k) {
          for (size_t col = 0; col < num_columns; col++) {
            this_server_values[i] -= friend_values[i * num_columns + col];
          }
        } else {
          for (size_t col = 0; col < _parity_k; col++) {
            this_server_values[i] += friend_values[i * num_columns + col];
          }
        }
      }
    } else {
      // iterate through each column in tensor
      for (size_t i = 0; i < _parity_k; i++) {
        this_server_values[i] = 0.0;
        auto friend_server_offset = VerticalToHorizontalId(friend_server_indices[i]) % _parity_n;
        if (friend_server_offset < _parity_k) {
          for (size_t col = 0; col < num_columns; col++) {
            this_server_values[i] -= friend_values[i * num_columns + col];
          }
        } else {
          for (size_t col = 0; col < _parity_k; col++) {
            this_server_values[i] += friend_values[i * num_columns + col];
          }
        }
      }
    }
    return true;
  }

  size_t FindServer(size_t server_id) {
    auto horizontal_id = VerticalToHorizontalId(server_id);
    return _servers[horizontal_id % _num_servers];
  }

private:
  size_t HorizontalToVerticalId(size_t horizontal_id) {
    auto server = horizontal_id % _num_servers;
    auto offset = horizontal_id / _num_servers;
    return offset + server * _single_server_size;
  }

  size_t VerticalToHorizontalId(size_t vertical_id) {
    auto server = vertical_id / _single_server_size;
    auto offset = vertical_id % _single_server_size;
    return server + offset * _num_servers;
  }

  size_t ConvertClientToServerSize(size_t si) {
    if (si * _parity_n % _parity_k == 0) return si * _parity_n / _parity_k;
    else return si * _parity_n / _parity_k + 1;
  }

  std::vector<size_t> _server_start_ids;
  size_t _max_part_size;
  size_t _single_server_size;
  size_t _num_servers;
  size_t _total_size;
  size_t _parity_n;
  size_t _parity_k;
  std::vector<size_t> _servers;
  std::unordered_map<size_t, std::vector<float>> _inverses;
};
} //ps

#endif  // PS_COMMON_NET_UTILS_H
