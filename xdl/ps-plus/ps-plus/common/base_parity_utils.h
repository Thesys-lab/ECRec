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
const size_t PARITY_N = 4;
const size_t PARITY_K = 3;
const size_t INIT_BATCH_NUM_CHUNKS = 1 << 26;
const size_t RECOVERY_BATCH_NUM_IDS = 1 << 26;
const std::vector<float> CLIENT_PARITY_FUNC = {1, 1, 1};
const std::unordered_set<std::string> VARIABLE_NAMES_WITH_PARITY = {"emb1"};
const std::unordered_set<size_t> SIMULATED_FAILED_SERVERS = {};
const std::unordered_set<size_t> SIMULATED_RECOVERY_SERVERS = {};
const bool SERVER_PARITY_UPDATE = true;

namespace ps {
class BaseParityScheme {
public:
  // currently requires parity_k <= 64
  BaseParityScheme(const VariableInfo *variableInfo, size_t parity_n, size_t parity_k,
                   const std::vector<float> parity_func) {
    _parity_n = parity_n;
    _parity_k = parity_k;
    _parity_func = parity_func;
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
      if (one_count == parity_k) {
        GenerateInverseMatrices(offset_bitmap);
      }
    }
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
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), result_ids->Raw<size_t>(i), nullptr);
        }
    });
  }


  // todo: add possible parallelism
  void
  SimpleMapClientToServerTensorWithParity(const Tensor &ids, const Tensor &diff, Tensor *result_ids, Tensor *result_diff,
                                    bool include_original_ids = false) {
    std::vector<size_t> ids_vec({ids.Shape().NumElements() * 2});
    TensorShape new_ids_shape(ids_vec);
    *result_ids = Tensor(ids.Type(), new_ids_shape, new ps::initializer::NoneInitializer());


    size_t col_size = diff.Shape().Dims()[1];
    std::vector<size_t> diff_vec({diff.Shape().Dims()[0] * 2, col_size});
    TensorShape new_diff_shape(diff_vec);
    *result_diff = Tensor(diff.Type(), new_diff_shape, new ps::initializer::NoneInitializer());

    auto num_elements = ids.Shape().NumElements();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), result_ids->Raw<size_t>(i * 2), result_ids->Raw<size_t>(i * 2 + 1));
          memcpy(result_diff->Raw<float>(i * 2), diff.Raw<float>(i), sizeof(float) * col_size);
          memcpy(result_diff->Raw<float>(i * 2 + 1), diff.Raw<float>(i), sizeof(float) * col_size);
        }
    });
  }


  void MapClientToServerIds(const Tensor &ids, Tensor *result_ids) {
    auto num_elements = ids.Shape().NumElements();
    *result_ids = Tensor(ids.Type(), ids.Shape(), new initializer::NoneInitializer());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elements), [&](tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          auto client_id = *(ids.Raw<size_t>(i));
          auto chunk_number = client_id / _parity_k;
          auto chunk_offset = client_id % _parity_k;
          auto horizontal_start = chunk_number * _parity_n;
          auto horizontal_id = horizontal_start + chunk_offset;
          auto server_id = HorizontalToVerticalId(horizontal_id);
          *(result_ids->Raw<size_t>(i)) = server_id;
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
          auto horizontal_start = chunk_number * _parity_n;
          for (auto j = _parity_k; j < _parity_n; j++) {
            *(result_ids[j - _parity_k].Raw<size_t>(i)) = (HorizontalToVerticalId(horizontal_start + j));
          }
        }
    });
  }

  // todo: add possible parallelism
  void
  MapClientToServerTensorWithParity(const Tensor &ids, const Tensor &diff, Tensor *result_ids, Tensor *result_diff,
                                    bool include_original_ids = false) {

    if (_parity_n - _parity_k == 1) {
      SimpleMapClientToServerTensorWithParity(ids, diff, result_ids, result_diff);
      return ;
    }
    // Step 1: get number of elements in ids
    auto num_elements = ids.Shape().NumElements();
    auto num_cols = diff.Shape().Dims()[1];

    // Step 2: create a mapping parity id to result diff
    std::unordered_map<size_t, std::vector<float>> parity_id_to_result_diff;

    // Create a vector to place original ids and original diffs
    std::vector<size_t> original_ids;

    // Step 3: for id at the ith position, place the corresponding server id into map, with the corresponding diff
    for (size_t i = 0; i < num_elements; i++) {
      size_t parity_ids[PARITY_N - PARITY_K];
      size_t result_id;
      this->MapClientIdToServerId(*(ids.Raw<size_t>(i)), &result_id, parity_ids);
      // store corresponding server_id if include_original_ids true
      if (include_original_ids) original_ids.push_back(result_id);

      for (size_t j = 0; j < _parity_n - _parity_k; j++) {
        auto parity_id = parity_ids[j];
        if (parity_id_to_result_diff.find(parity_id) == parity_id_to_result_diff.end()) {
          std::vector<float> row;
          for (size_t k = 0; k < num_cols; k++) {
            row.push_back(*(diff.Raw<float>(i) + k) * _parity_func[j * _parity_k + i % _parity_k]);
          }
          // key not present. then find the ith element in diff, and multiply by the corresponding factor
          parity_id_to_result_diff[parity_id] = row;
        } else {
          // key already present. add the corresponding diff
          for (size_t k = 0; k < num_cols; k++) {
            parity_id_to_result_diff[parity_id][k] +=
                    *(diff.Raw<float>(i) + k) * _parity_func[j * _parity_k + i % _parity_k];
          }
        }
      }
    }

    // create shapes with number of update parities
    std::vector<size_t> id_shape_vec;
    std::vector<size_t> diff_shape_vec;
    if (include_original_ids) {
      id_shape_vec.push_back(parity_id_to_result_diff.size() + num_elements);
      diff_shape_vec.push_back(parity_id_to_result_diff.size() + num_elements);
    } else {
      id_shape_vec.push_back(parity_id_to_result_diff.size());
      diff_shape_vec.push_back(parity_id_to_result_diff.size());
    }
    diff_shape_vec.push_back(num_cols);

    // create new shapes and tensors
    TensorShape new_id_shape(id_shape_vec);
    TensorShape new_diff_shape(diff_shape_vec);
    *result_ids = Tensor(ids.Type(), new_id_shape, new ps::initializer::NoneInitializer());
    *result_diff = Tensor(diff.Type(), new_diff_shape, new ps::initializer::NoneInitializer());

    // Step 4: convert the map to tensor
    auto counter = 0;
    if (include_original_ids) {
      // if this is true, original_ids now have num_elements items. result_ids and result_diff are still empty.
      // first memcpy in the resulting ids/diff for original elements.
      memcpy(result_ids->Raw<void>(), original_ids.data(), SizeOfType(ids.Type()) * num_elements);
      memcpy(result_diff->Raw<void>(), diff.Raw<void>(), SizeOfType(diff.Type()) * num_elements * num_cols);
      counter = num_elements;
    }

    for (auto &it : parity_id_to_result_diff) {
      *(result_ids->Raw<size_t>(counter)) = it.first;
      memcpy(result_diff->Raw<float>(counter), it.second.data(), SizeOfType(diff.Type()) * num_cols);
      counter++;
    }
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
          auto serverthis = VerticalToHorizontalId(this_id);
          auto server0 = VerticalToHorizontalId(friend_ids_vector[0]);
          auto server1 = VerticalToHorizontalId(friend_ids_vector[1]);
          auto server2 = VerticalToHorizontalId(friend_ids_vector[2]);
        }
    });
    return true;
  }


  // requires each entry in ids at index i correspond to its friend ids at i*k,
  // i*k+1, i*k+k-1;
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

  /*
   * Override the following FOUR methods for an alternative placement strategy.
   */
  void MapClientIdToServerId(size_t client_id, size_t *server_id, size_t *parity_ids) {
    auto chunk_number = client_id / _parity_k;
    auto chunk_offset = client_id % _parity_k;
    auto horizontal_start = chunk_number * _parity_n;
    auto horizontal_id = horizontal_start + chunk_offset;
    if (server_id) {
      *server_id = HorizontalToVerticalId(horizontal_id);
    }
    if (parity_ids) {
      for (auto i = _parity_k; i < _parity_n; i++) {
        parity_ids[i - _parity_k] = (HorizontalToVerticalId(horizontal_start + i));
      }
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
        friend_ids->push_back(HorizontalToVerticalId(horizontal_friend_id));
        if (friend_ids->size() == _parity_k) break;
      }
    }
  }

  bool RecoverSingleServerColumn(size_t *friend_server_indices, size_t server_index, float *friend_values,
                                 float *this_server_values, size_t num_columns) {
    // find horizontal ids and offsets
    auto server_offset = VerticalToHorizontalId(server_index) % _parity_n;
    size_t friend_server_offsets = 0;
    for (size_t i = 0; i < _parity_k; i++) {
      auto offset = VerticalToHorizontalId(friend_server_indices[i]) % _parity_n;
      friend_server_offsets |= 1 << offset;
    }
    // get inverse matrix
    std::vector<float> inverse_matrix;
    if (!GetRecoveryInverseMatrix(friend_server_offsets, &inverse_matrix)) return false;

    // if server_id is not parity, only need to calculate one entry
    if (server_offset < _parity_k) {
      // iterate through each column in tensor
      for (size_t i = 0; i < num_columns; i++) {
        this_server_values[i] = 0.0;
        for (size_t col = 0; col < _parity_k; col++) {
          this_server_values[i] +=
                  friend_values[col * num_columns + i] * inverse_matrix[server_offset * _parity_k + col];
        }
      }
      return true;
    }

    // need to recover all entries if server correspond to parity
    // calculate original values for each column
    for (size_t i = 0; i < num_columns; i++) {
      std::vector<float> original_values;
      for (size_t row = 0; row < _parity_k; row++) {
        float val = 0.0;
        for (size_t col = 0; col < _parity_k; col++) {
          val += friend_values[col * num_columns + i] * inverse_matrix[row * _parity_k + col];
        }
        original_values.push_back(val);
      }
      // calculate with the (server_offset - parity_k) row
      this_server_values[i] = 0.0;
      for (size_t col = 0; col < _parity_k; col++) {
        this_server_values[i] +=
                original_values[col * num_columns + i] * _parity_func[(server_offset - _parity_k) * _parity_k + col];
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

  bool GetRecoveryInverseMatrix(size_t friend_server_offset_bits, std::vector<float> *result) {
    *result = _inverses[friend_server_offset_bits];
    return true;
  }

  bool GenerateInverseMatrices(size_t friend_server_offset_bits) {
    std::vector<float> result;
    std::vector<float> matrix;
    size_t ind = 0;
    for (size_t ind_count = 0; ind_count < _parity_k; ind_count ++) {
      while (!(friend_server_offset_bits & (1 << ind))) {
        ind += 1;
      }
      if (ind < _parity_k) {
        // case 1: append an original row
        for (size_t j = 0; j < _parity_k; j++) {
          if (j == ind) matrix.push_back(1);
          else matrix.push_back(0);
        }
      } else {
        // case 2: copy over the original parity function
        auto row = ind - _parity_k;
        for (size_t j = 0; j < _parity_k; j++) {
          matrix.push_back(_parity_func[row * _parity_k + j]);
        }
      }
      ind += 1;
    }

    auto success = inverse(matrix, &result, _parity_k);
    if (success) {
      _inverses[friend_server_offset_bits] = result;
    }
    return success;
  }


  void getCofactor(std::vector<float> &A, std::vector<float> &temp, size_t p, size_t q, size_t n, size_t N) {
    size_t i = 0, j = 0;
    for (size_t row = 0; row < n; row++) {
      for (size_t col = 0; col < n; col++) {
        if (row != p && col != q) {
          temp[i * N + j] = A[row * N + col];
          j++;
          if (j == n - 1) {
            j = 0;
            i++;
          }
        }
      }
    }
  }

  float determinant(std::vector<float> &A, size_t n, size_t N) {
    float result = 0;
    if (n == 1) {
      return A[0];
    }
    std::vector<float> temp(N * N);
    float sign = 1;
    for (size_t f = 0; f < n; f++) {
      // Getting Cofactor of A[0][f]
      getCofactor(A, temp, 0, f, n, N);
      auto r = determinant(temp, n - 1, N);
      result += sign * A[f] * r;
      sign = -sign;
    }
    return result;
  }

  void adjoint(std::vector<float> &A, std::vector<float> &adj, size_t N) {
    if (N == 1) {
      adj[0] = 1;
      return;
    }

    auto sign = 1;
    std::vector<float> temp(N * N);

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        getCofactor(A, temp, i, j, N, N);
        sign = ((i + j) % 2 == 0) ? 1 : -1;
        adj[j * N + i] = (sign) * (determinant(temp, N - 1, N));
      }
    }
  }

  bool inverse(std::vector<float> &A, std::vector<float> *inverse, size_t N) {
    auto det = determinant(A, N, N);
    if (det == 0) {
      return false;
    }

    std::vector<float> adj(N * N);
    adjoint(A, adj, N);

    for (size_t i = 0; i < N; i++)
      for (size_t j = 0; j < N; j++)
        inverse->push_back(adj[i * N + j] / det);

    return true;
  }

  std::vector<size_t> _server_start_ids;
  size_t _max_part_size;
  size_t _single_server_size;
  size_t _num_servers;
  size_t _total_size;
  size_t _parity_n;
  size_t _parity_k;
  std::vector<float> _parity_func;
  std::vector<size_t> _servers;
  std::unordered_map<size_t, std::vector<float>> _inverses;
};
} //ps

#endif  // PS_COMMON_NET_UTILS_H
