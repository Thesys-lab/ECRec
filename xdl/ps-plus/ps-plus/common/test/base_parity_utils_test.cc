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

#include "gtest/gtest.h"
#include "ps-plus/common/base_parity_utils.h"
#include "ps-plus/ps-plus/message/variable_info.h"
#include <iomanip>
#include <sstream>

#define TENSOR_LENGTH 5
const std::vector<float> TEST_PARITY_FUNC = {1, 1, 1, 2};

using ps::VariableInfo;
using ps::BaseParityScheme;
using ps::TensorShape;
using ps::Tensor;

TEST(ParityUtilsTest, TestWithoutParity) {
  VariableInfo info;
  VariableInfo::Part part;
  part.server = 0;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 1;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 2;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 3;
  part.size = 100;
  info.parts.push_back(part);
  BaseParityScheme pu(&info, 4, 2, TEST_PARITY_FUNC);
  std::vector<size_t> shape;
  shape.push_back(TENSOR_LENGTH);
  TensorShape tensorShape(shape);
  Tensor ids = Tensor(ps::types::kInt64, tensorShape, new ps::initializer::NoneInitializer());
  Tensor result_ids;
  size_t ids_arr[] = {0, 5, 100, 300, 315};
  size_t expected_result_arr[] = {0, 202, 50, 150, 357};
  memcpy(ids.Raw<size_t>(), ids_arr, TENSOR_LENGTH * sizeof(size_t));
  pu.MapClientToServerTensor(ids, &result_ids);
  for (auto i = 0; i < TENSOR_LENGTH; i++) {
    EXPECT_EQ(*(result_ids.Raw<size_t>(i)), expected_result_arr[i]);
  }
}

TEST(ParityUtilsTest, TestWithParity) {
  VariableInfo info;
  VariableInfo::Part part;
  part.server = 0;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 1;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 2;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 3;
  part.size = 100;
  info.parts.push_back(part);
  BaseParityScheme pu(&info, 4, 2, TEST_PARITY_FUNC);
  std::vector<size_t> shape;
  shape.push_back(TENSOR_LENGTH);
  TensorShape tensorShape(shape);
  shape.push_back(1);
  TensorShape diffTensorShape(shape);
  Tensor ids = Tensor(ps::types::kInt64, tensorShape, new ps::initializer::NoneInitializer());
  Tensor diff = Tensor(ps::types::kFloat, diffTensorShape, new ps::initializer::NoneInitializer());
  Tensor result_ids;
  Tensor result_diff;
  size_t ids_arr[] = {0, 5, 100, 314, 315};
  float diff_arr[] = {1, 1, 1, 1, 1};
  size_t expected_result_arr[] = {400, 600, 402, 602, 450, 650, 557, 757};
  float expected_result_diff[] = {1, 1, 1, 2, 1, 1, 2, 3};
  std::unordered_map<size_t, float> result_map;
  for (auto i = 0; i < 8; i++) {
    result_map[expected_result_arr[i]] = expected_result_diff[i];
  }

  memcpy(ids.Raw<size_t>(), ids_arr, TENSOR_LENGTH * sizeof(size_t));
  memcpy(diff.Raw<float>(), diff_arr, TENSOR_LENGTH * sizeof(float));
  pu.MapClientToServerTensorWithParity(ids, diff, &result_ids, &result_diff);
  EXPECT_EQ(result_ids.Shape().NumElements(), 8);
  EXPECT_EQ(result_diff.Shape().NumElements(), 8);
  for (auto i = 0; i < 8; i++) {
    auto tid = *(result_ids.Raw<size_t >(i));
    auto tdiff = *(result_diff.Raw<float>(i));
    EXPECT_TRUE(result_map[tid] -tdiff < 0.001 && result_map[tid] -tdiff > -0.001);
  }
}

TEST(ParityUtilsTest, TestWithParityAndOriginal) {
  VariableInfo info;
  VariableInfo::Part part;
  part.server = 0;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 1;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 2;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 3;
  part.size = 100;
  info.parts.push_back(part);
  BaseParityScheme pu(&info, 4, 2, TEST_PARITY_FUNC);
  std::vector<size_t> shape;
  shape.push_back(TENSOR_LENGTH);
  TensorShape tensorShape(shape);
  shape.push_back(1);
  TensorShape diffTensorShape(shape);

  Tensor ids = Tensor(ps::types::kInt64, tensorShape, new ps::initializer::NoneInitializer());
  Tensor diff = Tensor(ps::types::kFloat, diffTensorShape, new ps::initializer::NoneInitializer());
  Tensor result_ids;
  Tensor result_diff;
  size_t ids_arr[] = {0, 5, 100, 314, 315};
  float diff_arr[] = {1, 1, 1, 1, 1};
  size_t expected_result_arr[] = {0, 202, 50, 157, 357, 400, 600, 402, 602, 450, 650, 557, 757};
  float expected_result_diff[] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3};
  std::unordered_map<size_t, float> result_map;
  for (auto i = 0; i < 13; i++) {
    result_map[expected_result_arr[i]] = expected_result_diff[i];
  }

  memcpy(ids.Raw<size_t>(), ids_arr, TENSOR_LENGTH * sizeof(size_t));
  memcpy(diff.Raw<float>(), diff_arr, TENSOR_LENGTH * sizeof(float));
  pu.MapClientToServerTensorWithParity(ids, diff, &result_ids, &result_diff, true);
  EXPECT_EQ(result_ids.Shape().NumElements(), 13);
  EXPECT_EQ(result_diff.Shape().NumElements(), 13);
  for (auto i = 0; i < 13; i++) {
    auto tid = *(result_ids.Raw<size_t >(i));
    auto tdiff = *(result_diff.Raw<float>(i));
    EXPECT_TRUE(result_map[tid] -tdiff < 0.001 && result_map[tid] -tdiff > -0.001);
  }
}

TEST(ParityUtilsTest, TestReverseTranslation) {
  VariableInfo info;
  VariableInfo::Part part;
  part.server = 0;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 1;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 2;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 5;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 8;
  part.size = 100;
  info.parts.push_back(part);

  std::vector<size_t> shape({TENSOR_LENGTH});

  TensorShape tensorShape(shape);
  Tensor ids = Tensor(ps::types::kInt64, tensorShape, new ps::initializer::NoneInitializer());
  size_t ids_arr[] = {0, 1, 2, 3, 4};
  size_t friend_ids_arr[] = {200, 400, 800, 201, 801, 202, 402, 802, 204, 404};
  Tensor friend_ids;

  std::unordered_set<size_t> failed_servers;
  failed_servers.insert(5);

  memcpy(ids.Raw<size_t>(), ids_arr, TENSOR_LENGTH * sizeof(size_t));

  BaseParityScheme pu(&info, 4, 2, TEST_PARITY_FUNC);
  pu.FindFriendIds(ids, &friend_ids, failed_servers, 0);
  EXPECT_EQ(friend_ids.Shape().NumElements(), TENSOR_LENGTH * 2);
  for (auto i = 0; i < TENSOR_LENGTH * 2; i++) {
    EXPECT_EQ(*(friend_ids.Raw<size_t >(i)), friend_ids_arr[i]);
  }
}


TEST(ParityUtilsTest, TestRecovery) {
  VariableInfo info;
  VariableInfo::Part part;
  part.server = 0;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 1;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 2;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 5;
  part.size = 100;
  info.parts.push_back(part);
  part.server = 8;
  part.size = 100;
  info.parts.push_back(part);

  std::vector<size_t> shape;
  shape.push_back(TENSOR_LENGTH);
  TensorShape tensor_shape(shape);
  shape.push_back(1);
  TensorShape value_tensor_shape(shape);

  shape.clear();
  shape.push_back(2 * TENSOR_LENGTH);
  TensorShape tensor_shape_friend(shape);
  shape.push_back(1);
  TensorShape value_tensor_shape_friend(shape);

  size_t ids_arr[] = {0, 1, 2, 3, 4};
  size_t friend_ids_arr[] = {200, 400, 800, 201, 801, 202, 402, 802, 204, 404};
  float friend_values_arr[] = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
  float expected_result_values[] = {-2, -2, -2, -3, -2};

  // tensors
  Tensor ids = Tensor(ps::types::kInt64, tensor_shape, new ps::initializer::NoneInitializer());
  Tensor friend_ids = Tensor(ps::types::kInt64, tensor_shape_friend, new ps::initializer::NoneInitializer());
  Tensor friend_values = Tensor(ps::types::kFloat, value_tensor_shape_friend, new ps::initializer::NoneInitializer());
  Tensor result_values = Tensor(ps::types::kFloat, value_tensor_shape, new ps::initializer::NoneInitializer());

  memcpy(ids.Raw<size_t>(), ids_arr, TENSOR_LENGTH * sizeof(size_t));
  memcpy(friend_ids.Raw<size_t>(), friend_ids_arr, 2 * TENSOR_LENGTH * sizeof(size_t));
  memcpy(friend_values.Raw<size_t>(), friend_values_arr, 2 * TENSOR_LENGTH * sizeof(float));

  BaseParityScheme pu(&info, 4, 2, TEST_PARITY_FUNC);
  pu.RecoverServerValues(ids, friend_ids, friend_values, result_values);

  for (auto i = 0; i < TENSOR_LENGTH; i++) {
    auto val = *(result_values.Raw<float>(i));
    EXPECT_TRUE(expected_result_values[i] - val < 0.001 && expected_result_values[i] - val > -0.001);
  }
}
