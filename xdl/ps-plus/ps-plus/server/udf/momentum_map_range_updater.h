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

#include "ps-plus/common/logging.h"
#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;

class MomentumMapRangeUpdater : public SimpleUdf<size_t, size_t> {
public:
  virtual Status SimpleRun(
          UdfContext* ctx,
          const size_t& new_range_start,
          const size_t& new_range_end) const {
    if (is_recovery_server) {
      return Status::Ok();
    }
    // Step 0: grab update lock
    update_allowed = false;

    // Step 1: wait for number of in flight updates to be zero
    while (ongoing_udpate_count > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Step 2: flush map
    if (map_range_start != map_range_end && original) {
      QuickMemcpy(original->Raw<float>(map_range_start), temp_map->Raw<float>(),
                  sizeof(float) * (map_range_end - map_range_start) * original->Shape().NumElements() / original->Shape().Dims()[0]);
      QuickMemcpy(original_acc->Raw<float>(map_range_start), acc_temp_map->Raw<float>(),
                  sizeof(float) * (map_range_end - map_range_start) * original_acc->Shape().NumElements() / original_acc->Shape().Dims()[0]);
    }

    // Step 3: update range
    map_range_start = new_range_start;
    map_range_end = new_range_end;

    // Step 4: add new items to map
    if (!is_recovery_server && map_range_start != map_range_end && original) {
      printf("temp_map shape: %lu %lu\n", temp_map->Shape().Dims()[0], temp_map->Shape().Dims()[1]);
      printf("original shape: %lu %lu\n", original->Shape().Dims()[0], original->Shape().Dims()[1]);
      printf("Copy size %lu\n", (map_range_end - map_range_start) * original->Shape().NumElements() /
                                original->Shape().Dims()[0]);
      QuickMemcpy(temp_map->Raw<float>(), original->Raw<float>(map_range_start),
                  sizeof(float) * (map_range_end - map_range_start) * original->Shape().NumElements() /
                  original->Shape().Dims()[0]);
      QuickMemcpy(acc_temp_map->Raw<float>(), original_acc->Raw<float>(map_range_start),
                  sizeof(float) * (map_range_end - map_range_start) * original_acc->Shape().NumElements() /
                  original_acc->Shape().Dims()[0]);
    }


    // Step 4: release lock
    notify_update_allowed();
    LOG(INFO) << "Range updated to " << map_range_start << ", "<< map_range_end;
    return Status::Ok();
  }

  static void wait_update_allowed() {
    std::unique_lock<std::mutex> lck(update_allowed_mtx);
    while (!update_allowed) update_allowed_cv.wait(lck);
  }

  static void notify_update_allowed() {
    std::unique_lock<std::mutex> lck(update_allowed_mtx);
    update_allowed = true;
    update_allowed_cv.notify_all();
  }

  static size_t map_range_start;
  static size_t map_range_end;
  static Tensor *temp_map;
  static Tensor *acc_temp_map;
  static Tensor *original;
  static Tensor *original_acc;
  static bool update_allowed;
  static std::mutex update_allowed_mtx;
  static std::condition_variable update_allowed_cv;
  static std::mutex ongoing_update_count_mtx;
  static size_t ongoing_udpate_count;
  static bool is_recovery_server;
};
}
}
}