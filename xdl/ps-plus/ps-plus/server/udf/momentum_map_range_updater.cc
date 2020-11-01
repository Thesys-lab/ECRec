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

#include "ps-plus/server/udf/momentum_map_range_updater.h"

namespace ps {
namespace server {
namespace udf {
using std::vector;
size_t MomentumMapRangeUpdater::map_range_start = 0;
size_t MomentumMapRangeUpdater::map_range_end = 0;
Tensor *MomentumMapRangeUpdater::temp_map = nullptr;
Tensor *MomentumMapRangeUpdater::acc_temp_map = nullptr;
Tensor *MomentumMapRangeUpdater::original = nullptr;
Tensor *MomentumMapRangeUpdater::original_acc = nullptr;
bool MomentumMapRangeUpdater::update_allowed = true;
std::mutex MomentumMapRangeUpdater::update_allowed_mtx;
std::condition_variable MomentumMapRangeUpdater::update_allowed_cv;
std::mutex MomentumMapRangeUpdater::ongoing_update_count_mtx;
size_t MomentumMapRangeUpdater::ongoing_udpate_count = 0;
bool MomentumMapRangeUpdater::is_recovery_server = false;

SIMPLE_UDF_REGISTER(MomentumMapRangeUpdater, MomentumMapRangeUpdater);
}
}
}

