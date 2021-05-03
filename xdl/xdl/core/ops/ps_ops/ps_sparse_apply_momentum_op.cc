/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"
#include "ps-plus/ps-plus/common/base_parity_utils.h"

namespace xdl {

template <typename T>
class PsSparseApplyMomentumOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_lr), done);
    Tensor t_momentum;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_momentum), done);    
    double lr = t_lr.Scalar<double>();
    double momentum = t_momentum.Scalar<double>();
    Tensor t_use_nesterov;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(2, &t_use_nesterov), done);
    bool use_nesterov = t_use_nesterov.Scalar<bool>();
    Tensor grad;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(3, &grad), done);
    Tensor indices;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(4, &indices), done);    
    ps::Tensor convert_grad;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(grad, &convert_grad),
        done);
    ps::Tensor convert_indices;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(indices, &convert_indices),
        done);

    LOG(INFO) << "Tianyu: ps_sparse_apply_momentum: lr=" << lr << " momentum=" << momentum;

    Tensor t_write_num;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(5, &t_write_num), done);
    Tensor t_write_interval;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(6, &t_write_interval), done);
    int write_num = t_write_num.Scalar<int>();
    int write_interval = t_write_interval.Scalar<int>();

    LOG(INFO) << "Tianyu: ps_sparse_apply_momentum: write_num=" << write_num << " write_interval=" << write_interval;

    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    std::vector<ps::Tensor> grad_vec = {convert_grad};
    std::vector<double> lr_vec = {lr};
    std::vector<double> momentum_vec = {momentum};
    std::vector<bool> use_nesterov_vec = {use_nesterov};

    switch(var_type_) {
    case VarType::kIndex:
        if (var_name_ == "emb1" && !SERVER_PARITY_UPDATE) {
          std::vector<ps::Tensor> grad_vec2 = {convert_grad.Clone()};
          client->SparsePush(
                  var_name_,
                  convert_indices,
                  "MomentumUpdater",
                  client->Args(grad_vec, lr_vec, momentum_vec, use_nesterov_vec,
                          grad_vec2, lr_vec, momentum_vec, use_nesterov_vec),
                  cb);
        } else {
          LOG(INFO) << "Tianyu: ps_sparse_apply_momentum: calling correct SparsePush";
          client->SparsePush(
                  var_name_,
                  convert_indices,
                  "MomentumUpdater",
                  client->Args(grad_vec, lr_vec, momentum_vec, use_nesterov_vec,
                          write_num, write_interval),
                  cb);
        }
      break;
    case VarType::kHash128:
    case VarType::kHash64:
      client->HashPush(
          var_name_,
          convert_indices,
          0.0,
          false,
          "MomentumUpdater",
          client->Args(grad_vec, lr_vec, momentum_vec, use_nesterov_vec), 
          cb);      
      break;
    default:
      XDL_CHECK_COND_ASYNC(
          false, 
          Status::ArgumentError("unsupported vartype"),
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsSparseApplyMomentumOp)
  .Input("learning_rate", DataType::kDouble)
  .Input("momentum", DataType::kDouble)
  .Input("use_nesterov", DataType::kBool)
  .Input("grad", DataType::kFloat)
  .Input("indices", "dtype")
  .Input("write_num", DataType::kInt64)
  .Input("write_interval", DataType::kInt64)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType);

DEFINE_INT_OP(XDL_REGISTER_KERNEL(PsSparseApplyMomentumOp, PsSparseApplyMomentumOp<T>)
              .Device("CPU")
              .AttrDataType<T>("dtype"))

} // namespace xdl


