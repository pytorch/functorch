// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <ATen/Operators.h>

// NB: most activation functions are pointwise unary or binary. These are the ones that have special batch rules
namespace at { namespace functorch {
std::tuple<Tensor,optional<int64_t>> prelu_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim) {
  if (!weight_bdim && weight.dim() == 0) {
    return std::make_tuple(at::prelu(input, weight), input_bdim);
  }

  const auto input_ = moveBatchDimToFront(input, input_bdim);
  auto weight_flatten = moveBatchDimToFront(weight, weight_bdim);

  if (weight_flatten.dim() > 1) {
    // for an input [N, C, ...]
    // weight can be a non-vector but the total number of elements must be the same as C
    weight_flatten = at::flatten(weight_flatten, weight_bdim.has_value() ? 1 : 0, -1);
  }

  const int64_t input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  VmapDimVector new_shape(weight_flatten.sizes().begin(), weight_flatten.sizes().end());
  const int64_t final_size = weight_bdim ? (input_logical_rank + 1) : input_logical_rank;
  new_shape.reserve(final_size);

  if (weight_flatten.dim() == 2 || !weight_bdim) {
    // if weight (without batching) is not a scalar, its size must match the "channel dimension" of input. To do the
    // decomposition, we pad the weight to

    // copies checks from prelu if the weight (without vmap) is not a scalar
    TORCH_CHECK(input_logical_rank > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    if (input_logical_rank > 1) {
      const auto channel_dim = input_bdim ? 2 : 1;
      channel_size = input_.size(channel_dim);
    }
    const auto weight_num = weight_flatten.size(-1);
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    // pads to the left so that the flattened shape matches up with the channel
    if (!weight_bdim) {
      new_shape.insert(new_shape.begin(), 1); 
    } else {
      new_shape.insert(new_shape.begin() + 1, 1);
    }
  }

  for (int64_t i = new_shape.size(); i < final_size; i ++) {
    new_shape.push_back(1);
  }
  TORCH_INTERNAL_ASSERT(new_shape.size() == final_size);
  const auto weight_padded = weight_flatten.view(new_shape);
  auto zero_tensor = at::zeros(1, input.options());

  // decomposes function, 
  auto res = at::maximum(zero_tensor, input_) + weight_padded * at::minimum(zero_tensor, input_);
  return std::make_tuple(res, 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT(prelu, prelu_batch_rule)
}
}} // namespace at::functorch
