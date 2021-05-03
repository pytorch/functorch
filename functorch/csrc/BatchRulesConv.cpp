#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

// batching rules translated from jax: https://github.com/google/jax/blob/master/jax/_src/lax/lax.py#L3143
Tensor reshape_axis_into(int64_t src, int64_t dst, const Tensor &x) {
  auto new_shape = x.sizes().vec();
  new_shape.erase(new_shape.begin() + src);
  new_shape[dst] *= x.sizes()[src];
  return at::reshape(x.movedim(dst, src), new_shape);
}

Tensor reshape_axis_outof(int64_t src, int64_t size1, const Tensor &x) {
  auto shape = x.sizes().vec();
  TORCH_INTERNAL_ASSERT(shape[src] % size1 == 0);
  int64_t size2 = shape[src] / size1;
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);
  return at::reshape(x, shape);
}

// Does not support batch_group_count (needed for convolution backwards)
std::tuple<Tensor,optional<int64_t>>
conv2d_batching_rule(const Tensor& lhs, optional<int64_t> lhs_bdim, const Tensor& rhs, optional<int64_t> rhs_bdim, const optional<Tensor>& bias, optional<int64_t> bias_bdim, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  std::vector<int64_t> lhs_spec = {0,1,2,3};
  std::vector<int64_t> rhs_spec = {0,1,2,3};
  std::vector<int64_t> out_spec = {0,1,2,3};
  optional<Tensor> new_bias;
  if (bias_bdim) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    new_bias = nullopt;
  } else {
    TORCH_INTERNAL_ASSERT(!bias.has_value());
    new_bias = bias;
  }
  std::tuple<Tensor, optional<int64_t>> result;
  if (lhs_bdim && !rhs_bdim) {
    auto new_x = reshape_axis_into(*lhs_bdim, lhs_spec[0], lhs);
    auto out = at::conv2d(new_x, rhs, new_bias, stride, padding, dilation, groups);
    out = reshape_axis_outof(out_spec[0], lhs.sizes()[*lhs_bdim], out);
    result = {out, out_spec[0]};
  } else if (!lhs_bdim && rhs_bdim) {
    if (groups == 1) {
      auto new_w = reshape_axis_into(*rhs_bdim, rhs_spec[0], rhs);
      auto out = at::conv2d(lhs, new_w, new_bias, stride, padding, dilation, groups);
      out = reshape_axis_outof(out_spec[1], rhs.sizes()[*rhs_bdim], out);
      result =  {out, out_spec[1]};
    } else {
      auto new_w = reshape_axis_outof(rhs_spec[0] + (*rhs_bdim <= rhs_spec[0]), groups, rhs);
      new_w = reshape_axis_into(*rhs_bdim + (rhs_spec[0] < rhs_bdim), rhs_spec[0] + 1, new_w);
      new_w = reshape_axis_into(rhs_spec[0], rhs_spec[0], new_w);
      auto out = at::conv2d(lhs, new_w, new_bias, stride, padding, dilation, groups);
      out = reshape_axis_outof(out_spec[1], groups, out);
      out = reshape_axis_outof(out_spec[1] + 1, rhs.sizes()[*rhs_bdim], out);
      out = reshape_axis_into(out_spec[1], out_spec[1] + 1, out);
      result = {out, out_spec[1]};
    }
  } else if (lhs_bdim && rhs_bdim) {
    auto new_x = reshape_axis_into(*lhs_bdim, lhs_spec[1], lhs);
    groups *= lhs.sizes()[*lhs_bdim];
    auto new_w = reshape_axis_into(*rhs_bdim, rhs_spec[0], rhs);
    auto out = at::conv2d(new_x, new_w, new_bias, stride, padding, dilation, groups);
    out = reshape_axis_outof(out_spec[1], lhs.sizes()[*lhs_bdim], out);
    result = {out, out_spec[1]};
  } else {
    result = {at::conv2d(lhs, rhs, new_bias, stride, padding, dilation, groups), nullopt};
  }
  if (new_bias) {
    auto out_dim = std::get<1>(result);
    TORCH_INTERNAL_ASSERT(out_dim, "Batching bias without weights not currently implemented");
    auto bias_tensor = new_bias->movedim(*bias_bdim, *out_dim);
    return {std::get<0>(result) + *new_bias, out_dim};
  } else {
    return result;
  }
}


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  // VMAP_SUPPORT("conv2d", conv2d_batching_rule);
}
}}

