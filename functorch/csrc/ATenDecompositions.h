#pragma once

#include <ATen/ATen.h>

namespace at { namespace functorch {

// TODO: Figure out how to delete all of these and replace with
// with the "official" decompositions that are written in Python.

Tensor _log_softmax_backward_data_decomp(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  auto result = grad_output - at::exp(output) * at::sum(grad_output, dim, /*keepdim=*/true);
  return result.to(input_dtype);
}

Tensor _softmax_backward_data_decomp(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  auto new_grad = grad_output * output;
  auto result = (new_grad - output * at::sum(new_grad, dim, /*keepdim*/true));
  return result.to(input_dtype);
}

Tensor mse_loss_backward_decomp(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
  return norm * (input - target) * grad_output;
}

Tensor l1_loss_backward_decomp(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto sign = at::sign(self - target);
  auto norm = reduction == Reduction::Mean ? sign / self.numel() : sign;
  return grad_output * norm;
}

Tensor binary_cross_entropy_backward_decomp(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const optional<Tensor>& weight,
    int64_t reduction) {
  auto EPSILON = 1e-12;
  auto result = grad_output * (self - target) / at::clamp_min(self * (1 - self), EPSILON);
  if (weight.has_value() && weight->defined()) {
    result = result * weight.value();
  }
  if (reduction == Reduction::Mean) {
    result = result / self.numel();
  }
  return result;
}

}} // namespace at::functorch
