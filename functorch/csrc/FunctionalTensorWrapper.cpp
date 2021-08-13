
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/FunctionalTensorWrapper.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

namespace at {

FunctionalTensorWrapper::FunctionalTensorWrapper(Tensor value, int64_t level)
  : FunctionalTensorImplBase(value.dtype(), value.device()),
    value_(value),
    level_(level)
{
  TORCH_INTERNAL_ASSERT(value_.defined());
}

void FunctionalTensorWrapper::replace_(const Tensor& other) {
    auto self_impl = value_.unsafeGetTensorImpl();
    auto other_functional = dynamic_cast<FunctionalTensorWrapper*>(other.unsafeGetTensorImpl());
    // new invariant: every time the fucntionalization pass redispatches during functionalize() calls,
    // we'll hit the DynamicLayerModeBackFallback which should wrap outputs in a FunctionalTensorWrapper
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_functional != nullptr);
    auto other_impl = other_functional->value().unsafeGetTensorImpl();
    if (typeid(*self_impl) == typeid(*other_impl)) {
        // It is valid to swap out the metadata on the tensorImpl
        // but we can only do that if the two tensor's we're swapping have the same type.
        // This allows us to ensure that programs that mutate their inputs
        // preserve their semantics under a functionalization pass.
        self_impl->replace_(other_impl);
    } else {
        value_ = other_functional->value();
    }
}

void FunctionalTensorWrapper::set_size(int64_t dim, int64_t new_size) {
    value_.unsafeGetTensorImpl()->set_size(dim, new_size);
}
void FunctionalTensorWrapper::set_stride(int64_t dim, int64_t new_stride) {
    value_.unsafeGetTensorImpl()->set_stride(dim, new_stride);
}
void FunctionalTensorWrapper::set_storage_offset(int64_t storage_offset) {
    value_.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
}
bool FunctionalTensorWrapper::has_storage() const {
    return value_.unsafeGetTensorImpl()->has_storage();
}
IntArrayRef FunctionalTensorWrapper::sizes() const {
    return value_.unsafeGetTensorImpl()->sizes();
}
int64_t FunctionalTensorWrapper::dim() const {
    return value_.unsafeGetTensorImpl()->dim();
}
const Storage& FunctionalTensorWrapper::storage() const {
    return value_.unsafeGetTensorImpl()->storage();
}
int64_t FunctionalTensorWrapper::numel() const {
    return value_.unsafeGetTensorImpl()->numel();
}
bool FunctionalTensorWrapper::is_contiguous(at::MemoryFormat memory_format) const {
    return value_.unsafeGetTensorImpl()->is_contiguous(memory_format);
}
int64_t FunctionalTensorWrapper::storage_offset() const {
    return value_.unsafeGetTensorImpl()->storage_offset();
}
int64_t FunctionalTensorWrapper::size(int64_t d) const {
    return value_.unsafeGetTensorImpl()->size(d);
}
int64_t FunctionalTensorWrapper::stride(int64_t d) const {
    return value_.unsafeGetTensorImpl()->stride(d);
}
c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
    // TODO: maybe just don't allow this
    return value_.unsafeGetTensorImpl()->shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
}
c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
    // TODO: maybe just don't allow this
    return value_.unsafeGetTensorImpl()->shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
}
void FunctionalTensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
    // TODO: maybe just don't allow this
    value_.unsafeGetTensorImpl()->shallow_copy_from(impl);
}
const char* FunctionalTensorWrapper::tensorimpl_type_name() const {
    return "FunctionalTensorWrapper";
}

namespace functionalization {
namespace impl {

Tensor makeFunctional(const Tensor& tensor, int64_t level) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!dynamic_cast<FunctionalTensorWrapper*>(tensor.unsafeGetTensorImpl()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!tensor.key_set().has(c10::DispatchKey::Functionalize));
  return at::detail::make_tensor<FunctionalTensorWrapper>(tensor, level);
}

c10::optional<Tensor> makeFunctional(const c10::optional<Tensor>& tensor, int64_t level) {
  if (tensor.has_value()) {
    return makeFunctional(*tensor, level);
  }
  return c10::nullopt;
}

c10::List<Tensor> makeFunctional(const c10::List<Tensor>& t_list, int64_t level) {
  std::vector<at::Tensor> functional_tensors;
  for (auto& t: t_list.vec()) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return c10::List<at::Tensor>(functional_tensors);
}

std::vector<Tensor> makeFunctional(const at::TensorList t_list, int64_t level) {
  std::vector<at::Tensor> functional_tensors;
  for (auto& t: t_list) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return functional_tensors;
}

c10::List<c10::optional<Tensor>> makeFunctional(const c10::List<c10::optional<Tensor>>& t_list, int64_t level) {
  std::vector<c10::optional<at::Tensor>> functional_tensors;
  for (auto& t: t_list.vec()) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return c10::List<c10::optional<at::Tensor>>(functional_tensors);
}

} // namespace impl
} // namespace functionalization
} // namespace at
