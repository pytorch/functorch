
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/ArrayRef.h>
#include <ATen/core/List.h>
#include <ATen/FunctionalTensorImplBase.h>

namespace at {

struct TORCH_API FunctionalTensorWrapper : public at::FunctionalTensorImplBase {
  explicit FunctionalTensorWrapper(Tensor value, int64_t level);

  const Tensor& value() const { return value_; };
  int64_t level() const { return level_; };

  // Override the FunctionalTensorImplBase method describing how to re-use a tensor in the functionalization pass.
  //void replace_(const Tensor& other) override;
  void replace_(const TensorImpl* other_impl) override;

  // Override ALL virtual functions on the TensorImpl to call into the wrapped value's implementation
  IntArrayRef sizes() const override;
  int64_t dim() const override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t numel() const override;
  bool is_contiguous(at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const override;
  //bool is_contiguous_custom(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const override;
  int64_t storage_offset() const override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

 private:
  const char* tensorimpl_type_name() const override;

  Tensor value_;
  int64_t level_;
};

TORCH_API inline FunctionalTensorWrapper* unsafeGetFunctionalWrapper(const Tensor& tensor) {
  auto functional_impl = static_cast<FunctionalTensorWrapper*>(tensor.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  return functional_impl;
}

namespace functionalization {
namespace impl {

// Utility functions for the functionalization pass.

TORCH_API Tensor makeFunctional(const Tensor& tensor, int64_t level);
TORCH_API c10::optional<Tensor> makeFunctional(const c10::optional<Tensor>& tensor);
TORCH_API c10::List<Tensor> makeFunctional(const c10::List<Tensor>& t_list);
TORCH_API std::vector<Tensor> makeFunctional(const TensorList t_list);
TORCH_API c10::List<c10::optional<Tensor>> makeFunctional(const c10::List<c10::optional<Tensor>>& tensor);

} // namespace impl
} // namespace functionalization
} // namespace at
