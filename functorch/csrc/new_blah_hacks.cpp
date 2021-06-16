// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/Constants.h>
#include <functorch/csrc/functorch.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/custom_function.h>
#include <c10/core/ScalarTypeToTypeMeta.h>

namespace at { namespace functorch {

#define NEW_BLAH_HACK(new_blah) \
  static Tensor new_blah##_hack( \
      const Tensor& self, \
      IntArrayRef size, \
      c10::optional<ScalarType> dtype, \
      c10::optional<Layout> layout, \
      c10::optional<Device> device, \
      c10::optional<bool> pin_memory \
      ) { \
    static auto op = c10::Dispatcher::singleton() \
      .findSchemaOrThrow("functorch::"#new_blah"_hack", "") \
      .typed<decltype(new_blah##_hack)>(); \
    return op.call(self, size, dtype, layout, device, pin_memory); \
  } \
  static Tensor new_blah##_hack_impl( \
      const Tensor& self, \
      IntArrayRef size, \
      c10::optional<ScalarType> dtype, \
      c10::optional<Layout> layout, \
      c10::optional<Device> device, \
      c10::optional<bool> pin_memory \
      ) { \
    auto layer = maybeCurrentDynamicLayer(); \
    if (!layer.has_value()) { \
      return self.new_blah(size, dtype, layout, device, pin_memory); \
    } \
    AutoNonVariableTypeMode dispatch_after_grad_guard; \
    c10::impl::ExcludeDispatchKeyGuard dispatch_after_vmap_guard(kBatchedKey); \
    return new_blah##_hack(self, size, dtype, layout, device, pin_memory); \
  }

NEW_BLAH_HACK(new_zeros);
NEW_BLAH_HACK(new_empty);
NEW_BLAH_HACK(new_ones);

#undef NEW_BLAH_HACK

Tensor contiguous_decomp(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  return self.clone(memory_format);
}

namespace prim {

Tensor to_kernel(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
  ) {
  // c10::impl::ExcludeDispatchKeyGuard guard(kDynamicLayerFrontModeKey);
  // return self.to(dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format);
  return self.to(dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format);
}

using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

struct ToFunction : public torch::autograd::Function<ToFunction> {
  static Variable forward(AutogradContext* ctx, Variable x,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format
  ) {
    ctx->save_for_backward({x});
    AutoNonVariableTypeMode dispatch_after_grad_guard;
    return to(x, dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format);
    // return to(x, dtype, layout, device, pin_memory, non_blocking, copy);
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad) {
    auto x = ctx->get_saved_variables()[0];
    return { to(
        grad[0], x.scalar_type(),
        x.options().layout_opt(), x.options().device_opt(),
        x.options().pinned_memory_opt(), false, true, nullopt),
				{}, {}, {}, {}, {}, {}, {},
		};
  }
};

Tensor to_autograd(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
  ) {
  // c10::impl::ExcludeDispatchKeyGuard guard(kDynamicLayerFrontModeKey);
  // return self.to(dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format);
  return ToFunction::apply(self, dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format);
}

Tensor to(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
  ) {

  // static auto op = c10::Dispatcher::singleton()
  auto op = c10::Dispatcher::singleton() \
    .findSchemaOrThrow("functorch::to", "") \
    .typed<decltype(to_autograd)>(); \
  return op.call(self, dtype, layout, device, pin_memory, non_blocking, copy, optional_memory_format); \
}


}

Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

Tensor to_device_decomp(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  device = ensure_has_index(device);
  return prim::to(
      self, dtype, self.options().layout_opt(), device, self.options().pinned_memory_opt(),
      non_blocking, copy, optional_memory_format);
}

Tensor to_dtype_decomp(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return prim::to(
      self, dtype, self.options().layout_opt(), self.options().device_opt(),
      self.options().pinned_memory_opt(),
      non_blocking, copy, optional_memory_format);
}

// Tensor to_other_decomp(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
//   return prim::to(
//       self, dtype, self.options().layout_opt(), self.options().device_opt(),
//       self.options().pinned_memory_opt(),
//       non_blocking, copy, optional_memory_format);
// }


TORCH_LIBRARY(functorch, m) {
  m.def("new_zeros_hack", new_zeros_hack_impl);
  m.def("new_empty_hack", new_empty_hack_impl);
  m.def("new_ones_hack", new_ones_hack_impl);
  m.def("to", prim::to_autograd);
}

TORCH_LIBRARY_IMPL(aten, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  // m.impl("new_zeros", new_zeros_hack);
  // m.impl("new_empty", new_empty_hack);
  // m.impl("new_ones", new_ones_hack);
  m.impl("contiguous", contiguous_decomp);
  m.impl("to.device", to_device_decomp);
  m.impl("to.dtype", to_dtype_decomp);
  m.impl("to.dtype_layout", prim::to);
}

TORCH_LIBRARY_IMPL(functorch, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  m.impl("to", torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}
TORCH_LIBRARY_IMPL(functorch, Autograd, m) {
  m.impl("to", prim::to_autograd);
}
TORCH_LIBRARY_IMPL(functorch, CPU, m) {
  m.impl("to", prim::to_kernel);
}
TORCH_LIBRARY_IMPL(functorch, CUDA, m) {
  m.impl("to", prim::to_kernel);
}

// TORCH_LIBRARY_IMPL(aten, DispatchKey::Autograd, m) {
//   m.impl("to.device", to_device_decomp);
//   m.impl("to.dtype", to_dtype_decomp);
//   m.impl("to.dtype_layout", prim::to);
//   m.impl("to", prim::to);
// }

}}

