// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/BatchRulesHelper.h>

namespace at {
namespace functorch {

template <typename F, F Func, typename... ExtraArgs>
Tensor random_batching_rule(IntArrayRef shape, ExtraArgs... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    VmapDimVector shapeVec(shape.begin(), shape.end());
    shapeVec.insert(shapeVec.begin(), maybe_layer->batchSize());
    if (maybe_layer->useBatchedRandom()) {
      return makeBatched(Func(shapeVec, std::forward<ExtraArgs>(extra_args)...), 0, maybe_layer->layerId());
    } else {
      const auto res = Func(shape, std::forward<ExtraArgs>(extra_args)...);
      return makeBatched(res.unsqueeze(0).expand(shapeVec), 0, maybe_layer->layerId());
    }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor& random_inplace_batching_rule(Tensor& self, ExtraArgs... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    const auto cur_level = maybe_layer->layerId();
    Tensor self_value;
    optional<int64_t> self_bdim;
    std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
    self_value = moveBatchDimToFront(self_value, self_bdim);
    if (maybe_layer->useBatchedRandom() || !self_bdim) {
      Func(self_value, std::forward<ExtraArgs>(extra_args)...);
      return self;
    } else {
      auto intermediate = empty(self.sizes(), self.options());
      Func(intermediate, std::forward<ExtraArgs>(extra_args)...);
      self.copy_(intermediate); // batching should make this just work out...
      return self;
    }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor randint_batching_rule(int64_t high, IntArrayRef shape, ExtraArgs... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    VmapDimVector shapeVec(shape.begin(), shape.end());
    shapeVec.insert(shapeVec.begin(), maybe_layer->batchSize());
    if (maybe_layer->useBatchedRandom()) {
      return makeBatched(Func(high, shapeVec, std::forward<ExtraArgs>(extra_args)...), 0, maybe_layer->layerId());
    } else {
      const auto res = Func(high, shape, std::forward<ExtraArgs>(extra_args)...);
      return makeBatched(res.unsqueeze(0).expand(shapeVec), 0, maybe_layer->layerId());
    }
}

template <typename F, F Func, typename T0, typename T1, typename... ExtraArgs>
Tensor rand_two_leading_scalars_batching_rule(T0 scalar0, T1 scalar1, IntArrayRef shape, ExtraArgs... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    VmapDimVector shapeVec(shape.begin(), shape.end());
    shapeVec.insert(shapeVec.begin(), maybe_layer->batchSize());
    if (maybe_layer->useBatchedRandom()) {
      return makeBatched(Func(scalar0, scalar1, shapeVec, std::forward<ExtraArgs>(extra_args)...), 0, maybe_layer->layerId());
    } else {
      const auto res = Func(scalar0, scalar1, shape, std::forward<ExtraArgs>(extra_args)...);
      return makeBatched(res.unsqueeze(0).expand(shapeVec), 0, maybe_layer->layerId());
    }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor randperm_batching_rule(int64_t n, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  auto const batch_size = maybe_layer->batchSize();
  if (maybe_layer->useBatchedRandom()) {
    std::vector<at::Tensor> stackedList(batch_size);
    stackedList.reserve(batch_size);
    for (int64_t idx = 0; idx < batch_size; ++idx) {
      // since this is done in a loop, need to pass by reference for generator to update
      stackedList[idx] = Func(n, extra_args...);
    }
    return makeBatched(at::stack(stackedList), 0, maybe_layer->layerId());
  } else {
    const auto res = Func(n, std::forward<ExtraArgs>(extra_args)...);
    return makeBatched(res.unsqueeze(0).expand({batch_size, n}), 0, maybe_layer->layerId());
  }
}

template <typename A, A a, typename C>
struct RandomBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandomBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor apply(IntArrayRef shape, T... extra_args) {
    return random_batching_rule<F, Func, T...>(shape, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandomInplaceBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandomInplaceBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor& apply(Tensor& self, T... extra_args) {
    return random_inplace_batching_rule<F, Func, T...>(self, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandIntBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct RandIntBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static Tensor apply(int64_t high, IntArrayRef shape, T... extra_args) {
    return randint_batching_rule<F, Func, T...>(high, shape, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandTwoLeadingScalarsBatchRuleHelper;

template <typename F, F Func, typename T0, typename T1, typename T2, typename... T>
struct RandTwoLeadingScalarsBatchRuleHelper<F, Func, typelist<T0, T1, T2, T...>> {
  static Tensor apply(T0 scalar0, T1 scalar1, IntArrayRef shape, T... extra_args) {
    return rand_two_leading_scalars_batching_rule<F, Func, T0, T1, T...>(scalar0, scalar1, shape, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandpermBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandpermBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor apply(int64_t n, T... extra_args) {
    return randperm_batching_rule<F, Func, T...>(n, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct UnaryBatchRuleLeadingFloatHelper;

template <typename F, F Func, typename A0, typename A1, typename... T>
struct UnaryBatchRuleLeadingFloatHelper<F, Func, typelist<A0, A1, T...>> {
  static std::tuple<Tensor, optional<int64_t>> apply(
      double scalar,
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    return std::make_tuple(Func(scalar, tensor, std::forward<T>(extra_args)...), batch_dim);
  }
};

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define UNARY_POINTWISE_LEADING_FLOAT(op, overload) \
    VMAP_SUPPORT(#op"."#overload, SINGLE_ARG(\
      UnaryBatchRuleLeadingFloatHelper<\
        decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload),\
        c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  VMAP_SUPPORT("normal.Tensor_float", BASIC_UNARY_BATCH_RULE(ATEN_FN2(normal, Tensor_float)));
  UNARY_POINTWISE_LEADING_FLOAT(normal, float_Tensor);
  // normal Tensor_Tensor needs binary batching rule so it's in that file

  #undef UNARY_POINTWISE_LEADING_FLOAT
}

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  #define RANDOM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDOM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))
  
  #define RANDOM_INPLACE_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define RANDINT_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                             c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDINT_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define RAND_TWO_LEADING_SCALARS_BATCH_RULE(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandTwoLeadingScalarsBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))
  #define RANDPERM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDPERM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  RANDOM_BATCH_RULE(randn);
  RANDOM_BATCH_RULE2(randn, generator);
  RANDOM_BATCH_RULE2(randn, generator_with_names);
  RANDOM_BATCH_RULE2(randn, names);

  RANDOM_BATCH_RULE(rand);
  RANDOM_BATCH_RULE2(rand, generator);
  RANDOM_BATCH_RULE2(rand, generator_with_names);
  RANDOM_BATCH_RULE2(rand, names);

  RANDOM_INPLACE_BATCH_RULE(random_);
  RANDOM_INPLACE_BATCH_RULE2(random_, from);
  RANDOM_INPLACE_BATCH_RULE2(random_, to);

  RANDINT_BATCH_RULE(randint);
  RANDINT_BATCH_RULE2(randint, generator);
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low);
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low_generator);

  RANDPERM_BATCH_RULE(randperm);
  RANDPERM_BATCH_RULE2(randperm, generator);

  RANDOM_INPLACE_BATCH_RULE(normal_);
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(normal, float_float);

  #undef RANDOM_BATCH_RULE
  #undef RANDOM_BATCH_RULE2
  #undef RANDOM_INPLACE_BATCH_RULE
  #undef RANDOM_INPLACE_BATCH_RULE2
  #undef RANDINT_BATCH_RULE
  #undef RANDINT_BATCH_RULE2
  #undef RANDINT_LOW_BATCH_RULE
  #undef RANDPERM_BATCH_RULE2
}
}} // namespace at::functorch