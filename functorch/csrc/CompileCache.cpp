// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

///
/// This design stemmed of from the PointwiseOperatorCompileCache with the
/// purpose of making it more generic for AOTAutograd. This is Compile Cache
/// allowing different types of hashing functions, and is agnostic of the
/// compiler.
///
#include <functorch/csrc/CompileCache.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/utils/pybind.h>

using namespace torch::jit::tensorexpr;

namespace {
/// Record of thread-local state that changes operator behavior.
struct LocalState {
  c10::impl::LocalDispatchKeySet dispatchModifier;
  bool gradModeEnabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatchModifier.included_) - dispatchModifier.excluded_;
  }

  LocalState()
      : dispatchModifier(c10::impl::tls_local_dispatch_key_set()),
        gradModeEnabled(at::GradMode::is_enabled()) {}
};

/// Helper to pack tensor (dtype, requires grad) into an 8-bit key.
static uint8_t packFlags(const LocalState &state, const at::Tensor &v) {
  static_assert(static_cast<int>(at::ScalarType::NumOptions) < 128,
                "overflow possible");
  at::ScalarType dtype = v.dtype().toScalarType();
  bool requires_grad = state.gradModeEnabled && v.requires_grad();
  return static_cast<uint8_t>(requires_grad) |
         (static_cast<uint8_t>(dtype) << 1);
}

/// Per-tensor cache specialization key targetting dynamic shapes. Records
/// dtype, dispatch options, aliasing, and per-dim contiguity/broadcasting
/// information.

struct DynamicArgSpecializationKey {
  /// Default constructor; does no initialization, use only for
  /// declarations, e.g., std::array.
  DynamicArgSpecializationKey() {} // NOLINT: intentionally not initialized

  /// Construct a specialization key from a given TLS state and
  /// Tensor.
  // NOLINTNEXTLINE: intentionally not initializing dimflags_
  DynamicArgSpecializationKey(const LocalState &state, const at::Tensor &v,
                              int8_t aliasGroup)
      : flags_(packFlags(state, v)), aliasGroup_(aliasGroup),
        dispatchKey_(state.apply(v.key_set()).raw_repr()),
        nDims_(v.ndimension()) {
    initDimflags(v.sizes(), v.strides());
  }

  // TODO (anijain) - Code seems expensive for each comparison. Revisit if cache
  // latency is bad.
  bool operator<(const DynamicArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return this_tie < other_tie;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      if (dimflags_[dim] != other.dimflags_[dim]) {
        return dimflags_[dim] < other.dimflags_[dim];
      }
    }
    return false;
  }

  bool operator==(const DynamicArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return false;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      if (dimflags_[dim] != other.dimflags_[dim]) {
        return false;
      }
    }
    return true;
  }

  /// Get the dispatch key for this specialization.
  at::DispatchKeySet dispatchKey() const {
    return at::DispatchKeySet(at::DispatchKeySet::RAW, dispatchKey_);
  }

  std::string to_string() {
    std::string hash = "";
    hash += std::to_string(flags_);
    hash += std::to_string(aliasGroup_);
    hash += std::to_string(dispatchKey_);
    hash += std::to_string(nDims_);
    for (auto dimFlag : dimflags_) {
      hash += std::to_string(dimFlag);
    }
    return hash;
  }

private:
  /// Flag bits indicating tensor shape properties like contiguity and
  /// broadcasting that are relevant for codegen.
  enum DimFlags {
    /// A leading dimension implicitly added by broadcasting.
    SIZE_MISSING = 1 << 0,

    /// Size == 1.
    SIZE_ONE = 1 << 1,

    /// Size > 1.
    SIZE_OTHER = 1 << 2,

    /// Stride == 0; broadcasting.
    STRIDE_ZERO = 1 << 3,

    /// Stride == 1; packed contiguously in memory.
    STRIDE_ONE = 1 << 4,

    /// Stride = Stride[i + 1] * Size[i + 1].
    /// Used to collapse dimensions.
    STRIDE_CONTIGUOUS = 1 << 5,

    /// Stride = Stride[i - 1] * Size[i - 1].
    /// Used to collapse dimensions in the other direction.
    STRIDE_TRANSPOSED_CONTIGUOUS = 1 << 6, // stride[i-1] * sizes[i-1]

    /// Stride must be provided as an argument.
    STRIDE_AS_ARG = 1 << 7,
  };

  /// Initialize the shape flags for each dimension.
  void initDimflags(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
    // Pack all the properties for each dimension into a uint8.
    dimflags_.reserve(nDims_);
    for (int64_t dim = 0; dim < nDims_; ++dim) {
      uint8_t flag =
          (sizes[dim] == 0 ? SIZE_MISSING
                           : (sizes[dim] == 1 ? SIZE_ONE : SIZE_OTHER));
      if (strides[dim] == 0) {
        flag |= STRIDE_ZERO;
      } else if (strides[dim] == 1) {
        flag |= STRIDE_ONE;
      } else if (dim + 1 < (int64_t)sizes.size() &&
                 strides[dim] == strides[dim + 1] * sizes[dim + 1]) {
        flag |= STRIDE_CONTIGUOUS;
      } else if (dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1] &&
                 (dimflags_[dim - 1] & STRIDE_CONTIGUOUS) == 0) {
        flag |= STRIDE_TRANSPOSED_CONTIGUOUS;
      } else {
        flag |= STRIDE_AS_ARG;
      }
      dimflags_.push_back(flag);
    }
  }

private:
  /// Packed field with requires_grad and dtype.
  uint8_t flags_;

  /// Bits indicating whether there is aliasing in this group.
  /// 0 = no aliasing
  /// >0 = same data, strides, and shapes within group
  /// <0 = overlapping storage madness
  int8_t aliasGroup_;

  /// DispatchKeySet includes device/layout.
  uint64_t dispatchKey_;

  /// Saving the number of dimensions
  int nDims_;

  /// Per-dimension shape flags.
  // NOLINTNEXTLINE: C-style arrays
  std::vector<uint8_t> dimflags_;
};

/// Per-tensor cache specialization key targetting static shapes. Recordsdtype,
/// dispatch options, aliasing, and full shapes and strides.
struct StaticArgSpecializationKey {
  /// Default constructor; does no initialization, use only for
  /// declarations, e.g., std::array.
  StaticArgSpecializationKey() {} // NOLINT: intentionally not initialized

  StaticArgSpecializationKey(const LocalState &state, const at::Tensor &v,
                             int8_t aliasGroup)
      : flags_(packFlags(state, v)), aliasGroup_(aliasGroup),
        dispatchKey_(state.apply(v.key_set()).raw_repr()),
        nDims_(v.ndimension()) {
    for (int dim = 0; dim < nDims_; dim++) {
      shapes_.push_back(v.sizes()[dim]);
      strides_.push_back(v.strides()[dim]);
    }
  }

  // TODO (anijain) - Code seems expensive for each comparison. Revisit if cache
  // latency is bad.
  bool operator<(const StaticArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return this_tie < other_tie;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      auto this_tie = std::tie(shapes_[dim], strides_[dim]);
      auto other_tie = std::tie(other.shapes_[dim], other.strides_[dim]);
      if (this_tie != other_tie) {
        return this_tie < other_tie;
      }
    }
    return false;
  }

  bool operator==(const StaticArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return false;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      auto this_tie = std::tie(shapes_[dim], strides_[dim]);
      auto other_tie = std::tie(other.shapes_[dim], other.strides_[dim]);
      if (this_tie != other_tie) {
        return false;
      }
    }
    return true;
  }

  std::string to_string() {
    std::string hash = "";
    hash += std::to_string(flags_);
    hash += std::to_string(aliasGroup_);
    hash += std::to_string(dispatchKey_);
    hash += std::to_string(nDims_);
    for (auto shape : shapes_) {
      hash += std::to_string(shape);
    }
    for (auto stride : strides_) {
      hash += std::to_string(stride);
    }
    return hash;
  }

private:
  /// Packed field with requires_grad and dtype.
  uint8_t flags_;

  /// Bits indicating whether there is aliasing in this group.
  /// 0 = no aliasing
  /// >0 = same data, strides, and shapes within group
  /// <0 = overlapping storage madness
  int8_t aliasGroup_;

  /// DispatchKeySet includes device/layout.
  uint64_t dispatchKey_;

  /// Saving the number of dimensions
  int nDims_;

  /// Record all tensor shapes.
  std::vector<uint64_t> shapes_;

  /// Record all tensor strides.
  std::vector<uint64_t> strides_;
};

/// This is the base class for recording Arg or Tensor propoerties. To create a
/// new Compile cache, we can inherit from this base class and record the
/// properties we are interested in.
struct ArgCompileCacheBase {
  /// Destructor.
  virtual ~ArgCompileCacheBase() = default;

  /// Check if a key (computed from args and kwargs) is present in the cache.
  virtual py::object at(PyObject *args, PyObject *kwargs) = 0;

  /// Inserts a new compiled_function for given args.
  virtual void insert(const py::object &compileFn, PyObject *args,
                      PyObject *kwargs) = 0;

  /// Get name of kernel.
  virtual const std::string &getName() const = 0;

  /// Get the size of the cache. Helps in counting the number of recompilations.
  virtual const int64_t size() const = 0;

  /// Clear the cache.
  virtual void clear() = 0;
};

/// ArgCompileCache is a templated class allowing plugging of different types of
/// Hasher/Specialization Keys.
struct CompileCache {
public:
  CompileCache() = default;
  ~CompileCache() = default;

  /// Array defining groups of aliased tensors.
  using AliasGroups = std::vector<int8_t>;

  /// Cache type mapping specialization keys to compiled kernels.
  using Cache = std::unordered_map<std::string, py::object>;

  /// Compute aliasing relationships between tensors a and b.
  /// 0 means a/b don't alias.
  /// 1 means a/b alias and are the same.
  /// -1 means a/b have crazy aliasing overlaps.
  int8_t computeAliasing(const at::Tensor &a, const at::Tensor &b) {
    if (a.is_alias_of(b)) {
      if (a.is_set_to(b)) {
        return 1;
      } else {
        // TODO(jansel): check for non-overlapping and return 0 in cases where
        // we can prove no aliasing. Possibly could take some logic from
        // tensoriterator.
        return -1;
      }
    } else {
      return 0;
    }
  }

  /// Compute aliasing groups: group of tensors that alias each other.
  AliasGroups computeAliasGroups(std::vector<at::Tensor> args, int numArgs) {
    AliasGroups aliasGroups;
    int8_t currentId = 0;
    for (int i = 0; i < numArgs; ++i) {
      aliasGroups.push_back(0);
    }
    for (int i = 0; i < numArgs; ++i) {
      if (aliasGroups[i] == 0) {
        for (int j = i + 1; j < numArgs; ++j) {
          int8_t alias_type = computeAliasing(args[i], args[j]);
          if (alias_type != 0) {
            if (aliasGroups[i] == 0)
              ++currentId;
            aliasGroups[i] = currentId;
            aliasGroups[j] = currentId * alias_type;
          }
        }
      }
    }
    return aliasGroups;
  }

  /// Compute the set of specialization keys based on the inputs to
  /// the kernel.
  std::string computeCacheKey(std::vector<at::Tensor> args, int numArgs,
                              std::string hasherType, int id) {
    LocalState state;
    AliasGroups aliasGroups = computeAliasGroups(args, numArgs);
    std::string cacheKey;
    for (int i = 0; i < numArgs; ++i) {
      if (hasherType == "StaticShapeHasher") {
        cacheKey += StaticArgSpecializationKey(state, args[i], aliasGroups[i])
                        .to_string();
      } else if (hasherType == "DynamicShapeHasher") {
        cacheKey += DynamicArgSpecializationKey(state, args[i], aliasGroups[i])
                        .to_string();
      }
    }
    cacheKey += std::to_string(id);
    cacheKey += std::to_string(numArgs);
    return cacheKey;
  }

  std::vector<std::string> buildSignature(int numArgs) {
    std::string signature = "(";
    for (int i = 0; i < numArgs; i++) {
      signature += "Tensor inp" + std::to_string(i) + ", ";
    }
    signature += "*)";
    std::vector<std::string> result;
    result.push_back(signature);
    return result;
  }

  template <int N>
  std::vector<at::Tensor> parsingN(const std::vector<std::string> &signatures,
                                   PyObject *args, PyObject *kwargs) {
    torch::PythonArgParser parser(signatures);
    torch::ParsedArgs<N> parsed_args;
    torch::PythonArgs r = parser.parse(args, kwargs, parsed_args);
    std::vector<at::Tensor> tensorArgs; // NOLINT: c-style arrays
    for (int i = 0; i < N; ++i) {
      tensorArgs.push_back(r.tensor(i));
    }
    return tensorArgs;
  }

  std::vector<at::Tensor> parsing(int numArgs, PyObject *args,
                                  PyObject *kwargs) {

    const std::vector<std::string> signatures = buildSignature(numArgs);
    switch (numArgs) {
    case 1:
      return parsingN<1>(signatures, args, kwargs);
    case 2:
      return parsingN<2>(signatures, args, kwargs);
    case 3:
      return parsingN<3>(signatures, args, kwargs);
    case 4:
      return parsingN<4>(signatures, args, kwargs);
    case 5:
      return parsingN<5>(signatures, args, kwargs);
    case 6:
      return parsingN<6>(signatures, args, kwargs);
    case 7:
      return parsingN<7>(signatures, args, kwargs);
    case 8:
      return parsingN<8>(signatures, args, kwargs);
    default:
      throw std::runtime_error("TODO: support other arg counts");
    }
  }

  /// Check if the function has already been compiled.
  // py::object at(int64_t id, int numArgs, PyObject *args, PyObject *kwargs) {
  py::object at(int64_t id, int numArgs, const std::string hasherType,
                PyObject *args, PyObject *kwargs) {
    std::vector<at::Tensor> tensorArgs = parsing(numArgs, args, kwargs);
    LocalState state;
    std::string cacheKey = computeCacheKey(tensorArgs, numArgs, hasherType, id);

    auto item = cache_.find(cacheKey); // protected by GIL

    if (C10_LIKELY(item != cache_.end())) {
      auto c = cache_.at(cacheKey);
      return c;
    }
    return py::none();
  }

  /// Insert a new compiled functions for new tensor properties.
  void insert(int64_t id, int numArgs, const std::string hasherType,
              const py::object &compileFn, PyObject *args, PyObject *kwargs) {
    std::vector<at::Tensor> tensorArgs = parsing(numArgs, args, kwargs);
    LocalState state;
    std::string cacheKey = computeCacheKey(tensorArgs, numArgs, hasherType, id);
    cache_.emplace(cacheKey, compileFn);
  }

  const int64_t size() const { return cache_.size(); }

  /// Clear the cache.
  void clear() { cache_.clear(); }

private:
  /// Compilation cache holding key and the compiled function.
  Cache cache_;
};

static CompileCache *createCompileCache() { return new CompileCache(); }

} // namespace

namespace at {
namespace functorch {

// TODO(anijain) - Add static compilation cache
void initCompileCacheBindings(PyObject *module) {
  py::handle te(module);
  py::class_<CompileCache>(te, "CompileCache")
      .def(py::init(&createCompileCache))
      .def("at",
           [](CompileCache &self, int64_t id, int numArgs,
              const std::string hasherType, py::args args, py::kwargs kwargs) {
             return self.at(id, numArgs, hasherType, args.ptr(), kwargs.ptr());
           })
      .def("insert",
           [](CompileCache &self, int64_t id, int numArgs,
              const std::string hasherType, const py::object &compileFn,
              py::args args, py::kwargs kwargs) {
             self.insert(id, numArgs, hasherType, compileFn, args.ptr(),
                         kwargs.ptr());
           })
      .def("clear", [](CompileCache &self) { self.clear(); })
      .def("size", [](CompileCache &self) { return self.size(); });
}

} // namespace functorch
} // namespace at
