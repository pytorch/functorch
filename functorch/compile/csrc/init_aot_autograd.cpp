#include <torch/extension.h>
#include <functorch/compile/csrc/CompileCache.h>

namespace at {
namespace compile {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  at::functorch::initCompileCacheBindings(m.ptr());
}
}}