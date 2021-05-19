#pragma once
#include <c10/core/impl/LocalDispatchKeySet.h>


namespace at {
namespace functorch {
struct VmapMode {
  // Returns the vmap level, aka the count of how many nested vmaps we're in.
  static int64_t current_vmap_level();

  // Increment the count of nested vmaps. If this causes the vmap level to be
  // greater than 0, then it enables DispatchKey::VmapMode on all tensors.
  static int64_t increment_nesting();

  // Decrements the count of nested vmaps. If this causes the vmap level to be
  // equal to 0, then it disables DispatchKey::VmapMode on all tensors.
  static int64_t decrement_nesting();
};


}
} // namespace at

