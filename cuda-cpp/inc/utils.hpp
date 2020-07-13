#pragma once

#include <cuda.h>

namespace nsparse {

template <typename T>
using allocator = backend::allocator<T>;

// Rebind Allocator<U> as type T
template <typename Allocator, typename T>
using rebind_allocator_t = typename Allocator::rebind<T>::other;

template <typename T, typename Allocator>
T* allocate_with(size_t size) {
  return rebind_allocator_t<Allocator, T>{}.allocate(size);
}

template <typename Allocator, typename T>
void deallocate_with(T* ptr) {
  return rebind_allocator_t<Allocator, T>{}.deallocate(ptr);
}

template <typename T>
class cuda_allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using is_always_equal = std::true_type;

  template <class U> struct rebind {
    typedef cuda_allocator<U> other;
  };

  cuda_allocator() = default;
  cuda_allocator(const cuda_allocator&) = default;

  pointer allocate(size_type n) {
    T* ptr;
    cudaMalloc(&ptr, n*sizeof(value_type));
    if (ptr == nullptr) {
      throw std::bad_alloc();
    } else {
      return ptr;
    }
  }

  void deallocate(pointer ptr, size_type n = 0) {
    cudaFree(ptr);
  }

  template<typename... Args>
  void construct(pointer ptr, Args&&... args) {
    new(ptr) T(std::forward<Args>(args)...);
  }

  void destroy(pointer ptr) {
    ptr->~T();
  }

  bool operator==(const cuda_allocator&) const {
    return true;
  }

  bool operator!=(const cuda_allocator& other) const {
    return !operator==(other);
  }
};

} // ending nsparse