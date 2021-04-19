#include <iostream>
#include <chrono>
#include <functional>
#include <cstdlib>

#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std::chrono;

template<typename key_t, typename value_t>
using func_t = void(const key_t *, key_t *, const value_t *, value_t *, size_t);

size_t max_temp_storage_bytes = 1024 * 1024 * 1024;
void *tmp_storage = nullptr;

class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::size_t size) {
    assert(size + offset < max_temp_storage_bytes);
    auto ret = static_cast<char*>(tmp_storage) + offset;
    offset += size;
    return ret;
  }

  void deallocate(char* p, size_t size) {}
private:
  size_t offset = 0;
};

template<typename key_t, typename value_t>
bool check_correctness(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
  return true;
}

template<typename key_t, typename value_t>
void cub_sort(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
  size_t temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out , N);
  assert(temp_storage_bytes <= max_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(tmp_storage, temp_storage_bytes, keys_in, keys_out, values_in, values_out , N);
}

template<typename key_t, typename value_t>
void thrust_sort(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
  ThrustAllocator thrust_allocator;
  auto policy = thrust::cuda::par(thrust_allocator).on(0);
  thrust::copy(policy, keys_in, keys_in + N, keys_out);
  thrust::copy(policy, values_in, values_in + N, values_out);
  thrust::sort_by_key(policy, keys_out, keys_out + N, values_out, []__device__(key_t a, key_t b) { return a < b; });
}

template<typename key_t, typename value_t>
void benchmark_sort(func_t<key_t, value_t> *f, size_t N, const char *name=nullptr) {
  constexpr size_t ntrials = 50;
  key_t *h_keys, *d_keys_in, *d_keys_out;
  value_t *h_values, *d_values_in, *d_values_out;
  h_keys = new key_t[N];
  h_values = new value_t[N];
  for(size_t i = 0; i < N; i++) {
    h_keys[i] = std::rand();
    h_values[i] = std::rand();
  }
  cudaMalloc(&d_keys_in, sizeof(key_t) * N);
  cudaMalloc(&d_keys_out, sizeof(key_t) * N);
  cudaMalloc(&d_values_in, sizeof(value_t) * N);
  cudaMalloc(&d_values_out, sizeof(value_t) * N);
  cudaMemcpy(d_keys_in, h_keys, sizeof(key_t) * N, cudaMemcpyDefault);
  cudaMemcpy(d_values_in, h_values, sizeof(value_t) * N, cudaMemcpyDefault);
  f(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
  cudaDeviceSynchronize();
  auto start = high_resolution_clock::now();
  for (int64_t i = 0; i < ntrials; i++) {
    f(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
  }
  auto stop_cpu = high_resolution_clock::now();
  cudaDeviceSynchronize();
  auto stop_gpu = high_resolution_clock::now();
  auto duration_cpu = double(duration_cast<microseconds>(stop_cpu - start).count()) / ntrials;
  auto duration_gpu = double(duration_cast<microseconds>(stop_gpu - start).count()) / ntrials;

  if (name != nullptr) {
    std::cout << "[" << name << "] ";
  }
  if (auto err = cudaGetLastError(); err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    return;
  }
  if (check_correctness(d_keys_in, d_keys_out, d_values_in, d_values_out, N)) {
    std::cout << "problem_size: " << N << ", ";
    std::cout << "cpu time: " << duration_cpu << ", ";
    std::cout << "gpu time: " << duration_gpu << std::endl;
  } else {
    std::cout << "wrong results" << std::endl;
  }
}

int main() {
  cudaMalloc(&tmp_storage, max_temp_storage_bytes);
  benchmark_sort(cub_sort<int64_t, float>, 100, "cub_sort<int64_t, int64_t>");
  benchmark_sort(thrust_sort<int64_t, float>, 100, "thrust_sort<int64_t, int64_t>");
}