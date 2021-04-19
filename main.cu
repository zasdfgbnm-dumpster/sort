#include <iostream>
#include <chrono>
#include <functional>
#include <cstdlib>

#include <cub/cub.cuh>

using namespace std::chrono;

template<typename key_t, typename value_t>
using func_t = void(const key_t *, key_t *, const value_t *, value_t *, size_t);

size_t max_temp_storage_bytes = 1024 * 1024 * 1024;
void *tmp_storage = nullptr;

template<typename key_t, typename value_t>
void check_correctness(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
}

template<typename key_t, typename value_t>
void cub_sort(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
  size_t temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out , N);
  assert(temp_storage_bytes <= max_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(tmp_storage, temp_storage_bytes, keys_in, keys_out, values_in, values_out , N);
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
  check_correctness(d_keys_in, d_keys_out, d_values_in, d_values_out, N);
  if (name != nullptr) {
    std::cout << "[" << name << "] ";
  }
  std::cout << "problem_size: " << N << ", ";
  std::cout << "cpu time: " << duration_cpu << ", ";
  std::cout << "gpu time: " << duration_gpu << std::endl;
}

int main() {
  cudaMalloc(&tmp_storage, max_temp_storage_bytes);
  benchmark_sort(cub_sort<int64_t, int64_t>, 1024, "cub_sort<int64_t, int64_t>");
}