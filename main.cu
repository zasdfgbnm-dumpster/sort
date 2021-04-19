#include <iostream>
#include <cub/cub.cuh>
#include <chrono>
#include <functional>

using namespace std::chrono;

template<typename key_t, typename value_t>
using func_t = std::function<void(const key_t *, key_t *, const value_t *, value_t *)>;

template<typename key_t, typename value_t, typename>
void check_correctness(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out) {
}

template<typename key_t, typename value_t, typename>
void cub_sort(const key_t *keys_in, key_t *keys_out, const value_t *values_in, value_t *values_out, size_t N) {
  cub::DeviceRadixSort::SortPairs(tmp, temp_storage_bytes, d1, d2, d3, d4, N);
}

template<typename key_t, typename value_t, typename>
void benchmark_sort(size_t N, func_t f) {
  constexpr size_t ntrials = 50;
  key_t *h1, *d1, *d2;
  value_t *h2, *d3, *d4;
  void *tmp;
  h1 = new type[N];
  h2 = new type[N];
  for(size_t i = 0; i < N; i++) {
    h1[i] = N - i;
  }
  cudaMalloc(&d1, sizeof(key_t) * N);
  cudaMalloc(&d2, sizeof(key_t) * N);
  cudaMalloc(&d3, sizeof(key_t) * N);
  cudaMalloc(&d4, sizeof(key_t) * N);
  cudaMemcpy(d1, h1, sizeof(type) * N, cudaMemcpyDefault);
  size_t temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d1, d2, d3, d4, N);
  cudaMalloc(&tmp, temp_storage_bytes);
  cudaDeviceSynchronize();
  auto start = high_resolution_clock::now();
  for (int i = 0; i < ntrials; i++) {
    cub::DeviceRadixSort::SortPairs(tmp, temp_storage_bytes, d1, d2, d3, d4, N);
  }
  auto stop_cpu = high_resolution_clock::now();
  cudaDeviceSynchronize();
  auto stop_gpu = high_resolution_clock::now();
  auto duration_cpu = double(duration_cast<microseconds>(stop_cpu - start).count())/ntrials;
  auto duration_gpu = double(duration_cast<microseconds>(stop_gpu - start).count())/ntrials;
  std::cout << duration_cpu << std::endl;
  std::cout << duration_gpu << std::endl;
  cudaMemcpy(h2, d2, sizeof(type) * N, cudaMemcpyDefault);
  for(size_t i = 0; i < 10; i++) {
    std::cout << h2[i] << ", ";
  }
}

int main() {
}