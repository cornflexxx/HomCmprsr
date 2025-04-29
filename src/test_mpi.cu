#include <cuda.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void compute_sqrt(float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = sqrtf(input[idx]);
  }
}

int main(int argc, char **argv) {
  int rank, size;
  const int N = 1024;
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  CUDA_CHECK(cudaSetDevice(rank % deviceCount));

  // Allocazione e inizializzazione dell'array host
  float *h_input = (float *)malloc(N * sizeof(float));
  float *h_output = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    h_input[i] = (float)(i + 1);
  }

  // Allocazione della memoria sul dispositivo
  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

  compute_sqrt<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("Processo %d: primi 10 risultati:\n", rank);
  for (int i = 0; i < 10; ++i) {
    printf("sqrt(%f) = %f\n", h_input[i], h_output[i]);
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_input);
  free(h_output);

  MPI_Finalize();
  return 0;
}
