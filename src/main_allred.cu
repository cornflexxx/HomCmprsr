#include "../include/AWcudaAllreduce.cuh"
#include <cstddef>
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

float *read_data(const char *filename, size_t *dim) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Err");
    return NULL;
  }

  size_t sz = 1000;
  *dim = 0;
  float *vec = (float *)malloc(sz * sizeof(float));
  if (!vec) {
    perror("mem allocation failed");
    fclose(file);
    return NULL;
  }

  char row[100];

  while (fgets(row, sizeof(row), file)) {
    if (*dim >= sz) {
      sz *= 2;
      float *temp = (float *)realloc(vec, sz * sizeof(float));
      if (!temp) {
        perror("mem allocation failed");
        free(vec);
        fclose(file);
        return NULL;
      }
      vec = temp;
    }
    vec[*dim] = strtof(row, NULL);
    (*dim)++;
  }

  fclose(file);
  return vec;
}

void write_dataf(const char *filename, float *data, size_t dim) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Err");
    return;
  }

  for (size_t i = 0; i < dim; i++) {
    fprintf(file, "%f\n", data[i]);
  }

  fclose(file);
}
int main() {
  MPI_Init(NULL, NULL);
  size_t count;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int deviceCount;
  CUDA_CHECK(cudaSetDevice(0));

  float *h_sbuf;
  h_sbuf = read_data("smooth.in", &count);
  float *h_rbuf = (float *)malloc(count * sizeof(float));
  float *d_sbuf, *d_rbuf;
  cudaMalloc((void **)&d_sbuf, count * sizeof(float));
  cudaMemcpy(d_sbuf, h_sbuf, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_rbuf, count * sizeof(float));

  float eb = 0.0001;
  allreduce_ring_comprs_hom_sum(d_sbuf, d_rbuf, count, MPI_COMM_WORLD, eb);
  cudaMemcpy(h_rbuf, d_rbuf, count * sizeof(float), cudaMemcpyDeviceToHost);
  if (rank == 0) {
    write_dataf("smooth.out", h_rbuf, count);
  }
  cudaFree(d_sbuf);
  cudaFree(d_rbuf);
  MPI_Finalize();
}