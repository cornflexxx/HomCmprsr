#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/GSZ_timer.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

int main(int argc, char *argv[]) {
  // Read input information.

  float errorBound = 0.001;

  // For measuring the end-to-end throughput.
  TimingGPU timer_GPU;
  // Input data preparation on CPU.
  float *oriData = NULL;
  float *decData = NULL;
  unsigned char *cmpBytes = NULL;
  size_t nbEle;
  size_t cmpSize = 0;
  oriData = read_data("smooth.in", &nbEle);
  decData = (float *)malloc(nbEle * sizeof(float));
  cmpBytes = (unsigned char *)malloc(nbEle * sizeof(float));

  /* Yafan added for RTM Project. CAN BE REMOVED*/
  // Get value range, making it a REL errMode test.
  float max_val = oriData[0];
  float min_val = oriData[0];
  for (size_t i = 0; i < nbEle; i++) {
    if (oriData[i] > max_val)
      max_val = oriData[i];
    else if (oriData[i] < min_val)
      min_val = oriData[i];
  }
  errorBound = errorBound * (max_val - min_val);

  // Input data preparation on GPU.
  float *d_oriData;
  float *d_decData;
  unsigned char *d_cmpBytes;
  size_t pad_nbEle =
      (nbEle + 32768 - 1) / 32768 *
      32768; // A temp demo, will add more block sizes in future implementation.
  cudaMalloc((void **)&d_oriData, sizeof(float) * pad_nbEle);
  cudaMemcpy(d_oriData, oriData, sizeof(float) * pad_nbEle,
             cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_decData, sizeof(float) * pad_nbEle);
  cudaMalloc((void **)&d_cmpBytes, sizeof(float) * pad_nbEle);

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Warmup for NVIDIA GPU.
  for (int i = 0; i < 3; i++) {
    GSZ_compress_deviceptr_outlier(d_oriData, d_cmpBytes, nbEle, &cmpSize,
                                   errorBound, stream);
  }

  // GSZ compression.
  timer_GPU.StartCounter(); // set timer
  GSZ_compress_deviceptr_outlier(d_oriData, d_cmpBytes, nbEle, &cmpSize,
                                 errorBound, stream);
  float cmpTime = timer_GPU.GetCounter();

  // GSZ decompression.
  timer_GPU.StartCounter(); // set timer
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytes, nbEle, cmpSize,
                                   errorBound, stream);
  float decTime = timer_GPU.GetCounter();

  // Print result.
  printf("GSZ finished!\n");
  printf("GSZ compression   end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / cmpTime);
  printf("GSZ decompression end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / decTime);
  printf("GSZ compression ratio: %f\n\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) /
             (cmpSize * sizeof(unsigned char) / 1024.0 / 1024.0));

  // Error check
  cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(decData, d_decData, sizeof(float) * nbEle, cudaMemcpyDeviceToHost);
  int not_bound = 0;
  write_dataf("outCmp", decData, nbEle);
  for (size_t i = 0; i < nbEle; i += 1) {
    if (fabs(oriData[i] - decData[i]) > errorBound * 1.1) {
      not_bound++;
      // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound:
      // %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]),
      // errorBound);
    }
  }
  if (!not_bound)
    printf("\033[0;32mPass error check!\033[0m\n");
  else
    printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n",
           not_bound);

  free(oriData);
  free(decData);
  free(cmpBytes);
  cudaFree(d_oriData);
  cudaFree(d_decData);
  cudaFree(d_cmpBytes);
  cudaStreamDestroy(stream);
  return 0;
}