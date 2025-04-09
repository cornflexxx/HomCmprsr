#include "../include/GSZ_entry.h"
#include "../include/GSZ_timer.h"
#include <cstddef>
#include <cuda.h>
#include <stdlib.h>

#include <stdio.h>

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

int main() {
  size_t nbEle;
  float *vec = read_data("smooth.in", &nbEle);
  float *d_vec;
  float eb = 1e-4;
  unsigned char *cmpBytes = NULL;
  float *decData;

  cmpBytes = (unsigned char *)malloc(nbEle * sizeof(float));
  TimingGPU timer_GPU;

  decData = (float *)malloc(nbEle * sizeof(float));
  float *d_decData;
  cudaStream_t str;
  unsigned char *d_cmpBytes;
  cudaStreamCreate(&str);

  cudaMalloc((void **)&d_vec, nbEle * sizeof(float));
  cudaMemcpy(&d_vec, &vec, nbEle * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_decData, nbEle * sizeof(float));
  cudaMalloc((void **)&d_cmpBytes, nbEle * sizeof(float));
  size_t cmpSize;

  for (int i = 0; i < 3; i++) {
    GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, nbEle, &cmpSize, eb, str);
  }
  timer_GPU.StartCounter();
  GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, nbEle, &cmpSize, eb, str);
  float cmpTime = timer_GPU.GetCounter();
  cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  timer_GPU.StartCounter();
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytes, nbEle, cmpSize, eb,
                                   str);
  float decTime = timer_GPU.GetCounter();

  printf("GSZ finished!\n");
  printf("GSZ compression   end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / cmpTime);
  printf("GSZ decompression end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / decTime);
  printf("GSZ compression ratio: %f\n\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) /
             (cmpSize * sizeof(unsigned char) / 1024.0 / 1024.0));
  for (int i = 0; i < 3; i++) {
    GSZ_compress_deviceptr_outlier_vec(d_vec, d_cmpBytes, nbEle, &cmpSize, eb,
                                       str);
  }
  timer_GPU.StartCounter();
  GSZ_compress_deviceptr_outlier_vec(d_vec, d_cmpBytes, nbEle, &cmpSize, eb,
                                     str);
  float cmpTime2 = timer_GPU.GetCounter();
  cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  timer_GPU.StartCounter();
  GSZ_decompress_deviceptr_outlier_vec(d_decData, d_cmpBytes, nbEle, cmpSize,
                                       eb, str);
  float decTime2 = timer_GPU.GetCounter();

  printf("GSZ-Vec finished!\n");
  printf("GSZ compression   end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / cmpTime2);
  printf("GSZ decompression end-to-end speed: %f GB/s\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) / decTime2);
  printf("GSZ compression ratio: %f\n\n",
         (nbEle * sizeof(float) / 1024.0 / 1024.0) /
             (cmpSize * sizeof(unsigned char) / 1024.0 / 1024.0));
}