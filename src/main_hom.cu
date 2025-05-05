#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/GSZ_timer.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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

void write_datai(const char *filename, int *data, size_t dim) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Err");
    return;
  }

  for (size_t i = 0; i < dim; i++) {
    fprintf(file, "%d\n", data[i]);
  }

  fclose(file);
}

int main() {
  size_t nbEle;
  float *vec = read_data("smooth.in", &nbEle);
  float *vec_local = read_data("smooth.in", &nbEle);
  float eb = 1e-4;
  unsigned char *cmpBytes = NULL;
  float *decData;

  cmpBytes = (unsigned char *)malloc(nbEle * sizeof(float));
  decData = (float *)malloc(nbEle * sizeof(float));

  float *d_decData;
  float *d_vec;
  unsigned char *d_cmpBytes;

  float max_val = vec[0];
  float min_val = vec[0];
  for (size_t i = 0; i < nbEle; i++) {
    if (vec[i] > max_val)
      max_val = vec[i];
    else if (vec[i] < min_val)
      min_val = vec[i];
  }
  eb = eb * (max_val - min_val);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMalloc((void **)&d_vec, nbEle * sizeof(float));
  cudaMemcpy(d_vec, vec, nbEle * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_decData, nbEle * sizeof(float));
  cudaMalloc((void **)&d_cmpBytes, nbEle * sizeof(float));

  size_t cmpSize;

  // Homomorphic compression simulation:
  /* 1. Compress the data (we've to send data to another device, but for the
   purpose of testing, we can do all operation on the same device)
    2. Apply prediction + quantization to data local to the device
    3. Apply homomoprhic sum kernel
    4. Decompress the data and write it to the output file
  */
  GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, nbEle, &cmpSize, eb, 0,
                                 stream);
  printf("cmpSize = %zu\n", cmpSize);
  float *d_localData;
  int *d_quantLocOut;

  cudaMalloc((void **)&d_localData, nbEle * sizeof(float));
  cudaMemcpy(d_localData, vec_local, nbEle * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_quantLocOut, nbEle * sizeof(int));

  int bsize = dec_tblock_size;
  int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);

  kernel_quant_prediction<<<grid, block>>>(d_localData, d_quantLocOut, eb,
                                           nbEle, 0);

  unsigned char *d_cmpBytesOut;
  cudaMalloc((void **)&d_cmpBytesOut, nbEle * sizeof(float));

  size_t cmpSize2;

  homomorphic_sum(d_cmpBytes, d_quantLocOut, d_cmpBytesOut, nbEle, 0, eb,
                  &cmpSize2, stream);
  printf("cmpSize2 = %zu\n", cmpSize2);
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytesOut, nbEle, cmpSize2,
                                   eb, stream);
  kernel_quant_prediction<<<grid, block>>>(d_decData, d_quantLocOut, eb, nbEle,
                                           0);
  homomorphic_sum(d_cmpBytes, d_quantLocOut, d_cmpBytesOut, nbEle, 0, eb,
                  &cmpSize2);
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytesOut, nbEle, cmpSize2,
                                   eb, stream);
  cudaMemcpy(decData, d_decData, nbEle * sizeof(float), cudaMemcpyDeviceToHost);

  write_dataf("output", decData, nbEle);

  int not_bound = 0;
  for (size_t i = 0; i < nbEle; i += 1) {
    if (fabs(vec[i] * 2 - decData[i]) > eb * 2.2) {
      not_bound++;
    }
  }
  if (!not_bound)
    printf("\033[0;32mPass error check!\033[0m\n");
  else
    printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n",
           not_bound);
  cudaFree(d_cmpBytes);
  cudaFree(d_cmpBytesOut);
  cudaFree(d_decData);
  cudaFree(d_vec);
  cudaFree(d_localData);
  cudaFree(d_quantLocOut);
  free(cmpBytes);
  free(vec);
  free(vec_local);
  free(decData);
  cudaStreamDestroy(stream);
  return 0;
}