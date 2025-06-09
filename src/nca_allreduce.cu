#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* TODO : cuda Stream to overlap compression and quant prediction*/

#define MPI_call_check(call)                                                   \
  {                                                                            \
    int err_code = call;                                                       \
    if (err_code != MPI_SUCCESS) {                                             \
      char error_string[BUFSIZ];                                               \
      int length_of_error_string;                                              \
      MPI_Error_string(err_code, error_string, &length_of_error_string);       \
      fprintf(stderr, "\nMPI error in line %d : %s\n", __LINE__,               \
              error_string);                                                   \
      fflush(stderr);                                                          \
      MPI_Abort(MPI_COMM_WORLD, err_code);                                     \
    }                                                                          \
  }
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define COLL_BASE_COMPUTE_BLOCKCOUNT(COUNT, NUM_BLOCKS, SPLIT_INDEX,           \
                                     EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT)      \
  EARLY_BLOCK_COUNT = COUNT / NUM_BLOCKS;                                      \
  LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                                       \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }

int cpuCopy_allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                          size_t count, MPI_Comm comm,
                                          float eb) {
  int rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;
  unsigned char *cmpSendBytes;
  unsigned char *cmpReduceBytes;
  unsigned char *d_cmpSendBytes;
  unsigned char *d_cmpReduceBytes;

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *inbuf[2];

  unsigned char *d_tmpbuf;
  float *d_rtmpbuf;
  ptrdiff_t block_offset_elements;
  size_t max_real_segsize_bytes;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;
  int count_;

  if (1 == size) {

    CUDA_CHECK(cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return MPI_SUCCESS;
  }
  size_t pad_nbEle = (count + 32768 - 1) / 32768 * 32768;
  COLL_BASE_COMPUTE_BLOCKCOUNT(pad_nbEle, size, split_rank, early_segcount,
                               late_segcount);
  early_segcount = (early_segcount % 4 == 0)
                       ? early_segcount
                       : early_segcount + (4 - early_segcount % 4);
  late_segcount = (late_segcount % 4 == 0)
                      ? late_segcount
                      : late_segcount + (4 - late_segcount % 4);
  max_segcount = early_segcount;
  max_real_segsize_bytes = max_segcount * sizeof(float);

  // Alloca memoria host allineata
  /*
  if (posix_memalign((void **)&cmpSendBytes, ALIGNMENT,
                     max_real_segsize_bytes) != 0) {
    fprintf(stderr, "Error allocating aligned host memory.\n");
    exit(EXIT_FAILURE);
  }
  if (posix_memalign((void **)&cmpReduceBytes, ALIGNMENT,
                     max_real_segsize_bytes) != 0) {
    fprintf(stderr, "Error allocating aligned host memory.\n");
    exit(EXIT_FAILURE);
  }
  if (posix_memalign((void **)&inbuf[0], ALIGNMENT, max_real_segsize_bytes) !=
      0) {
    fprintf(stderr, "Error allocating aligned host memory.\n");
    exit(EXIT_FAILURE);
  }
  if (size > 2) {
    if (posix_memalign((void **)&inbuf[1], ALIGNMENT, max_real_segsize_bytes) !=
        0) {
      fprintf(stderr, "Error allocating aligned host memory.\n");
      exit(EXIT_FAILURE);
    }
  }*/
  cmpSendBytes = (unsigned char *)malloc(max_real_segsize_bytes);
  if (cmpSendBytes == NULL) {
    fprintf(stderr, "Error allocating host memory for cmpSendBytes.\n");
    exit(EXIT_FAILURE);
  }
  cmpReduceBytes = (unsigned char *)malloc(max_real_segsize_bytes);
  if (cmpReduceBytes == NULL) {
    fprintf(stderr, "Error allocating host memory for cmpReduceBytes.\n");
    exit(EXIT_FAILURE);
  }
  inbuf[0] = (unsigned char *)malloc(max_real_segsize_bytes);
  if (inbuf[0] == NULL) {
    fprintf(stderr, "Error allocating host memory for inbuf[0].\n");
    exit(EXIT_FAILURE);
  }
  if (size > 2) {
    inbuf[1] = (unsigned char *)malloc(max_real_segsize_bytes);
    if (inbuf[1] == NULL) {
      fprintf(stderr, "Error allocating host memory for inbuf[1].\n");
      exit(EXIT_FAILURE);
    }
  }

  size_t padded_count_elements = (size_t)split_rank * early_segcount +
                                 (size_t)(size - split_rank) * late_segcount;
  size_t padded_count_bytes = padded_count_elements * sizeof(float);

  CUDA_CHECK(cudaMalloc((void **)&d_rtmpbuf, padded_count_bytes));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpSendBytes, max_segcount * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_quant_predData, max_segcount * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, max_segcount * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_tmpbuf, max_real_segsize_bytes));

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;

  if (rank < split_rank) {
    block_offset_elements = (ptrdiff_t)rank * early_segcount;
    block_count = early_segcount;
  } else {
    block_offset_elements = (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)rank - split_rank) * late_segcount;
    block_count = late_segcount;
  }

  float *d_rbuf_ = d_rtmpbuf + block_offset_elements;

  GSZ_compress_deviceptr_outlier(d_rbuf_, d_cmpSendBytes, block_count, &cmpSize,
                                 eb);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(cmpSendBytes, d_cmpSendBytes, cmpSize + (cmpSize * 0.1),
                        cudaMemcpyDeviceToHost));

  MPI_call_check(MPI_Irecv(inbuf[inbi], max_real_segsize_bytes, MPI_BYTE,
                           recv_from, 0, comm, &reqs[inbi]));
  MPI_call_check(MPI_Send(cmpSendBytes, cmpSize + (cmpSize * 0.1), MPI_BYTE,
                          send_to, 0, comm));

  for (k = 2; k < size; k++) {

    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;

    if (prevblock < split_rank) {
      block_offset_elements = (ptrdiff_t)prevblock * early_segcount;
      block_count = early_segcount;
    } else {
      block_offset_elements =
          (ptrdiff_t)split_rank * early_segcount +
          ((ptrdiff_t)prevblock - split_rank) * late_segcount;
      block_count = late_segcount;
    }

    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    dim3 grid(gsize);
    dim3 block(bsize);
    d_rbuf_ = d_rtmpbuf + block_offset_elements;
    kernel_quant_prediction<<<grid, block>>>(d_rbuf_, d_quant_predData, eb,
                                             block_count);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(MPI_Irecv(inbuf[inbi], max_real_segsize_bytes, MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));
    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;
    CUDA_CHECK(cudaMemcpy(d_tmpbuf, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));

    homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count,
                    eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes,
                          cmpSize + (cmpSize * 0.1), cudaMemcpyDeviceToHost));

    MPI_call_check(MPI_Send(cmpReduceBytes, cmpSize + (cmpSize * 0.1), MPI_BYTE,
                            send_to, 0, comm));
  }

  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_call_check(MPI_Get_count(&status, MPI_BYTE, &count_));
  cmpSize = (size_t)count_;
  CUDA_CHECK(
      cudaMemcpy(d_tmpbuf, inbuf[inbi], cmpSize, cudaMemcpyHostToDevice));

  recv_from = (rank + 1) % size;
  if (recv_from < split_rank) {
    block_offset_elements = (ptrdiff_t)recv_from * early_segcount;
    block_count = early_segcount;
  } else {
    block_offset_elements = (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)recv_from - split_rank) * late_segcount;
    block_count = late_segcount;
  }
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  d_rbuf_ = d_rtmpbuf + block_offset_elements;
  kernel_quant_prediction<<<grid, block>>>(d_rbuf_, d_quant_predData, eb,
                                           block_count);
  CUDA_CHECK(cudaGetLastError());
  homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count, eb,
                  &cmpSize);
  GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                   d_cmpReduceBytes, block_count, cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  cmpSize = cmpSize + (cmpSize * 0.1);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    if (recv_data_from < split_rank) {
      block_offset_elements = (ptrdiff_t)recv_data_from * early_segcount;
      block_count = early_segcount;
    } else {
      block_offset_elements =
          (ptrdiff_t)split_rank * early_segcount +
          ((ptrdiff_t)recv_data_from - split_rank) * late_segcount;
      block_count = late_segcount;
    }
    CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes, cmpSize,
                          cudaMemcpyDeviceToHost));
    MPI_call_check(MPI_Sendrecv(cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0,
                                inbuf[inbi], max_real_segsize_bytes, MPI_BYTE,
                                recv_from, 0, comm, &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;
    CUDA_CHECK(cudaMemcpy(d_cmpReduceBytes, inbuf[inbi], cmpSize,
                          cudaMemcpyHostToDevice));

    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                     d_cmpReduceBytes, (size_t)block_count,
                                     cmpSize, eb);
  }

  CUDA_CHECK(cudaMemcpy(d_rbuf, d_rtmpbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  free(cmpSendBytes);
  free(cmpReduceBytes);
  free(inbuf[0]);
  if (size > 2) {
    free(inbuf[1]);
  }

  CUDA_CHECK(cudaFree(d_rtmpbuf));
  CUDA_CHECK(cudaFree(d_cmpSendBytes));
  CUDA_CHECK(cudaFree(d_quant_predData));
  CUDA_CHECK(cudaFree(d_cmpReduceBytes));
  CUDA_CHECK(cudaFree(d_tmpbuf));

  return 0;
}