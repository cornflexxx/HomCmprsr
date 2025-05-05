#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

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
  EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                   \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }

int cpuCopy_allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                          size_t count, MPI_Comm comm,
                                          float eb) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;

  unsigned char *cmpSendBytes;
  unsigned char *cmpReduceBytes;

  unsigned char *d_cmpSendBytes;
  unsigned char *d_cmpReduceBytes;

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *inbuf[2];

  unsigned char *d_tmpbuf;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank); // get rank
  MPI_Comm_size(comm, &size); // get size of comm

  MPI_Status status;
  int count_;

  if (1 == size) {
    return MPI_SUCCESS;
  }

  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank, early_segcount,
                               late_segcount);
  max_segcount = early_segcount;
  max_real_segsize = max_segcount * sizeof(float);

  cmpSendBytes = (unsigned char *)malloc(max_real_segsize);
  cmpReduceBytes = (unsigned char *)malloc(max_real_segsize);

  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpSendBytes, max_segcount * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_quant_predData, max_segcount * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, max_segcount * sizeof(float)));

  inbuf[0] = (unsigned char *)malloc(max_real_segsize);
  if (size > 2) {
    inbuf[1] = (unsigned char *)malloc(max_real_segsize);
  }

  CUDA_CHECK(cudaMalloc((void **)&d_tmpbuf, max_real_segsize));
  CUDA_CHECK(cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;

  memset(inbuf[inbi], 0, max_real_segsize);
  block_offset =
      ((rank < split_rank)
           ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
           : ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  float *d_rbuf_ = d_rbuf + block_offset;
  GSZ_compress_deviceptr_outlier(d_rbuf_, d_cmpSendBytes, block_count, &cmpSize,
                                 eb, rank);
  CUDA_CHECK((cudaGetLastError()));
  CUDA_CHECK(cudaMemcpy(cmpSendBytes, d_cmpSendBytes,
                        cmpSize + (size_t)(cmpSize * 0.1),
                        cudaMemcpyDeviceToHost));

  MPI_call_check(MPI_Irecv(inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm, &reqs[inbi]));
  MPI_call_check(MPI_Send(cmpSendBytes, cmpSize + (size_t)(cmpSize * 0.1),
                          MPI_BYTE, send_to, 0, comm));

  for (k = 2; k < size; k++) {

    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;

    memset(inbuf[inbi], 0, max_real_segsize);
    block_offset = ((prevblock < split_rank)
                        ? ((ptrdiff_t)prevblock * early_segcount)
                        : ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    dim3 grid(gsize);
    dim3 block(bsize);
    d_rbuf_ = d_rbuf + block_offset;
    kernel_quant_prediction<<<grid, block>>>(d_rbuf_, d_quant_predData, eb,
                                             block_count, rank);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(MPI_Irecv(inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                             0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = count_;
    CUDA_CHECK(cudaMemcpy(d_tmpbuf, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));

    homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count,
                    rank, eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes,
                          cmpSize + (size_t)(cmpSize * 0.1),
                          cudaMemcpyDeviceToHost));

    MPI_call_check(MPI_Send(cmpReduceBytes, cmpSize + (size_t)(cmpSize * 0.1),
                            MPI_BYTE, send_to, 0, comm));
  }

  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_call_check(MPI_Get_count(&status, MPI_BYTE, &count_));
  cmpSize = count_;
  CUDA_CHECK(
      cudaMemcpy(d_tmpbuf, inbuf[inbi], cmpSize, cudaMemcpyHostToDevice));
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)
                      ? ((ptrdiff_t)recv_from * early_segcount)
                      : ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  d_rbuf_ = d_rbuf + block_offset;
  kernel_quant_prediction<<<grid, block>>>(d_rbuf_, d_quant_predData, eb,
                                           block_count, rank);
  CUDA_CHECK(cudaGetLastError());
  homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count,
                  rank, eb, &cmpSize);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(d_rbuf_, d_cmpReduceBytes,
                                   (size_t)block_count, cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  // REDUCE SCATTER
  //  TODO : ALL_GATHER
  /*
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    for (k = 0; k < size - 1; k++) {
      const int recv_data_from = (rank + size - k) % size;
      const ptrdiff_t recv_block_offset =
          ((recv_data_from < split_rank)
               ? ((ptrdiff_t)recv_data_from * early_segcount)
               : ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
      block_count =
          ((recv_data_from < split_rank) ? early_segcount : late_segcount);

      MPI_call_check(MPI_Sendrecv(d_cmpReduceBytes, max_real_segsize, MPI_BYTE,
                                  send_to, 0, d_inbuf[inbi], max_real_segsize,
                                  MPI_BYTE, recv_from, 0, comm, &status));

      MPI_Get_count(&status, MPI_BYTE, &count_);
      cmpSize = count_;
      GSZ_decompress_deviceptr_outlier(d_rbuf + recv_block_offset,
    d_inbuf[inbi], (size_t)block_count, cmpSize, eb);
      cudaMemcpy(d_cmpReduceBytes, d_inbuf[inbi], max_real_segsize,
                 cudaMemcpyDeviceToDevice);
    }*/
  return 0;
}