#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

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

#define COLL_BASE_COMPUTE_BLOCKCOUNT(COUNT, NUM_BLOCKS, SPLIT_INDEX,           \
                                     EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT)      \
  EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                   \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }
int allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                  size_t count, MPI_Comm comm, float eb) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;
  unsigned char *d_cmpSendBytes;
  unsigned char *d_cmpReduceBytes;

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *d_inbuf[2];
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank); // get rank
  MPI_Comm_size(comm, &size); // get size of comm

  size_t cmpSizes[size];
  /* Special case for size == 1 */
  if (1 == size) { // if only one rank, no need to do anything
    return MPI_SUCCESS;
  }

  /* Special case for count less than size - use recursive doubling */
  /*if (count < (size_t)size) { // if count (elements to send) is less than
  size,
                              // use recursive doubling
    return (allreduce_recursivedoubling(sbuf, rbuf, count, dtype, op, comm));
  }*/

  /* Allocate and initialize temporary buffers */
  /* Determine the number of elements per block and corresponding
  block sizes.
  The blocks are divided into "early" and "late" ones:
  blocks 0 .. (split_rank - 1) are "early" and
  blocks (split_rank) .. (size - 1) are "late".
  Early blocks are at most 1 element larger than the late ones.
  */
  // compute the block with +1 element respect to the other
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank, early_segcount,
                               late_segcount);
  max_segcount = early_segcount;
  max_real_segsize = max_segcount * sizeof(float);

  cudaMalloc((void **)&d_cmpSendBytes, max_segcount * sizeof(float));
  cudaMalloc((void **)&d_quant_predData, max_segcount * sizeof(int));

  cudaMalloc((void **)&d_inbuf[0], max_real_segsize);
  if (size > 2) {
    cudaMalloc((void **)&d_inbuf[1], max_real_segsize);
  }
  // TODO : check if d_sbuf == MPI_IN_PLACE

  cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float),
             cudaMemcpyDeviceToDevice); // copy data from send buffer to receive
                                        // buffer
  /* Computation loop */

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;
  block_offset =
      ((rank < split_rank)
           ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
           : ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  /* Initialize first receive from the neighbor on the left */
  GSZ_compress_deviceptr_outlier((d_rbuf + block_offset), d_cmpSendBytes,
                                 block_count, &cmpSize, eb,
                                 rank); // compress data

  MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm,
                           &reqs[inbi])); // recv compressed data
  MPI_call_check(MPI_Send(d_cmpSendBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset = ((prevblock < split_rank)
                        ? ((ptrdiff_t)prevblock * early_segcount)
                        : ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    dim3 grid(gsize);
    dim3 block(bsize);
    // d_rbuf + (ptrdiff_t)block_offset * extent;
    kernel_quant_prediction<<<grid, block>>>(d_rbuf + block_offset,
                                             d_quant_predData, eb, block_count);
    /* Post irecv for the current block */
    MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from, 0, comm,
              &reqs[inbi]); // recv compressed data

    /* Wait on previous block to arrive */
    MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);

    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, &cmpSize);
    MPI_Barrier(comm);

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }

  MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)
                      ? ((ptrdiff_t)recv_from * early_segcount)
                      : ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  kernel_quant_prediction<<<grid, block>>>(d_rbuf + block_offset,
                                           d_quant_predData, eb, block_count);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_cmpReduceBytes,
                  block_count, eb, &cmpSize);

  cmpSizes[rank] = cmpSize;
  MPI_Alltoall(cmpSizes, 1, MPI_UNSIGNED_LONG_LONG, cmpSizes, 1,
               MPI_UNSIGNED_LONG_LONG, comm);
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

    MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from, 0, comm,
              &reqs[inbi]);

    MPI_Send(d_cmpReduceBytes, cmpSizes[send_to], MPI_BYTE, send_to, 0, comm);

    MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);

    GSZ_decompress_deviceptr_outlier(d_rbuf + recv_block_offset * sizeof(float),
                                     d_inbuf[inbi], (size_t)block_count,
                                     cmpSize, eb);
    cudaMemcpy(d_cmpReduceBytes, d_inbuf[inbi], cmpSize,
               cudaMemcpyDeviceToDevice);

    inbi = inbi ^ 0x1;
  }
  return 0;
}