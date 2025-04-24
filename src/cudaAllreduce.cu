#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

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
  void *ret_;
  int bsize, gsize;
  unsigned char *d_cmpSendBytes;
  unsigned char *d_cmpReduceBytes;
  cudaStream_t streamQuanPred;
  cudaStream_t streamDecomprs;
  cudaStream_t streamComprs;
  cudaStreamCreate(&streamQuanPred);
  cudaStreamCreate(&streamComprs);
  cudaStreamCreate(&streamDecomprs);

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *d_inbuf[2];
  ptrdiff_t true_lb, true_extent, lb, extent;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_rank(comm, &rank); // get rank
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  ret = MPI_Comm_size(comm, &size); // get size of comm
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  // if(rank == 0) {
  //   printf("4: RING\n");
  //   fflush(stdout);
  // }

  /* Special case for size == 1 */
  if (1 == size) { // if only one rank, no need to do anything
    if (MPI_IN_PLACE != d_sbuf) {
      cudaMemcpy((char *)d_sbuf, (char *)d_rbuf, count * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
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
  short early_late = 0;
  /* Computation loop */

  /*
  For each of the remote nodes:
  - post irecv for block (r-1)
  - send block (r)
  - in loop for every step k = 2 .. n
  - post irecv for block (r + n - k) % n
  - wait on block (r + n - k + 1) % n to arrive
  - compute on block (r + n - k + 1) % n
  - send block (r + n - k + 1) % n
  - wait on block (r + 1)
  - compute on block (r + 1)
  - send block (r + 1) to rank (r + 1)
  Note that we must be careful when computing the beginning of buffers and
  for send operations and computation we must compute the exact block size.
  */
  size_t cmpSize;
  inbi = 0;
  block_offset =
      ((rank < split_rank)
           ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
           : ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);

  /* Initialize first receive from the neighbor on the left */
  GSZ_compress_deviceptr_outlier((d_rbuf + block_offset * sizeof(float)),
                                 d_cmpSendBytes, block_count, &cmpSize, eb,
                                 streamComprs); // compress data
  MPI_Irecv(d_inbuf[inbi], max_segcount, MPI_BYTE, recv_from, 0, comm,
            &reqs[inbi]); // recv compressed data

  MPI_Send(d_cmpSendBytes, block_count, MPI_BYTE, send_to, 0, comm);

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
    kernel_quant_prediction<<<grid, block, 0, streamQuanPred>>>(
        d_rbuf + block_offset * sizeof(float), d_quant_predData, eb,
        block_count);
    /* Post irecv for the current block */

    MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from, 0, comm,
              &reqs[inbi]); // recv compressed data

    /* Wait on previous block to arrive */
    MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);

    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, 0, &cmpSize);
    /* Apply operation on previous block: result goes to rbuf
    rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */

    /* send previous block to send_to */
    MPI_Send(d_cmpReduceBytes, block_count * sizeof(float), MPI_BYTE, send_to,
             0, comm);
  }

  /* Wait on the last block to arrive */
  MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);

  /* Apply operation on the last block (from neighbor (rank + 1)
  rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)
                      ? ((ptrdiff_t)recv_from * early_segcount)
                      : ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  // d_rbuf + (ptrdiff_t)block_offset * extent;
  kernel_quant_prediction<<<grid, block, 0, streamQuanPred>>>(
      d_rbuf + block_offset * sizeof(float), d_quant_predData, eb, block_count);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_cmpReduceBytes,
                  block_count, eb, 0, &cmpSize);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  for (k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    const int send_data_from = (rank + 1 + size - k) % size;

    const ptrdiff_t recv_block_offset =
        ((recv_data_from < split_rank)
             ? ((ptrdiff_t)recv_data_from * early_segcount)
             : ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
    const ptrdiff_t send_block_offset =
        ((send_data_from < split_rank)
             ? ((ptrdiff_t)send_data_from * early_segcount)
             : ((ptrdiff_t)send_data_from * late_segcount + split_rank));

    block_count =
        ((recv_data_from < split_rank) ? early_segcount : late_segcount);

    MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from, 0, comm,
              &reqs[inbi]);

    MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm);

    MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);

    GSZ_decompress_deviceptr_outlier(d_rbuf + recv_block_offset * sizeof(float),
                                     d_inbuf[inbi], (size_t)block_count,
                                     cmpSize, eb, streamDecomprs);
    cudaMemcpy(d_cmpReduceBytes, d_inbuf[inbi], cmpSize,
               cudaMemcpyDeviceToDevice);

    inbi = inbi ^ 0x1;
  }

  return MPI_SUCCESS;

error_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line,
          rank, ret);
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  (void)line; // silence compiler warning
  if (NULL != d_cmpSendBytes) {
    cudaFree(d_cmpSendBytes);
  }
  if (NULL != d_cmpReduceBytes) {
    cudaFree(d_cmpReduceBytes);
  }
  if (NULL != d_quant_predData) {
    cudaFree(d_quant_predData);
  }
  if (NULL != d_inbuf[0]) {
    cudaFree(d_inbuf[0]);
  }
  if (NULL != d_inbuf[1]) {
    cudaFree(d_inbuf[1]);
  }
  cudaStreamDestroy(streamQuanPred);
  cudaStreamDestroy(streamComprs);
  return ret;
}