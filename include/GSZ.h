#ifndef GSZ_INCLUDE_GSZ_H
#define GSZ_INCLUDE_GSZ_H

static const int cmp_tblock_size =
    32; // 32 should be the best, not need to modify.
static const int dec_tblock_size =
    32; // 32 should be the best, not need to modify.
static const int cmp_chunk = 1024;
static const int dec_chunk = 1024;

__global__ void
GSZ_compress_kernel_outlier(const float *const __restrict__ oriData,
                            unsigned char *const __restrict__ cmpData,
                            volatile unsigned int *const __restrict__ cmpOffset,
                            volatile unsigned int *const __restrict__ locOffset,
                            volatile int *const __restrict__ flag,
                            const float eb, const size_t nbEle);
__global__ void GSZ_decompress_kernel_outlier(
    float *const __restrict__ decData,
    const unsigned char *const __restrict__ cmpData,
    volatile unsigned int *const __restrict__ cmpOffset,
    volatile unsigned int *const __restrict__ locOffset,
    volatile int *const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void
GSZ_compress_kernel_plain(const float *const __restrict__ oriData,
                          unsigned char *const __restrict__ cmpData,
                          volatile unsigned int *const __restrict__ cmpOffset,
                          volatile unsigned int *const __restrict__ locOffset,
                          volatile int *const __restrict__ flag, const float eb,
                          const size_t nbEle);
__global__ void
GSZ_decompress_kernel_plain(float *const __restrict__ decData,
                            const unsigned char *const __restrict__ cmpData,
                            volatile unsigned int *const __restrict__ cmpOffset,
                            volatile unsigned int *const __restrict__ locOffset,
                            volatile int *const __restrict__ flag,
                            const float eb, const size_t nbEle);

__global__ void GSZ_decompress_kernel_outlier_vec(
    float *const __restrict__ decData,
    const unsigned char *const __restrict__ cmpData,
    volatile unsigned int *const __restrict__ cmpOffset,
    volatile unsigned int *const __restrict__ locOffset,
    volatile int *const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void GSZ_compress_kernel_outlier_vec(
    const float *const __restrict__ oriData,
    unsigned char *const __restrict__ cmpData,
    volatile unsigned int *const __restrict__ cmpOffset,
    volatile unsigned int *const __restrict__ locOffset,
    volatile int *const __restrict__ flag, const float eb, const size_t nbEle);

#endif // GSZ_INCLUDE_GSZ_H