#pragma once

__global__ void kernel_quant_prediction(float *const __restrict__ localData,
                                        int *const __restrict__ quantPredData,
                                        const float eb, const size_t nbEle);

__global__ void
kernel_homomophic_sum(unsigned char *const __restrict__ CmpDataIn,
                      volatile unsigned int *const __restrict__ CmpOffsetIn,
                      unsigned char *const __restrict__ CmpDataOut,
                      volatile unsigned int *const __restrict__ locOffsetOut,
                      volatile unsigned int *const __restrict__ CmpOffsetOut,
                      volatile unsigned int *const __restrict__ locOffsetIn,
                      volatile int *const __restrict__ flag,
                      volatile int *const __restrict__ flag_cmp,
                      int *const __restrict__ predQuant, const float eb,
                      const size_t nbEle);