//
// Created by DanielSun on 2/12/2022.
//

#ifndef SEANN_GEMM_CUH
#define SEANN_GEMM_CUH

#include "Tensor.cuh"

namespace seblas {
    Tensor* sgemmNaive(Tensor* A, Tensor* B, Tensor* C);
    Tensor* sgemmNaiveTN(Tensor* A, Tensor* B, Tensor* C);
    Tensor* sgemmNaiveNTA(Tensor* A, Tensor* B, Tensor* C);

    Tensor* sgemm(Tensor* A, Tensor* B, Tensor* C);
    Tensor* sgemmTN(Tensor* A, Tensor* B, Tensor* C);
    Tensor* sgemmNT(Tensor* A, Tensor* B, Tensor* C);

    // A : adding the element instead of directly replacing
    Tensor* sgemmNTA(Tensor* A, Tensor* B, Tensor* C);

    Tensor* conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor* biases);
    Tensor* convDerive(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);
    Tensor* convError(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);
}

#endif //SEANN_GEMM_CUH
