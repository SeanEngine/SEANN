//
// Created by DanielSun on 2/12/2022.
//

#ifndef SEANN_GEMM_CUH
#define SEANN_GEMM_CUH

#include "Tensor.cuh"

namespace seblas {
    Tensor* sgemmNaive(Tensor* A, Tensor* B, Tensor* C);
    Tensor* sgemm(Tensor* A, Tensor* B, Tensor* C);
    Tensor* conv(Tensor* A, Tensor* B, Tensor* C, int stride, int padH, int padW);
}

#endif //SEANN_GEMM_CUH
