//
// Created by DanielSun on 2/12/2022.
//

#ifndef SEANN_GEMM_CUH
#define SEANN_GEMM_CUH

#include "Tensor.cuh"

namespace seblas {
    Tensor* callGemmPrefetching(Tensor* A, Tensor* B, Tensor* C);
}


#endif //SEANN_GEMM_CUH
