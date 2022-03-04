#include <iostream>
#include <windows.h>
#include "cuda.h"
#include "seblas/gemm/GEMM.cuh"
#include "cuda_runtime.h"
#include "seblas/gemm/Tensor.cuh"
#include "seblas/gemm/TensorTools.cuh"

#include "cublas_v2.h"
#include "seblas/assist/DBGTools.cuh"
#include "seio/loader/ImageReader.cuh"
#pragma comment(lib, "cublas.lib")


using namespace seblas;
using namespace seio;
using namespace std;

int main(int argc, char **argv) {
    auto* A = Tensor::declare(24,12)->create()->randomFill();
    auto* B = Tensor::declare(12,24)->create();

    transpose(A,B);
    inspect(B);
}