#include <iostream>
#include <windows.h>
#include "cuda.h"
#include "seblas/gemm/GEMM.cuh"
#include "cuda_runtime.h"
#include "seblas/gemm/Tensor.cuh"
#include "cublas_v2.h"
#include "seblas/assist/DBGTools.cuh"
#include "seio/loader/ImageReader.cuh"
#pragma comment(lib, "cublas.lib")


using namespace seblas;
using namespace seio;
using namespace std;

int main(int argc, char **argv) {
    auto* A = Tensor::declare(shape4(64,3,3,3))->create();
    auto* B = Tensor::declare(shape4(64,224,224))->create()->randomFill();
    auto* C = Tensor::declare(shape4(3,224,224))->create()->randomFill();

    convD(A,B,C,1,1,1);
    inspect(C);
}