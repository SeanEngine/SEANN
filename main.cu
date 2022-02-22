#include <iostream>
#include <windows.h>
#include "cuda.h"
#include "seblas/gemm/GEMM.cuh"
#include "cuda_runtime.h"
#include "seblas/gemm/Tensor.cuh"
#include "cublas_v2.h"
#include "seblas/assist/DBGTools.cuh"
#pragma comment(lib, "cublas.lib")


using namespace seblas;
using namespace std;

int main(int argc, char **argv) {
    auto *A = Tensor::declare(shape4(12, 8))->create()->randomFill();
    auto *B = Tensor::declare(shape4(8, 16))->create()->randomFill();
    auto *C = Tensor::declare(shape4(12, 16))->create();

    callGemmPrefetching(A,B,C);
    inspect(C);

    cout<<"-------------------------------------"<<endl;

    callGemmNaive(A,B,C);
    inspect(C);
}