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
    auto *A = Tensor::declare(shape4(16384, 16384))->create()->randomFill();
    auto *B = Tensor::declare(shape4(16384, 16384))->create()->randomFill();
    auto *C = Tensor::declare(shape4(16384, 16384))->create();

    callGemmPrefetching(A,B,C);

    LARGE_INTEGER frq;
    LARGE_INTEGER beg;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&frq);
    QueryPerformanceCounter(&beg);

    callGemmPrefetching(A,B,C);

    QueryPerformanceCounter(&end);
    cout<<(double)(end.QuadPart - beg.QuadPart)/1e7<<endl;

    cout<<"-------------------------------------"<<endl;

    callGemmNaive(A,B,C);
}