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

    LARGE_INTEGER beg;
    LARGE_INTEGER end;
    LARGE_INTEGER frq;

    float a = 1, b=0;

    QueryPerformanceFrequency(&frq);
    QueryPerformanceCounter(&beg);
    callGemmPrefetching(A,B,C);
    QueryPerformanceCounter(&end);
    cout << "fastGemm:   " << (double) (end.QuadPart - beg.QuadPart) / 1e7 << endl;

    C->zeroFill();

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)A->dims.rows, (int)B->dims.cols, (int)A->dims.cols, &a, A->elements,
                (int)A->dims.cols, B->elements, (int)B->dims.cols, &b, C->elements, (int)C->dims.cols);
    cublasDestroy(handle);

    cout<<"-----------------"<<endl;

    QueryPerformanceCounter(&beg);
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, (int)A->dims.rows, (int)B->dims.cols, (int)A->dims.cols, &a, A->elements,
                (int)A->dims.cols, B->elements, (int)B->dims.cols, &b, C->elements, (int)C->dims.cols);
    cublasDestroy(handle);
    QueryPerformanceCounter(&end);
    cout<<"Cublas:      " <<(double)(end.QuadPart - beg.QuadPart)/1e7<<endl;
}

