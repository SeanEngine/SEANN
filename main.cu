#include <iostream>
#include <windows.h>
#include "cuda.h"
#include "seblas/gemm/GEMM.cuh"
#include "cuda_runtime.h"
#include "seblas/gemm/Tensor.cuh"
#include "seblas/gemm/TensorTools.cuh"
#include "seblas/gemm/NeuralUtils.cuh"

#include "cublas_v2.h"
#include "seblas/assist/DBGTools.cuh"
#include "seio/loader/ImageReader.cuh"
#pragma comment(lib, "cublas.lib")


using namespace seblas;
using namespace seio;
using namespace std;

int main(int argc, char **argv) {
    auto* A = Tensor::declare(32,12)->create()->randomFill();
    auto* B = Tensor::declare(12,32)->create()->randomFill();
    auto* C = Tensor::declare(32,32)->create();
    auto* BT = transpose(B,Tensor::declare(32,12)->create());

    sgemmNT(A,BT,C);
    inspect(C);

    cout<<"-----------"<<endl;
    sgemm(A,B,C);
    inspect(C);
}