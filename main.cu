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
    auto* A = Tensor::declare(30,11)->create()->constFill(1);
    auto* B = Tensor::declare(11,30)->create()->constFill(1);
    auto* C = Tensor::declare(30,30)->create();
    auto* BT = transpose(B,Tensor::declare(30,11)->create());
    inspect(BT);

    sgemmNT(A,BT,C);
    inspect(C);

    cout<<"-----------"<<endl;
    sgemm(A,B,C);
    inspect(C);
}