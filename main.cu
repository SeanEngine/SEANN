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
    auto* B = readRGBSquare(R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\validation\convTest2.png)",shape4(3,32,32));
    auto* A = Tensor::declare(shape4(6,3,3,3))->create();
    inspect(B);
    float filters[] = {-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,
                       1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1
                       -1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,
                       1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1
                       -1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,
                       1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1};
    cudaMemcpy(A->elements,filters,sizeof(float) * 162, cudaMemcpyHostToDevice);
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    auto* C = Tensor::declare(shape4(6,32,32))->create();
    conv(A,B,C,1,1,1);
    inspect(C);
}