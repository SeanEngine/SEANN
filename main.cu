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
     auto* A = Tensor::declare(3,1,3,3)->create()->constFill(1);
     auto* B = readRGBSquare(R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\validation\convTest2.png)",
                             shape4(3,32,32));
     auto* C = Tensor::declare(1,32,32)->create();

     convDeriv(A,B,C,1,1,1);
     inspect(C);
}