#include <iostream>
#include <windows.h>
#include <cstdio>
#include "cuda.h"
#include "seblas/gemm/GEMM.cuh"
#include "cuda_runtime.h"
#include "seblas/gemm/Tensor.cuh"
#include "seblas/gemm/TensorTools.cuh"
#include "seblas/gemm/NeuralUtils.cuh"

#include "cublas_v2.h"
#include "seblas/assist/DBGTools.cuh"
#include "seio/loader/ImageReader.cuh"
#include "seio/logging/LogUtils.cuh"
#pragma comment(lib, "cublas.lib")


using namespace seblas;
using namespace seio;
using namespace std;

int main(int argc, char **argv) {
    logInfo(seio::LOG_SEG_SEANN, "Test");
    return 0;
}