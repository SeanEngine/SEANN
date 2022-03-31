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
#include "seio/loader/DataLoader.cuh"
#include "seio/logging/LogUtils.cuh"
#include "sexec/threading/ThreadController.cuh"
#include "seann/models/testModel/ConvSTD.cuh"
#include "seann/models/MNISTNN/MNISTNN.cuh"
#include "seblas/gemm/Initializers.cuh"
#include "seann/models/Config.cuh"

#pragma comment(lib, "cublas.lib")

using namespace sexec;
using namespace seblas;
using namespace seann;
using namespace seio;
using namespace std;

int main(int argc, char **argv) {

    auto* A = Tensor::declare(50,1)->create()->constFill(2);
    auto* B = Tensor::declare(1,50)->create()->constFill(2);
    auto* C = Tensor::declare(50,50)->create()->zeroFill();

    inspect(sgemm(A,B,C));
    C->zeroFill();
    inspect(sgemmNaive(A,B,C));
/*
    Model *model = new MNISTNN();
    model->train(*new Config());
*/
}