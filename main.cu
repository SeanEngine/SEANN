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

    Model *model = new MNISTNN();
    model->train(*new Config());

/*
    auto* F2 = readRGB(R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist\test\0\0.bmp)",
                       Tensor::declare(1, 3, 28, 28)->createHost())->toDevice();

    auto* F1 = readRGB(R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist\test\0\1.bmp)",
                      Tensor::declare(1, 3, 28, 28)->createHost())->toDevice();

    auto* A = Tensor::declare(1, 3, 28, 28)->attachElements(F1->elements);
    auto* B = Tensor::declare(1, 3, 28, 28)->attachElements(F2->elements);

    auto* C = Tensor::declare(3, 3, 3, 3)->create();

    C = convError(A, B, C, 2,2,1,1);
    inspect(C);
*/
 }