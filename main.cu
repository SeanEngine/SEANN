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
    Tensor* input1 = readRGBSquare(R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist\train\0\0.bmp)"
            , shape4(1,3,28,28));
    Tensor* input2 = readRGBSquare(R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist\train\0\1.bmp)"
            , shape4(1,3,28,28));

    Tensor* input = Tensor::declare(2,3,28,28)->create();
    cudaMemcpy(input->elements, input1->elements, input1->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(input->elements + input2->dims.size, input2->elements, input2->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
    Tensor* output = Tensor::declare(2,3,28,28)->create();

    Tensor* mean = Tensor::declare(1,3,28,28)->create();
    Tensor* variance = Tensor::declare(1,3,28,28)->create();

    Tensor* gamma = Tensor::declare(3,1)->create()->constFill(1);
    Tensor* beta = Tensor::declare(3,1)->create()->constFill(0);

    batchNormConv(input, output, mean, variance, gamma, beta, 1e-3);

    inspect(output);
}