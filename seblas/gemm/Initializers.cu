//
// Created by DanielSun on 3/22/2022.
//

#include <curand_kernel.h>
#include "Initializers.cuh"
namespace seblas{
    //this method will generate a pair of random floats subjecting to normal distribution
    __global__ void randNormalD(Tensor* A, float mean, float stddev, long seed){
        uint32 id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= A->dims.size) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_normal(&state);
        A->elements[id] = val * stddev + mean;
    }

    __global__ void randUniformD(Tensor* A, float min, float max, long seed){

        uint32 id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= A->dims.size) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_uniform(&state);
        A->elements[id] = val * (max - min) + min;
    }

    Tensor* randNormal(Tensor* A, float mean, float stddev){
        long seed = (long)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (A->dims.size + block - 1) / block;
        randNormalD<<<grid, block>>>(A, mean, stddev, seed);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return A;
    }

    Tensor* randUniform(Tensor* A, float min, float max){
        long seed = (long)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (A->dims.size + block - 1) / block;
        randUniformD<<<grid, block>>>(A, min, max, seed);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return A;
    }
}