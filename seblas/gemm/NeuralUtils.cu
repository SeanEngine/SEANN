//
// Created by DanielSun on 3/5/2022.
//

#include "NeuralUtils.cuh"
#include "../assist/DBGTools.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define sigmoidCalc(x) (1.0f / (1.0f + expf(-x)))
#define tanhCalc(x) (expf(x) - expf(-x) / (expf(x) + expf(-x)))
#define topOff(a,b) (a + b - 1)/b

namespace seblas{

    __device__ __forceinline__ float warpReduce(float val){
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0x1, val, mask);
        }
        return val;
    }

    __global__ void reluD(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            input->elements[idx] = max(input->elements[idx], 0.0f);
        }
    }

    __global__ void reluDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            input->elements[idx] = (input->elements[idx] > 0.0f) ? 1.0f : 0.0f;
        }
    }

    __global__ void relu4D(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = max(regis[0], 0.0f);
            regis[1] = max(regis[1],0.0f);
            regis[2] = max(regis[2],0.0f);
            regis[3] = max(regis[3],0.0f);
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void relu4DGrad(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0.0f ? 1.0f : 0.0f;
            regis[1] = regis[1] > 0.0f ? 1.0f : 0.0f;
            regis[2] = regis[2] > 0.0f ? 1.0f : 0.0f;
            regis[3] = regis[3] > 0.0f ? 1.0f : 0.0f;
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void leakyReluD(Tensor* input, float alpha){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            input->elements[idx] = val > 0 ? val : alpha * val;
        }
    }

    __global__ void leakyReluDGrad(Tensor* input, float alpha){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            input->elements[idx] = val > 0 ? 1.0f : alpha;
        }
    }

    __global__ void leakyRelu4D(Tensor* input, float alpha){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0 ? regis[0] : alpha * regis[0];
            regis[1] = regis[1] > 0 ? regis[1] : alpha * regis[1];
            regis[2] = regis[2] > 0 ? regis[2] : alpha * regis[2];
            regis[3] = regis[3] > 0 ? regis[3] : alpha * regis[3];
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void leakyRelu4DGrad(Tensor* input, float alpha){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0 ? 1.0f : alpha;
            regis[1] = regis[1] > 0 ? 1.0f : alpha;
            regis[2] = regis[2] > 0 ? 1.0f : alpha;
            regis[3] = regis[3] > 0 ? 1.0f : alpha;
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void sigmoidD(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = sigmoidCalc(x);
        }
    }

    __global__ void sigmoidDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = sigmoidCalc(x) * (1.0f - sigmoidCalc(x));
        }
    }

    __global__ void sigmoid4D(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = sigmoidCalc(regis[0]);
            regis[1] = sigmoidCalc(regis[1]);
            regis[2] = sigmoidCalc(regis[2]);
            regis[3] = sigmoidCalc(regis[3]);
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void sigmoid4DGrad(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = sigmoidCalc(regis[0]) * (1.0f - sigmoidCalc(regis[0]));
            regis[1] = sigmoidCalc(regis[1]) * (1.0f - sigmoidCalc(regis[1]));
            regis[2] = sigmoidCalc(regis[2]) * (1.0f - sigmoidCalc(regis[2]));
            regis[3] = sigmoidCalc(regis[3]) * (1.0f - sigmoidCalc(regis[3]));
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void tanhD(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = tanhCalc(x);
        }
    }

    __global__ void tanhDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = 1.0f - tanhCalc(x) * tanhCalc(x);
        }
    }

    __global__ void tanh4D(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = tanhCalc(regis[0]);
            regis[1] = tanhCalc(regis[1]);
            regis[2] = tanhCalc(regis[2]);
            regis[3] = tanhCalc(regis[3]);
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }

    __global__ void tanh4DGrad(Tensor* input){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = 1.0f - tanhCalc(regis[0]) * tanhCalc(regis[0]);
            regis[1] = 1.0f - tanhCalc(regis[1]) * tanhCalc(regis[1]);
            regis[2] = 1.0f - tanhCalc(regis[2]) * tanhCalc(regis[2]);
            regis[3] = 1.0f - tanhCalc(regis[3]) * tanhCalc(regis[3]);
            toFloat4R(input->elements[idx]) = toFloat4R(regis[0]);
        }
    }
    template <const unsigned int BLOCK_WARPS>
    __global__ void reduceD(const float* input, float* buffer, unsigned int procSize){
        //loading data
        //each thread loads 2 elements
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;
        float sum = idx < procSize ? input[idx] : 0.0f;
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) buffer[blockIdx.x] = sum;
        }
    }

    Tensor* relu(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            relu4D<<<grid, block>>>(input);
        }else{
            reluD<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* reluDerive(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            relu4DGrad<<<grid, block>>>(input);
        }else{
            reluDGrad<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* leakyRelu(Tensor* input, float alpha){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            leakyRelu4D<<<grid, block>>>(input, alpha);
        }else{
            leakyReluD<<<grid, block>>>(input, alpha);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* leakyReluDerive(Tensor* input, float alpha){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            leakyRelu4DGrad<<<grid, block>>>(input, alpha);
        }else{
            leakyReluDGrad<<<grid, block>>>(input, alpha);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* sigmoid(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            sigmoid4D<<<grid, block>>>(input);
        }else{
            sigmoidD<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* sigmoidDerive(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            sigmoid4DGrad<<<grid, block>>>(input);
        }else{
            sigmoidDGrad<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* tanh(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            tanh4D<<<grid, block>>>(input);
        }else{
            tanhD<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    Tensor* tanhDerive(Tensor* input){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            tanh4DGrad<<<grid, block>>>(input);
        }else{
            tanhDGrad<<<grid, block>>>(input);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }

    float reduce(Tensor* input, float* buffer){
        float* source = input->elements;
        unsigned int procSize = input->dims.size;
        unsigned int block = BLOCK_WARP * WARP_SIZE;

        while(procSize > 1){
            unsigned int grid = topOff(procSize, block);
            reduceD<BLOCK_WARP><<<grid,block>>>(source, buffer, procSize);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
            procSize = grid;
            source = buffer;
        }

        float result = 0;
        cudaMemcpy(&result, buffer, sizeof(float), cudaMemcpyDeviceToHost);
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return result;
    }
}