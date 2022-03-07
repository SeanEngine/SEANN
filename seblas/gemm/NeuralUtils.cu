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

    __device__ __forceinline__ float warpCompare(float val) {
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            float temp = __shfl_xor_sync(0x1, val, mask);
            val = temp > val ? temp : val;
        }
        return val;
    }

    __device__ __forceinline__ float regisMax(const float* regis, unsigned int count){
        float max = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < count; i++) {
            max = regis[i] > max ? regis[i] : max;
        }
        return max;
    }

    __global__ void reluD(Tensor* input, Tensor* output){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            output->elements[idx] = max(input->elements[idx], 0.0f);
        }
    }

    __global__ void reluDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            input->elements[idx] = (input->elements[idx] > 0.0f) ? 1.0f : 0.0f;
        }
    }

    __global__ void relu4D(Tensor* input, Tensor* output){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = max(regis[0], 0.0f);
            regis[1] = max(regis[1],0.0f);
            regis[2] = max(regis[2],0.0f);
            regis[3] = max(regis[3],0.0f);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
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

    __global__ void leakyReluD(Tensor* input, Tensor* output, float alpha){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            output->elements[idx] = val > 0 ? val : alpha * val;
        }
    }

    __global__ void leakyReluDGrad(Tensor* input, float alpha){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float val = input->elements[idx];
            input->elements[idx] = val > 0 ? 1.0f : alpha;
        }
    }

    __global__ void leakyRelu4D(Tensor* input,Tensor* output, float alpha){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = regis[0] > 0 ? regis[0] : alpha * regis[0];
            regis[1] = regis[1] > 0 ? regis[1] : alpha * regis[1];
            regis[2] = regis[2] > 0 ? regis[2] : alpha * regis[2];
            regis[3] = regis[3] > 0 ? regis[3] : alpha * regis[3];
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
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

    __global__ void sigmoidD(Tensor* input, Tensor* output){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            output->elements[idx] = sigmoidCalc(x);
        }
    }

    __global__ void sigmoidDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = sigmoidCalc(x) * (1.0f - sigmoidCalc(x));
        }
    }

    __global__ void sigmoid4D(Tensor* input, Tensor* output){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = sigmoidCalc(regis[0]);
            regis[1] = sigmoidCalc(regis[1]);
            regis[2] = sigmoidCalc(regis[2]);
            regis[3] = sigmoidCalc(regis[3]);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
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

    __global__ void tanhD(Tensor* input, Tensor* output){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            output->elements[idx] = tanhCalc(x);
        }
    }

    __global__ void tanhDGrad(Tensor* input){
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < input->dims.size) {
            float x = input->elements[idx];
            input->elements[idx] = 1.0f - tanhCalc(x) * tanhCalc(x);
        }
    }

    __global__ void tanh4D(Tensor* input, Tensor* output){
        unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        float regis[4];
        if (idx < input->dims.size) {
            toFloat4R(regis[0]) = toFloat4R(input->elements[idx]);
            regis[0] = tanhCalc(regis[0]);
            regis[1] = tanhCalc(regis[1]);
            regis[2] = tanhCalc(regis[2]);
            regis[3] = tanhCalc(regis[3]);
            toFloat4R(output->elements[idx]) = toFloat4R(regis[0]);
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

    template <const unsigned int BLOCK_WARPS>
    __global__ void reduce4D(float* input, float* buffer, unsigned int procSize){
        //loading data
        //each thread loads 2 elements
        unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        unsigned int tid = threadIdx.x;
        float regisP[4];

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;
        if(idx < procSize){
            toFloat4R(regisP[0]) = toFloat4R(input[idx]);
        }
        float sum = regisP[0] + regisP[1] + regisP[2] + regisP[3];
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

    template <const unsigned int BLOCK_WARPS>
    __global__ void softmaxD1024(const float* input, float* output, unsigned int procSize){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int tid = threadIdx.x;

        __shared__ float warpCache[BLOCK_WARPS];
        __shared__ float maxCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;

        float value = idx < procSize ? input[idx] : 0.0f;
        float max = value;
        __syncthreads();

        max = warpCompare(max);
        if(laneId==0) maxCache[warpId] = max;

        __syncthreads();

        //final compare
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? maxCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) maxCache[0] = max;
        }
        __syncthreads();

        //max calculation done, starting softmax reduction
        value = idx < procSize ? exp(value-max) : 0.0f;
        float sum = value;
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) warpCache[0] = sum;
        }
        __syncthreads();

        if(idx < procSize){
            output[idx] = value / sum;
        }
    }

    template <const unsigned int BLOCK_WARPS>
    __global__ void softmax4D4096(float* input, float* output, unsigned int procSize){
        unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        unsigned int tid = threadIdx.x;

        __shared__ float warpCache[BLOCK_WARPS];
        __shared__ float maxCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;

        float regisP[4];
        if(idx < procSize){
            toFloat4R(regisP[0]) = toFloat4R(input[idx]);
        }
        float max = regisMax(regisP,4);
        __syncthreads();
        max = warpCompare(max);
        if(laneId==0) maxCache[warpId] = max;

        __syncthreads();

        //final compare
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? maxCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) maxCache[0] = max;
        }
        __syncthreads();

        //max calculation done, starting softmax reduction
        if(idx < procSize){
            regisP[0] = exp(regisP[0]-max);
            regisP[1] = exp(regisP[1]-max);
            regisP[2] = exp(regisP[2]-max);
            regisP[3] = exp(regisP[3]-max);
        }
        float sum = regisP[0] + regisP[1] + regisP[2] + regisP[3];
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) warpCache[0] = sum;
        }
        __syncthreads();

        if(idx < procSize){
            regisP[0] /= sum;
            regisP[1] /= sum;
            regisP[2] /= sum;
            regisP[3] /= sum;
            toFloat4R(output[idx]) = toFloat4R(regisP[0]);
        }
    }

    template <const unsigned int BLOCK_WARPS>
    __global__ void findMaxD(const float* input, float* buffer, unsigned int procSize){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int tid = threadIdx.x;

        __shared__ float warpCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;

        float value = idx < procSize ? input[idx] : 0.0f;
        float max = value;
        __syncthreads();

        max = warpCompare(max);
        if(laneId==0) warpCache[warpId] = max;

        __syncthreads();

        //final compare
        if(warpId==0){
            max = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            max = warpCompare(max);
            if(laneId==0) warpCache[0] = max;
        }
        __syncthreads();

        if(idx < procSize) buffer[idx] = max;
    }

    template <const unsigned int BLOCK_WARPS>
    __global__ void softmaxReduceD(const float* output, float* buffer, const float* maxBuf, unsigned int procSize){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        unsigned const int warpId = tid / WARP_SIZE;
        unsigned const int laneId = tid % WARP_SIZE;
        float sum = idx < procSize ? output[idx] : 0.0f;
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0){
                buffer[blockIdx.x] = sum;
            }
        }
    }

    __global__ void softmaxAssignD(const float* input, float* output, const float* maxBuf, unsigned int procSize){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < procSize){
            output[idx] = exp(input[idx] - maxBuf[0]);
        }
    }

    __global__ void softmaxResultD(float* output, const float* sumBuf, unsigned int procSize){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < procSize){
            output[idx] = output[idx] / sumBuf[0];
        }
    }

    __global__ void softmaxDeriveD(Tensor* input, Tensor* correct){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < input->dims.size){
            input->elements[idx] = input->elements[idx] - correct->elements[idx];
        }
    }

    __global__ void softmaxDerive4D(Tensor* input, Tensor* correct){
        unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        float regisP[4];
        float regisC[4];
        if(idx < input->dims.size){
            toFloat4R(regisP[0]) = toFloat4R(input->elements[idx]);
            toFloat4R(regisC[0]) = toFloat4R(correct->elements[idx]);
            regisP[0] = regisP[0] - regisC[0];
            regisP[1] = regisP[1] - regisC[1];
            regisP[2] = regisP[2] - regisC[2];
            regisP[3] = regisP[3] - regisC[3];
            toFloat4R(input->elements[idx]) = toFloat4R(regisP[0]);
        }
    }

    Tensor* relu(Tensor* input, Tensor* output){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            relu4D<<<grid, block>>>(input, output);
        }else{
            reluD<<<grid, block>>>(input, output);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return output;
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

    Tensor* leakyRelu(Tensor* input, Tensor* output, float alpha){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            leakyRelu4D<<<grid, block>>>(input, output, alpha);
        }else{
            leakyReluD<<<grid, block>>>(input, output, alpha);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return output;
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

    Tensor* sigmoid(Tensor* input, Tensor* output){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            sigmoid4D<<<grid, block>>>(input, output);
        }else{
            sigmoidD<<<grid, block>>>(input, output);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return output;
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

    Tensor* tanh(Tensor* input, Tensor* output){
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);
        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block*4);
            tanh4D<<<grid, block>>>(input, output);
        }else{
            tanhD<<<grid, block>>>(input, output);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return output;
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
        unsigned int block = REDUCE_WARP * WARP_SIZE;

        while(procSize > 1){
            unsigned int grid = topOff(procSize, block);
            if (procSize % 4== 0){
                grid = topOff(procSize, block * 4);
                reduce4D<REDUCE_WARP><<<grid, block>>>(source, buffer, procSize);
                goto OUT;
            }
            reduceD<REDUCE_WARP><<<grid,block>>>(source, buffer, procSize);
            OUT:
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

    Tensor* softmax(Tensor* input, Tensor* out){
        float* source = input->elements;
        unsigned int procSize = input->dims.size;
        unsigned int block = SOFTMAX_WARP * WARP_SIZE;

        if(procSize % 4 == 0) {
            if (procSize <= 4096) {
                unsigned int grid = topOff(procSize, block * 4);
                softmax4D4096<SOFTMAX_WARP><<<grid, block>>>(source, out->elements, procSize);
                cudaDeviceSynchronize();
                ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
                return out;
            }
            //TODO: Add general softmax with float4 optimization
        }

        if(procSize <= 1024){
            unsigned int grid = topOff(procSize, block);
            softmaxD1024<SOFTMAX_WARP><<<grid, block>>>(source, out->elements, procSize);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
            return out;
        }

        unsigned int grid = topOff(procSize, block);
        float* maxBuf, *sumBuf;
        cudaMalloc(&maxBuf, sizeof(float) * grid);
        cudaMalloc(&sumBuf, sizeof(float) * grid);

        auto* maxBufT  = Tensor::declare(1,2);
        maxBufT->elements = sumBuf;

        unsigned int maxProc = procSize;
        float* op = source;
        while (maxProc > 1) {
            grid = topOff(maxProc, block);
            findMaxD<SOFTMAX_WARP><<<grid, block>>>(op, maxBuf, maxProc);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
            maxProc = grid;
            op = maxBuf;
        }

        grid = topOff(procSize, block);
        softmaxAssignD<<<grid,block>>>(input->elements, out->elements, maxBuf, procSize);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);

        unsigned int sumProc = procSize;
        op = out->elements;
        while (sumProc > 1) {
            grid = topOff(sumProc, block);
            softmaxReduceD<SOFTMAX_WARP><<<grid, block>>>(op, sumBuf, maxBuf, sumProc);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
            sumProc = grid;
            inspect(maxBufT);
            op = sumBuf;
        }

        grid = topOff(procSize, block);
        softmaxResultD<<<grid,block>>>(out->elements, sumBuf, procSize);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);

        cudaFree(maxBuf);
        cudaFree(sumBuf);

        return out;
    }

    Tensor* softmaxDerive(Tensor* input, Tensor* correct) {
        assert(input->dims.size == correct->dims.size);
        unsigned int block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        unsigned int grid = topOff(input->dims.size, block);

        if(input->dims.size % 4 == 0){
            grid = topOff(input->dims.size, block * 4);
            softmaxDerive4D<<<grid,block>>>(input, correct);
        }else{
            softmaxDeriveD<<<grid,block>>>(input, correct);
        }
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return input;
    }
}