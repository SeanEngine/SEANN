//
// Created by DylanWake on 2/12/2022.
//

#include "Tensor.cuh"
#include "../assist/ErrorHandler.cuh"
#include <cstdarg>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>
#include "TensorTools.cuh"
#include <windows.h>

namespace seblas {

    uint32 MEMORY_OCCUPATION = 0;

    __global__ void randInit(Tensor *target, long seed) {
        uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
        curandStateXORWOW_t state;
        curand_init(index * seed, 0, 0, &state);
        target->setL(static_cast<float>((curand_uniform(&state)) * 2 - 1.0F),index);
    }

    __global__ void constInit(Tensor *target, float val) {
        uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
        target->setL(val, index);
    }

    Tensor *Tensor::reshape(shape4 newDims) {
        assert(newDims.size == dims.size);
        dims = newDims;
        return this;
    }

    Tensor *Tensor::create() {
        cudaMalloc(&elements, dims.size * sizeof(float));
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        MEMORY_OCCUPATION += dims.size * sizeof(float);
        return this;
    }

    Tensor *Tensor::createHost() {
        cudaMallocHost(&elements, dims.size * sizeof(float));
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::randomFill() {
        LARGE_INTEGER cpuFre;
        LARGE_INTEGER begin;

        QueryPerformanceFrequency(&cpuFre);
        QueryPerformanceCounter(&begin);

        dim3 block = dim3(CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y);
        uint32 totalProc = (this->dims.size + block.x - 1) / block.x;

        randInit<<<totalProc, block>>>(this, begin.HighPart);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::zeroFill() {
        cudaMemset(elements, 0, sizeof(float) * dims.size);
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::constFill(float val) {
        dim3 block = dim3(CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y);
        uint32 totalProc = (this->dims.size + block.x - 1) / block.x;

        constInit<<<totalProc, block>>>(this, val);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return this;
    }

    __device__ __host__ bool shape4::operator==(shape4 other) const {
        return rows == other.rows && cols == other.cols && c == other.c && n == other.n;
    }

    void shape4::copy(shape4 other) {
        rows = other.rows;
        cols = other.cols;
        c = other.c;
        n = other.n;
        size = other.size;
    }

    __device__ __host__ shape4 shape4::operator+(shape4 another) const {
        return {another.n + n, another.c + c, another.rows + rows, another.cols + cols};
    }

    __device__ __host__ uint32 index4::operator[](index4 indexes) const {
        return indexes.n * c * rows * cols + indexes.c * rows * cols + indexes.rows * cols + indexes.cols;
    }

    __device__ __host__ bool index4::operator<(const index4& other) const {
        return n < other.n && c < other.c && cols < other.cols && rows < other.rows;
    }

    __device__ __host__ index4 index4::operator-(const index4 &other) const {
        return {n-other.n, c-other.c, rows-other.rows, cols-other.cols};
    }

    __device__ __host__ uint32 index4::operator[](uint32 index) const {
        switch (index) {
            case 0: return cols;
            case 1: return rows;
            case 2: return c;
            case 3: return n;
            default: return 0;
        }
    }

    Tensor *Tensor::operator+(Tensor *other) {
        return add(this, other);
    }

    Tensor *Tensor::operator-(Tensor *other) {
        return subtract(this, other);
    }

    Tensor *Tensor::operator*(Tensor *other) {
        return hadamardProduct(this, other);
    }

    Tensor *Tensor::operator*(float val) {
        return constProduct(this, val);
    }

    Tensor *Tensor::toDevice() const {
        auto* output = Tensor::declare(this->dims)->create();
        cudaMemcpy(output->elements, this->elements, sizeof(float) * this->dims.size, cudaMemcpyHostToDevice);
        return output;
    }

    Tensor *Tensor::attachElements(float* target) {
        this->elements = target;
        return this;
    }
}
