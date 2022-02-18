//
// Created by DylanWake on 2/12/2022.
//

#include "Tensor.cuh"
#include "../assist/ErrorHandler.cuh"
#include <cstdarg>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>
#include <windows.h>

using namespace seblas;

__global__ void randInit(Tensor* target, long seed){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(index * seed, 0, 0, &state);
    target->setD(index, static_cast<float>((curand_uniform(&state))*2 - 1.0F));
}

__global__ void constInit(Tensor* target, float val){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    target->setD(index, val);
}

float Tensor::get(unsigned int index) const {
    return index < dims.size ? elements[index] : 0.0f;
}


float Tensor::get(unsigned int row, unsigned int col) const {
    return row < dims.rows
           && col < dims.cols
           ? elements[row * dims.cols + col] : 0.0f;
}

float Tensor::get(unsigned int depth, unsigned int row, unsigned int col) const {
    return row < dims.rows
           && col < dims.cols
           && depth < dims.depth
           ? elements[depth * dims.rows * dims.cols + row * dims.cols + col] : 0.0f;
}

float Tensor::get(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) const {
    return row < dims.rows
           && col < dims.cols
           && depth < dims.depth
           && w < dims.w
           ? elements[w * dims.depth * dims.rows * dims.cols + depth * dims.rows * dims.cols + row * dims.cols + col] : 0.0f;
}

__device__ float Tensor::getD(unsigned int index) const {
    return index < dims.size ? elements[index] : 0;
}


__device__ float Tensor::getD(unsigned int row, unsigned int col) const {
    return row < dims.rows
           && col < dims.cols
           ? elements[row * dims.cols + col] : 0.0f;
}

__device__ float Tensor::getD(unsigned int depth, unsigned int row, unsigned int col) const {
    return row < dims.rows
           && col < dims.cols
           && depth < dims.depth
           ? elements[depth * dims.rows * dims.cols + row * dims.cols + col] : 0.0f;
}

__device__ float Tensor::getD(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) {
    return row < dims.rows
           && col < dims.cols
           && depth < dims.depth
           && w < dims.w
           ? elements[w * dims.depth * dims.rows * dims.cols + depth * dims.rows * dims.cols + row * dims.cols + col] : 0.0f;
}

void Tensor::set(unsigned int index, float val) const {
    if(index < dims.size) elements[index] = val;
}

void Tensor::set(unsigned int row, unsigned int col, float val) const {
    if(row < dims.rows && col < dims.cols) elements[row * dims.cols + col] = val;
}

void Tensor::set(unsigned int depth, unsigned int row, unsigned int col, float val) const {
    if(depth < dims.depth && row < dims.rows && col < dims.cols)
        elements[depth * dims.rows * dims.cols + row * dims.cols + col] = val;
}

void Tensor::set(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float val) const {
    if(w < dims.w && depth < dims.depth && row < dims.rows && col < dims.cols)
        elements[w * dims.depth * dims.rows * dims.cols + depth * dims.rows * dims.cols + row * dims.cols + col] = val;
}

__device__ void Tensor::setD(unsigned int index, float val) const{
    if(index < dims.size) elements[index] = val;
}

__device__ void Tensor::setD(unsigned int row, unsigned int col, float val) const {
    if(row < dims.rows && col < dims.cols) elements[row * dims.cols + col] = val;
}

__device__ void Tensor::setD(unsigned int depth, unsigned int row, unsigned int col, float val) const {
    if(depth < dims.depth && row < dims.rows && col < dims.cols)
        elements[depth * dims.rows * dims.cols + row * dims.cols + col] = val;
}

__device__ void Tensor::setD(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float val) const {
    if(w < dims.w && depth < dims.depth && row < dims.rows && col < dims.cols)
        elements[w * dims.depth * dims.rows * dims.cols + depth * dims.rows * dims.cols + row * dims.cols + col] = val;
}

Tensor *Tensor::reshape(shape4 newDims) {
    assert(newDims.size == dims.size);
    dims = newDims;
    return this;
}

Tensor *Tensor::create() {
    cudaMalloc(&elements, dims.size * sizeof(float));
    ErrorHandler::checkDeviceStatus();
    return this;
}

Tensor *Tensor::createHost() {
    cudaMallocHost(&elements, dims.size * sizeof(float));
    ErrorHandler::checkDeviceStatus();
    return this;
}

Tensor *Tensor::randomFill() {
    LARGE_INTEGER cpuFre;
    LARGE_INTEGER begin;

    QueryPerformanceFrequency(&cpuFre);
    QueryPerformanceCounter(&begin);

    dim3 block = dim3(CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y);
    unsigned int totalProc = (this->dims.size + block.x-1)/block.x;

    randInit<<<totalProc, block>>>(this, begin.HighPart);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus();
    return this;
}

Tensor *Tensor::zeroFill() {
    cudaMemset(elements, 0, sizeof(float) * dims.size);
    ErrorHandler::checkDeviceStatus();
    return this;
}

Tensor *Tensor::constFill(float val) {
    dim3 block = dim3(CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y);
    unsigned int totalProc = (this->dims.size + block.x-1)/block.x;

    constInit<<<totalProc, block>>>(this, val);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus();
    return this;
}

bool shape4::operator==(shape4 other) const {
    return this->rows == other.rows
    && this->cols == other.cols
    && this->depth == other.depth
    && this->w == other.w
    && this->activeDims == other.activeDims;
}

void shape4::copy(shape4 other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->depth = other.depth;
    this->w = other.w;
    this->activeDims = other.activeDims;
    this->size = other.size;
}
