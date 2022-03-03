//
// Created by DylanWake on 2/12/2022.
//

#ifndef SEANN_TENSOR_CUH
#define SEANN_TENSOR_CUH

#include <cassert>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "../assist/ErrorHandler.cuh"
#include "vector"

namespace seblas {

    using namespace std;

    const dim3 CUDA_BLOCK_SIZE = dim3(16,16,4);
    static const unsigned int WARP_SIZE = 32;

    struct index4{
        unsigned int n=0, c=0, rows=0, cols=0;
        __device__ __host__ index4(unsigned int n, unsigned int c, unsigned int rows, unsigned int cols){
            this->n = n;
            this->c = c;
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ index4(unsigned int c, unsigned int rows, unsigned int cols){
            this->c = c;
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ index4(unsigned int rows, unsigned int cols){
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ unsigned int operator[](unsigned int in) const;
        __device__ __host__ unsigned int operator[](index4 indexes) const;
        __device__ __host__ bool operator<(const index4& other) const;
        __device__ __host__ index4 operator-(const index4& other) const;
    };

    struct shape4 : public index4{
        unsigned int size;
        unsigned int activeDims;

        __device__ __host__ shape4(unsigned int n, unsigned int c, unsigned int rows, unsigned int cols)
        : index4(n,c,rows,cols){
            activeDims = 4;
            size = cols * rows * rows * cols;
        }

        __device__ __host__ shape4(unsigned int c, unsigned int rows, unsigned int cols) :
                index4(1,c, rows, cols){
            activeDims = 3;
            size = cols * rows * c;
        }

        __device__ __host__ shape4(unsigned int rows, unsigned int cols) :
                index4(1,1,rows,cols){
            activeDims = 2;
            size = cols * rows;
        }

        __device__ __host__ bool operator==(shape4 another) const;
        __device__ __host__ shape4 operator+(shape4 another) const;

        void copy(shape4 other);
    };

    struct Tensor{

        shape4 dims;
        float* elements{};

        ///accessing the elements using a getter
        template<typename... Args>
        __device__ __host__ float get(Args&&... args) {
            auto location = index4(std::forward<Args>(args)...);
            if(!(location < dims)) return 0;
            return elements[dims[location]];
        }

        template<typename... Args>
        __host__ __device__ void set(float value, Args &&... args) {
            auto location = index4(std::forward<Args>(args)...);
            if(!(location < dims)) return;
            elements[dims[location]] = value;
        }

        __host__ __device__ void setL(float value, unsigned int index) const {
            if(index < dims.size) elements[index] = value;
        }

        [[nodiscard]] __device__ __host__ float getL(float value, unsigned int index) const {
            if(index < dims.size) return elements[index];
            return 0;
        }

        ///declare a tensor without allocating elements
        static Tensor* declare(shape4 shape){
            Tensor* t;
            cudaMallocHost(&t, sizeof(Tensor));
            cudaMemcpy(&t->dims, &shape, sizeof(shape4), cudaMemcpyHostToHost);
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return t;
        }
        template<typename... Args>
        static Tensor* declare(Args &&... args){
            Tensor* t;
            auto shape = shape4(std::forward<Args>(args)...);
            cudaMallocHost(&t, sizeof(Tensor));
            cudaMemcpy(&t->dims, &shape, sizeof(shape4), cudaMemcpyHostToHost);
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return t;
        }

        ///destroy a tensor object
        static void destroy(Tensor* tensor){
            cudaFree(tensor->elements);
            cudaFreeHost(tensor);
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        }

        ///destroy a tensor object on Host
        static void destroyHost(Tensor* tensor){
            cudaFreeHost(tensor->elements);
            cudaFreeHost(tensor);
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        }

        ///alloc the tensor on device memory
        Tensor* create();

        //alloc the tensor on host memory
        Tensor* createHost();

        //simple randomizer to init numbers between 1 and -1
        Tensor* randomFill();
        Tensor* zeroFill();
        Tensor* constFill(float val);

        ///automatic reshape the tensor (as long as the element size remains the same)
        //this will only change the way our program understands the object
        Tensor* reshape(shape4 newDims);
    };

    static void copyD2H(Tensor* onDevice, Tensor* onHost){
        assert(onDevice->dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, onDevice->elements, sizeof(float) * onDevice->dims.size, cudaMemcpyDeviceToHost);
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    }
}


#endif //SEANN_TENSOR_CUH
