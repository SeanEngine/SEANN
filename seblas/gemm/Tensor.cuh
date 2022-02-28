//
// Created by DylanWake on 2/12/2022.
//

#ifndef SEANN_TENSOR_CUH
#define SEANN_TENSOR_CUH

#include <cassert>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "../assist/ErrorHandler.cuh"

namespace seblas {

    const dim3 CUDA_BLOCK_SIZE = dim3(16,16,4);
    static const unsigned int WARP_SIZE = 32;

    struct shape4{
        unsigned int n{}, c{}, rows{}, cols{};
        unsigned int size;
        unsigned int activeDims;

        shape4(unsigned int w, unsigned int depth, unsigned int rows, unsigned int cols){
            this->n = w;
            this->c = depth;
            this->rows = rows;
            this->cols = cols;
            activeDims = 4;

            size = cols * rows * depth * w;
        }

        shape4(unsigned int depth, unsigned int rows, unsigned int cols){
            this->c = depth;
            this->rows = rows;
            this->cols = cols;
            activeDims = 3;

            size = cols * rows * depth;
        }

        shape4(unsigned int rows, unsigned int cols){
            this->rows = rows;
            this->cols = cols;
            activeDims = 2;

            size = cols * rows;
        }

        __device__ shape4(unsigned int w, unsigned int depth, unsigned int rows, unsigned int cols){
            this->n = w;
            this->c = depth;
            this->rows = rows;
            this->cols = cols;
            activeDims = 4;

            size = cols * rows * depth * w;
        }

        __device__ shape4(unsigned int depth, unsigned int rows, unsigned int cols){
            this->c = depth;
            this->rows = rows;
            this->cols = cols;
            activeDims = 3;

            size = cols * rows * depth;
        }

        __device__ shape4(unsigned int rows, unsigned int cols){
            this->rows = rows;
            this->cols = cols;
            activeDims = 2;

            size = cols * rows;
        }


        bool operator==(shape4 another) const;
        shape4 operator+(shape4 another) const;
        __device__ shape4 operator+(shape4 another) const;

        void copy(shape4 other);
    };

    struct Tensor{

        shape4 dims;
        float* elements{};

        ///accessing the elements using a getter
        [[nodiscard]] float get(unsigned int index) const;
        [[nodiscard]] float get(unsigned int row, unsigned int col) const;
        [[nodiscard]] float get(unsigned int depth, unsigned int row, unsigned int col) const;
        [[nodiscard]] float get(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) const;

        [[nodiscard]] __device__ float getD(unsigned int index) const;
        [[nodiscard]] __device__ float getD(unsigned int row, unsigned int col) const;
        [[nodiscard]] __device__ float getD(unsigned int depth, unsigned int row, unsigned int col) const;
        [[nodiscard]] __device__ float getD(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) const;

        void set(unsigned int index, float val) const;
        void set(unsigned int row, unsigned int col, float val) const;
        void set(unsigned int depth, unsigned int row, unsigned int col, float val) const;
        void set(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float val) const;

        __device__ void setD(unsigned int index, float val) const;
        __device__ void setD(unsigned int row, unsigned int col, float val) const;
        __device__ void setD(unsigned int depth, unsigned int row, unsigned int col, float val) const;
        __device__ void setD(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float val) const;


        ///declare a tensor without allocating elements
        static Tensor* declare(shape4 shape){
            Tensor* t;
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
