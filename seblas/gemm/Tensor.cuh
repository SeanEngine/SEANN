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
#include "string"

using namespace std;
namespace seblas {

    typedef unsigned int uint32;

    using namespace std;

    const dim3 CUDA_BLOCK_SIZE = dim3(16,16,4);
    static const uint32 WARP_SIZE = 32;

    extern uint32 MEMORY_OCCUPATION;

    struct index4{
        uint32 n=0, c=0, rows=0, cols=0;
        __device__ __host__ index4(uint32 n, uint32 c, uint32 rows, uint32 cols){
            this->n = n;
            this->c = c;
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ index4(uint32 c, uint32 rows, uint32 cols){
            this->c = c;
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ index4(uint32 rows, uint32 cols){
            this->rows = rows;
            this->cols = cols;
        }

        __device__ __host__ uint32 operator[](uint32 in) const;
        __device__ __host__ uint32 operator[](index4 indexes) const;
        __device__ __host__ bool operator<(const index4& other) const;
        __device__ __host__ index4 operator-(const index4& other) const;

        [[nodiscard]] string toString() const{
            return "(" + to_string(n) + "," + to_string(c) + "," + to_string(rows) + "," + to_string(cols) + ")";
        }
    };

    struct shape4 : public index4{
        uint32 size;
        __device__ __host__ shape4(uint32 n, uint32 c, uint32 rows, uint32 cols)
        : index4(n,c,rows,cols){
            size = n * c * rows * cols;
        }

        __device__ __host__ shape4(uint32 c, uint32 rows, uint32 cols) :
                index4(1,c, rows, cols){
            size = cols * rows * c;
        }

        __device__ __host__ shape4(uint32 rows, uint32 cols) :
                index4(1,1,rows,cols){
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

        __host__ __device__ void setL(float value, uint32 index) const {
            if(index < dims.size) elements[index] = value;
        }

        [[nodiscard]] __device__ __host__ float getL(float value, uint32 index) const {
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
            MEMORY_OCCUPATION -= tensor->dims.size * sizeof(float);
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

        Tensor* toDevice() const;

        Tensor* attachElements(float* target);

        //simple randomizer to init numbers between 1 and -1
        Tensor* randomFill();
        Tensor* zeroFill();
        Tensor* constFill(float val);

        ///automatic reshape the tensor (as long as the element size remains the same)
        //this will only change the way our program understands the object
        Tensor *reshape(shape4 newDims);

        template<typename... Args>
        Tensor* reshape(Args &&... args){
            auto shape = shape4(std::forward<Args>(args)...);
            return reshape(shape);
        };

        Tensor* operator+(Tensor* other);
        Tensor* operator-(Tensor* other);
        Tensor* operator*(Tensor* other);
        Tensor* operator*(float val);
    };

    static void copyD2H(Tensor* onDevice, Tensor* onHost){
        assert(onDevice->dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, onDevice->elements, sizeof(float) * onDevice->dims.size, cudaMemcpyDeviceToHost);
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    }
}


#endif //SEANN_TENSOR_CUH
