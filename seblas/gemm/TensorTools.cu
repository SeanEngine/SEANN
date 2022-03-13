//
// Created by DanielSun on 2/28/2022.
//

#include "TensorTools.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CUDA_BLOCK_SIZE_1D CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y
#define topOff(a,b) (a + b - 1)/(b)
#define RM 4
#define RN 4

namespace seblas{

    //for slice, the dimensions other than the chosen slice dimension should
    //be zero in the index4
    Tensor* slice(Tensor *in, Tensor *buffer, range4 range) {
        for(int i=0; i<4;i++){
            if(range.end[i] > 0){
                cudaMemcpy(buffer->elements, in->elements + in->dims[range.began],
                           sizeof(float) * in->dims[range.end - range.began],
                           cudaMemcpyDeviceToDevice);
            }
        }
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return buffer;
    }

    //------< TENSOR FUNCTIONALITIES >------

    __global__ void addD(Tensor* in, Tensor* other){
        unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] += other->elements[index];
        }
    }

    __global__ void add4D(Tensor* in, Tensor* other){
        unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] + regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void subtractD(Tensor* A, Tensor* B, Tensor* C){
        unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < A->dims.size){
            C->elements[index] = B->elements[index] - A->elements[index];
        }
    }

    __global__ void subtract4D(Tensor* A, Tensor* B, Tensor* C){
        unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < A->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(A->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(B->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] - regisB[i];
            }
            toFloat4R(C->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void hadamardProductD(Tensor* in, Tensor* other){
        unsigned int index = threadIdx.y + blockIdx.y * blockDim.y;
        if(index < in->dims.size){
            in->elements[index] *= other->elements[index];
        }
    }

    __global__ void hadamardProduct4D(Tensor* in, Tensor* other){
        unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] * regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void constProductD(Tensor* in, float val){
        unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] *= val;
        }
    }

    __global__ void constProduct4D(Tensor* in, float val){
        unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] * val;
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void transposeD(Tensor* in, Tensor* out){
        unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

        if(row < in->dims.rows && col < in->dims.cols){
            out->elements[col*in->dims.rows + row] = in->elements[row*in->dims.cols + col];
        }
    }

    Tensor* add(Tensor *in, Tensor *other){
        assert(in->dims.size == other->dims.size);
        unsigned int block = CUDA_BLOCK_SIZE_1D;
        unsigned int grid = topOff(in->dims.size, block);

        if(in->dims.size %4 == 0){
            grid = topOff(in->dims.size, block * 4);
            add4D<<<grid,block>>>(in,other);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return in;
        }

        addD<<<grid, block>>>(in,other);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return in;
    }

    Tensor* subtract(Tensor* A, Tensor* B, Tensor* C){
        assert(A->dims.size == B->dims.size == C->dims.size);
        unsigned int block = CUDA_BLOCK_SIZE_1D;
        unsigned int grid = topOff(A->dims.size, block);

        if(A->dims.size %4 == 0){
            grid = topOff(A->dims.size, block * 4);
            subtract4D<<<grid,block>>>(A,B,C);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return C;
        }

        subtractD<<<grid, block>>>(A,B,C);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return C;
    }

    Tensor* subtract(Tensor* in, Tensor* other){
        assert(in->dims.size == other->dims.size);
        return subtract(in, other, in);
    }

    Tensor* hadamardProduct(Tensor* in, Tensor* other){
        assert(in->dims.size == other->dims.size);
        unsigned int block = CUDA_BLOCK_SIZE_1D;
        unsigned int grid = topOff(in->dims.size, block);

        if(in->dims.size %4 == 0){
            grid = topOff(in->dims.size, block * 4);
            hadamardProduct4D<<<grid,block>>>(in,other);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return in;
        }

        hadamardProductD<<<grid, block>>>(in,other);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return in;
    }

    Tensor *constProduct(Tensor *in, float val){
        unsigned int block = CUDA_BLOCK_SIZE_1D;
        unsigned int grid = topOff(in->dims.size, block);

        if(in->dims.size %4 == 0){
            grid = topOff(in->dims.size, block * 4);
            constProduct4D<<<grid,block>>>(in,val);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return in;
        }

        constProductD<<<grid, block>>>(in,val);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return in;
    }

    //I'm tired of optimizing this
    //who fucking cares about transpose performance
    Tensor *transpose(Tensor* in, Tensor* out){
        assert(in->dims.cols == out->dims.rows);
        assert(in->dims.size == out->dims.size);
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = dim3(topOff(in->dims.cols,block.x),
                topOff(in->dims.rows,block.y));
        transposeD<<<grid,block>>>(in,out);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return out;
    }
}
