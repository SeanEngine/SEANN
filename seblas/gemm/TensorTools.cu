//
// Created by DanielSun on 2/28/2022.
//

#include "TensorTools.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CUDA_BLOCK_SIZE_1D CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y
#define topOff(a,b) (a + b - 1)/b
#define RM 8
#define RN 8

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

    __global__ void subtractD(Tensor* in, Tensor* other){
        unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] -= other->elements[index];
        }
    }

    __global__ void subtract4D(Tensor* in, Tensor* other){
        unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] - regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
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

    template<const int REGIS_M, const int REGIS_N>
    __global__ void transposeD(Tensor* in, Tensor* out){
        unsigned int M = in->dims.rows;
        unsigned int N = in->dims.cols;
        unsigned int procRow = (threadIdx.y + blockIdx.y * blockDim.y) * REGIS_M;
        unsigned int procCol = (threadIdx.x + blockIdx.x * blockDim.x) * REGIS_N;
        float regisB[REGIS_N][REGIS_M] = {0};
        if(procCol >= N || procRow >= M) return;
        int innerLoop = procCol + REGIS_N < N ? REGIS_N : N - procCol;

        #pragma unroll
        for(int i = 0; i<REGIS_M; i++){
            #pragma unroll
            for(int j = 0; j<innerLoop; j++){
                regisB[j][i] = in->elements[(procRow + i)*N + procCol+j];
            }
        }

        #pragma unroll
        for(int i = 0; i<innerLoop; i++){
            #pragma unroll
            for (int j = 0; j<REGIS_M; j++){
                out->elements[(procCol+i)*M + procRow + j] = regisB[i][j];
            }
        }
    }

    //fast transpose
    template<const int REGIS_M, const int REGIS_N>
    __global__ void transpose4D(Tensor* in, Tensor* out){
        unsigned int M = in->dims.rows;
        unsigned int N = in->dims.cols;
        unsigned int procRow = (threadIdx.y + blockIdx.y * blockDim.y) * REGIS_M;
        unsigned int procCol = (threadIdx.x + blockIdx.x * blockDim.x) * REGIS_N;
        float regisB[REGIS_N][REGIS_M] = {0};
        float row[4];
        if(procCol >= N || procRow >= M) return;
        int innerLoop = procCol + REGIS_N < N ? REGIS_N : N - procCol;

        #pragma unroll
        for(int i = 0; i<REGIS_M; i++){
            #pragma unroll
            for(int j = 0; j<innerLoop; j+=4){
                toFloat4R(row[0]) = toFloat4R(in->elements[(procRow + i)*N + procCol+j]);
                regisB[j][i] = row[0];
                regisB[j+1][i] = row[1];
                regisB[j+2][i] = row[2];
                regisB[j+3][i] = row[3];
            }
        }

        #pragma unroll
        for(int i = 0; i<innerLoop; i++){
            #pragma unroll
            for (int j = 0; j<REGIS_M; j+=4){
                toFloat4R(out->elements[(procCol+i)*M + procRow + j]) = toFloat4R(regisB[i][j]);
            }
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

    Tensor* subtract(Tensor *in, Tensor *other){
        assert(in->dims.size == other->dims.size);
        unsigned int block = CUDA_BLOCK_SIZE_1D;
        unsigned int grid = topOff(in->dims.size, block);

        if(in->dims.size %4 == 0){
            grid = topOff(in->dims.size, block * 4);
            subtract4D<<<grid,block>>>(in,other);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return in;
        }

        subtractD<<<grid, block>>>(in,other);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return in;
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

    Tensor *transpose(Tensor* in, Tensor* out){
        assert(in->dims.activeDims==2);
        assert(in->dims.cols == out->dims.rows);
        assert(in->dims.size == out->dims.size);
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = dim3(topOff(in->dims.cols,block.x * RN),
                topOff(in->dims.rows,block.y * RM));
        if(in->dims.rows % 4 == 0 && in->dims.cols % 4 == 0){
            transpose4D<RM,RN><<<grid, block>>>(in,out);
            cudaDeviceSynchronize();
            ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
            return out;
        }
        transposeD<RM,RN><<<grid, block>>>(in,out);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return out;
    }
}
