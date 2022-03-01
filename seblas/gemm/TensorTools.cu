//
// Created by DanielSun on 2/28/2022.
//

#include "TensorTools.cuh"
#include "GEMM.cuh"

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas{
    __global__ void extract4D(Tensor* in, Tensor* buffer, shape4 start, shape4 extractRange){
        unsigned int readRow = start.rows + threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int readCol = start.cols + 4*(threadIdx.x + blockIdx.x * blockDim.x);
        shape4 src = in->dims;
        shape4 out = start + extractRange;
        #pragma unroll
        for(unsigned int n = start.n; n < out.n; n++){
            #pragma unroll
            for(unsigned int c = start.c; c < out.c; c++){
                if(readRow < extractRange.rows && readCol < extractRange.cols){
                    toFloat4R(buffer->elements[extractRange[
                            shape4(n-start.n,c-start.c,readRow -start.rows, readCol - start.cols)]])
                    = toFloat4R(in->elements[src[shape4(n,c,readRow,readCol)]]);
                }
            }
        }
    }

    Tensor* slice(Tensor *in, Tensor *buffer, DimPrefix dim, range dimRange) {
        switch (dim) {
            case DEPTH:
                assert(dimRange.b < in->dims.n);
                cudaMemcpy(buffer->elements, in->elements + dimRange.a * in->dims.c * in->dims.rows * in->dims.cols
                        , sizeof(float) * (dimRange.b - dimRange.a) * in->dims.c * in->dims.rows * in->dims.cols,
                        cudaMemcpyDeviceToDevice);
                break;
            case ROW:
                assert(dimRange.b < in->dims.c);
                cudaMemcpy(buffer->elements, in->elements + dimRange.a * in->dims.rows * in->dims.cols
                        , sizeof(float) * (dimRange.b - dimRange.a) * in->dims.rows * in->dims.cols,
                           cudaMemcpyDeviceToDevice);
                break;
            case COL:
                assert(dimRange.b < in->dims.rows);
                cudaMemcpy(buffer->elements, in->elements + dimRange.a * in->dims.cols
                        , sizeof(float) * (dimRange.b - dimRange.a) * in->dims.cols,
                           cudaMemcpyDeviceToDevice);
                break;
        }
        return buffer;
    }

    Tensor* extract4(Tensor *in, Tensor *buffer, shape4 start, shape4 extractRange) {
        assert(extractRange.cols%4==0);
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = ((extractRange.cols + block.x - 1)/(block.x * 4),
                (extractRange.rows + block.y - 1)/block.y);
        extract4D<<<grid, block>>>(in, buffer, start, extractRange);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return buffer;
    }
}
