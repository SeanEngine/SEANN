//
// Created by DanielSun on 2/28/2022.
//

#include "TensorTools.cuh"
#include "GEMM.cuh"
#include "Tensor.cuh"


#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas{
    __global__ void extract4D(Tensor* in, Tensor* buffer, range4 range){
        unsigned int readRow = range.began.rows + threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int readCol = range.began.cols + 4*(threadIdx.x + blockIdx.x * blockDim.x);
        shape4 src = in->dims;
        index4 out = range.end;
        #pragma unroll
        for(unsigned int n = range.began.n; n < out.n; n++){
            #pragma unroll
            for(unsigned int c = range.began.c; c < out.c; c++){
                if(readRow < range.diff.rows && readCol < range.diff.cols){
                    toFloat4R(buffer->elements[range.diff[
                            index4(n-range.began.n,c-range.began.c,
                                   readRow -range.began.rows, readCol - range.began.cols)]])
                    = toFloat4R(in->elements[src[index4(n,c,readRow,readCol)]]);
                }
            }
        }
    }

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
        return buffer;
    }

    Tensor* extract4(Tensor *in, Tensor *buffer, range4 copyOff) {
        assert(copyOff.diff.cols%4==0);
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = dim3((copyOff.diff.cols + block.x - 1)/(block.x * 4),
                (copyOff.diff.rows + block.y - 1)/block.y);
        extract4D<<<grid, block>>>(in, buffer, copyOff);
        cudaDeviceSynchronize();
        ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
        return buffer;
    }

}
