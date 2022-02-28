//
// Created by DanielSun on 2/24/2022.
//

#include "ImageReader.cuh"
#define RGB_DECAY 1.0f/256.0f

namespace seio{

    /**
     * The method that would process opencv mats in host memory to data tensors in device memory
     * @param img the input image of cv::Mat
     * @param data the output data
     * @param decay a index that whether you choose to normalize the image
     */
    __global__ void mat2tensor(const uchar* elements, shape4 dims, Tensor* data, float decay){
        unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int depth = threadIdx.z + blockIdx.z * blockDim.z;
        uchar element = col < dims.cols && row < dims.rows && depth < dims.c
                        ? elements[(row * dims.cols + col)*dims.c + depth] : 0;
        data->setD(depth, row, col, (float)element * decay);
    }

    Tensor* seio::readRGBSquare(const char *path, shape4 dimensions) {
        auto* output = Tensor::declare(dimensions)->create();
        cv::Mat procImage = cv::imread(path, cv::IMREAD_COLOR);
        int h = procImage.size().height;
        int w = procImage.size().width;
        int smallDim = h < w ? h : w;
        procImage = procImage(cv::Range((h - smallDim) / 2, h - (h - smallDim) / 2),
                              cv::Range((w - smallDim) / 2, w - (w - smallDim) / 2));
        cv::Mat in;
        cv::resize(procImage, in, cv::Size((int)dimensions.rows, (int)dimensions.cols), cv::INTER_LINEAR);

        uchar* elements;
        cudaMalloc(&elements, dimensions.size * sizeof(uchar));
        cudaMemcpy(elements, in.data, dimensions.size * sizeof(uchar), cudaMemcpyHostToDevice);

        unsigned int blockDim = CUDA_BLOCK_SIZE.x;
        unsigned int blockDepth =  CUDA_BLOCK_SIZE.z;
        dim3 block = CUDA_BLOCK_SIZE;
        dim3 grid = dim3((dimensions.cols + blockDim - 1)/blockDim,
                          (dimensions.rows + blockDim - 1)/blockDim,
                         (dimensions.c + blockDepth - 1) / blockDepth);
        mat2tensor<<<grid,block>>>(elements,dimensions,output,RGB_DECAY);
        cudaDeviceSynchronize();
        cudaFree(elements);
        ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
        return output;
    }
}
