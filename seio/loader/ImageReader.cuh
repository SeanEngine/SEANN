//
// Created by DanielSun on 2/24/2022.
//

#ifndef SEANN_IMAGEREADER_CUH
#define SEANN_IMAGEREADER_CUH
#include "../../seblas/gemm/Tensor.cuh"
#include "opencv2/opencv.hpp"
#include "opencv2/core/matx.hpp"

using namespace seblas;

namespace seio{
    Tensor* readRGB(const char* path, Tensor* reserved);

    void loadBinFile(const char* path, uchar* buffer, uint32 size);

    Tensor* readBinPixels(const uchar* target, Tensor* reserved);
}

#endif //SEANN_IMAGEREADER_CUH
