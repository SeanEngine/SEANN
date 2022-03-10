//
// Created by DanielSun on 3/9/2022.
//

#ifndef SEANN_CONVLAYER_CUH
#define SEANN_CONVLAYER_CUH


#include "Layer.cuh"

class ConvLayer : public Layer{
public:
    Tensor* filters, *biases, *deltaFilters, *deltaBiases;
    Tensor* z, *errors;

    int strideH, strideW, padH, padW;

    ConvLayer(shape4 filterSize, uint32 ih, uint32 iw, int strideH, int strideW, int padH, int padW){
        uint32 oh = (ih - filterSize.rows + 2 * padH) / strideH + 1;
        uint32 ow = (iw - filterSize.cols + 2 * padW) / strideW + 1;

        filters = Tensor::declare(filterSize)->create();
        biases = Tensor::declare(filterSize.n,1)->create();

        deltaFilters = Tensor::declare(filterSize)->create();
        deltaBiases = Tensor::declare(filterSize.n,1)->create();

        a = Tensor::declare(filterSize.n, oh, ow)->create();
        z = Tensor::declare(filterSize.n, oh, ow)->create();
        errors = Tensor::declare(filterSize.n, oh, ow)->create();

        this->strideH = strideH;
        this->strideW = strideW;
        this->padH = padH;
        this->padW = padW;
    }

    void forwardCalc(Tensor* prevA);

    void backwardCalc(Tensor* prevZ, Tensor* prevError) const;

    void recFilters(Tensor* prevA) const;

    void recBiases();

    void applyFilters(float LEARNING_RATE, int BATCH_SIZE);

    void applyBiases(float LEARNING_RATE, int BATCH_SIZE);
};


#endif //SEANN_CONVLAYER_CUH
