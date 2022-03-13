//
// Created by DanielSun on 3/9/2022.
//

#include "ConvLayer.cuh"

namespace seann {
    void ConvLayer::forwardCalc(Tensor *prevA) {
        relu(conv(prevA, filters, z, strideH, strideW, padH, padW, biases), a);
    }

    void ConvLayer::backwardCalc(Tensor *prevZ, Tensor *prevError) const {
        *convDerive(filters, errors, prevError, strideH, strideW, padH, padW) * reluDerive(prevZ);
    }

    void ConvLayer::recFilters(Tensor *prevA) const {
        convError(errors, prevA, deltaFilters, strideH, strideW, padH, padW);
    }

    void ConvLayer::recBiases() const {
        if (enableBias)
            convBiasError(errors, deltaBiases, buffer);
    }

    void ConvLayer::applyFilters(float LEARNING_RATE, int BATCH_SIZE) const {
        *filters - *deltaFilters * (LEARNING_RATE / (float) BATCH_SIZE);
        deltaFilters->zeroFill();
    }

    void ConvLayer::applyBiases(float LEARNING_RATE, int BATCH_SIZE) const {
        if (enableBias) {
            *biases - *deltaBiases * (LEARNING_RATE / (float) BATCH_SIZE);
            deltaBiases->zeroFill();
        }
    }
}