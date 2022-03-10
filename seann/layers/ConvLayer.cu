//
// Created by DanielSun on 3/9/2022.
//

#include "ConvLayer.cuh"

void ConvLayer::forwardCalc(Tensor *prevA) {
    relu(*conv(prevA, filters, z, strideH, strideW, padH, padW) + biases,a);
}

void ConvLayer::backwardCalc(Tensor *prevZ, Tensor *prevError) const {
    *convDerive(filters, errors, prevError, strideH, strideW, padH, padW) * reluDerive(prevZ);
}

void ConvLayer::recFilters(Tensor *prevA) const {
    convError(errors, prevA, filters, strideH, strideW, padH, padW);
}

void ConvLayer::recBiases() {

}

void ConvLayer::applyFilters(float LEARNING_RATE, int BATCH_SIZE) {

}

void ConvLayer::applyBiases(float LEARNING_RATE, int BATCH_SIZE) {

}
