//
// Created by DanielSun on 3/9/2022.
//

#include "ConvLayer.cuh"

namespace seann {
    void ConvLayer::forwardCalc(Tensor *prevA) {
        relu(conv(filters, prevA, z, strideH, strideW, padH, padW, biases), a);
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

    void ConvLayer::applyFilters(float LEARNING_RATE, uint32 BATCH_SIZE) const {
        *filters - *(deltaFilters) * (LEARNING_RATE / (float) BATCH_SIZE);
        deltaFilters->zeroFill();
    }

    void ConvLayer::applyBiases(float LEARNING_RATE, uint32 BATCH_SIZE) const {
        if (enableBias) {
            *biases - *deltaBiases * (LEARNING_RATE / (float) BATCH_SIZE);
            deltaBiases->zeroFill();
        }
    }

    void ConvLayer::forward(Layer *prev) {
        forwardCalc(prev->a);
    }

    void ConvLayer::backward(Layer *prev) {
        if(strcmp(prev->TYPE,"INPUT")==0){
            recFilters(prev->a);
            recBiases();
            return;
        }
        backwardCalc(prev->z, prev->errors);
        recFilters(prev->a);
        recBiases();
    }

    void ConvLayer::learn(float LEARNING_RATE, uint32 BATCH_SIZE) {
        applyFilters(LEARNING_RATE, BATCH_SIZE);
        applyBiases(LEARNING_RATE, BATCH_SIZE);
    }

    void ConvLayer::backwardOut(Tensor *correct) {

    }

    void ConvLayer::initialize() {
        logDebug(LOG_SEG_SEANN, "CONV initialized with rand normal");
        uint32 K = filters->dims.size / filters->dims.n;
        randNormal(filters, 0, (float)sqrt(2.0 / (float) K));
        if (enableBias)
            randNormal(biases, 0, (float)sqrt(2.0 / (float) filters->dims.n));
    }
}