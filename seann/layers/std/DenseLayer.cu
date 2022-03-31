//
// Created by DanielSun on 3/8/2022.
//

#include "DenseLayer.cuh"
#include "ConvLayer.cuh"

namespace seann {
    void DenseLayer::forwardCalc(Tensor *prevA) const {
        shape4 shape = prevA->dims;
        prevA->reshape(shape.size, 1);
        relu((*sgemm(weights, prevA, z) + biases), a);
        prevA->reshape(shape);
    }

    void DenseLayer::backwardCalc(Tensor *prevError, Tensor *prevZ) const {
        shape4 shape = prevError->dims;
        prevError->reshape(shape.size, 1);
        prevZ->reshape(shape.size, 1);
        *sgemmTN(weights, errors, prevError) * reluDerive(prevZ);
        prevError->reshape(shape);
        prevZ->reshape(shape);
    }

    void DenseLayer::backwardCalcOut(Tensor *correct) const {
        *subtract(a, correct, errors) * reluDerive(z);
    }

    void DenseLayer::recWeights(Tensor *prevA) const {
        shape4 shape = prevA->dims;
        prevA->reshape(shape.size, 1);
        sgemmNaiveNTA(errors, prevA, deltaWeights);
        prevA->reshape(shape);
    }

    void DenseLayer::recBiases() const {
        add(deltaBiases, errors);
    }

    void DenseLayer::applyWeights(uint32 BATCH_SIZE, float LEARNING_RATE) const {
        *weights - *deltaWeights * (LEARNING_RATE / (float) BATCH_SIZE);
    }

    void DenseLayer::applyBiases(uint32 BATCH_SIZE, float LEARNING_RATE) const {
        *biases - *deltaBiases * (LEARNING_RATE / (float) BATCH_SIZE);
    }

    void DenseLayer::forward(Layer *prev) {
        forwardCalc(prev->a);
    }

    void DenseLayer::backward(Layer *prev) {
        if (strcmp(prev->TYPE,"INPUT") == 0) {
            recWeights(prev->a);
            recBiases();
            return;
        }

        backwardCalc(prev->errors, prev->z);
        recWeights(prev->a);
        recBiases();
    }

    void DenseLayer::backwardOut(Tensor *correct) {
        backwardCalcOut(correct);
    }

    void DenseLayer::learn(float LEARNING_RATE, uint32 BATCH_SIZE) {
        applyWeights(BATCH_SIZE, LEARNING_RATE);
        applyBiases(BATCH_SIZE, LEARNING_RATE);
    }

    void DenseLayer::initialize() {
        logDebug(LOG_SEG_SEANN, "FC initialized with rand normal");
        uint32 K = weights->dims.cols;
        randNormal(weights, 0, (float)sqrt(2.0 / (float) K));
        randNormal(biases, 0, (float)sqrt(2.0 / (float) biases->dims.size));
    }
}


