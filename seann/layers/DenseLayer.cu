//
// Created by DanielSun on 3/8/2022.
//

#include "DenseLayer.cuh"

namespace seann {
    void DenseLayer::forwardCalc(Tensor *prevA) const {
        shape4 shape = prevA->dims;
        prevA->reshape(shape.size, 1);
        relu((*sgemm(weights, prevA, z) + biases), a);
        prevA->reshape(shape);
    }

    void DenseLayer::backwardCalc(Tensor *prevError, Tensor *prevZ) const {
        *sgemmTN(weights, error, prevError) * reluDerive(prevZ);
    }

    void DenseLayer::backwardCalcOut(Tensor *correct) const {
        *subtract(a, correct, error) * reluDerive(z);
    }

    void DenseLayer::recWeights(Tensor *prevA) const {
        sgemmNTA(error, prevA, deltaWeights);
    }

    void DenseLayer::recBiases() const {
        add(deltaBiases, error);
    }

    void DenseLayer::applyWeights(int BATCH_SIZE, float LEARNING_RATE) const {
        *weights - *deltaWeights * (LEARNING_RATE / (float) BATCH_SIZE);
    }

    void DenseLayer::applyBiases(int BATCH_SIZE, float LEARNING_RATE) const {
        *biases - *deltaBiases * (LEARNING_RATE / (float) BATCH_SIZE);
    }

    void DenseLayer::forward(Layer *prev) {
        forwardCalc(prev->a);
    }

    void DenseLayer::backward(Layer *prev) {
        if (strcmp(prev->TYPE, "DENSE") == 0) {
            auto *proc = (DenseLayer *) prev;
            backwardCalc(proc->error, proc->z);
            recWeights(proc->a);
            recBiases();
        }
    }

    void DenseLayer::backwardOut(Tensor *correct) {
        backwardCalcOut(correct);
    }

    void DenseLayer::learn(float LEARNING_RATE, int BATCH_SIZE) {
        applyWeights(BATCH_SIZE, LEARNING_RATE);
        applyBiases(BATCH_SIZE, LEARNING_RATE);
    }
}


