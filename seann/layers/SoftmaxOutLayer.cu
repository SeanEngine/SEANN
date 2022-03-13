//
// Created by DanielSun on 3/9/2022.
//

#include "SoftmaxOutLayer.cuh"

void SoftmaxOutLayer::forward(Layer *prev) {
    softmax(*sgemm(weights, prev->a, z) + biases, a);
}

void SoftmaxOutLayer::backwardOut(Tensor *correct) {
    softmaxDeriveCE(a, correct, error);
}
