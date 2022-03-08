//
// Created by DanielSun on 3/8/2022.
//

#include "DenseLayer.cuh"

void DenseLayer::forwardCalc(Tensor *prevA) const {
    relu((*sgemm(weights, prevA, z) + biases),a);
}

void DenseLayer::backwardCalc(Tensor *prevError, Tensor *prevZ) const{
    *sgemmTN(weights, error, prevError) * reluDerive(prevZ);
}

void DenseLayer::backwardCalcOut(Tensor *correct) const {
    *subtract(a, correct, error) * reluDerive(z);
}

