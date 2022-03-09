//
// Created by DanielSun on 3/8/2022.
//

#ifndef SEANN_DENSELAYER_CUH
#define SEANN_DENSELAYER_CUH

#include "Layer.cuh"

using namespace seblas;

class DenseLayer : public Layer {
public:
    Tensor* weights, *biases, *deltaWeights, *deltaBiases;
    Tensor* z, *a, *error;

    DenseLayer(unsigned int inputSize, unsigned int outputSize)
    : Layer(inputSize, outputSize) {

        z = Tensor::declare(outputSize,1)->create();
        a = Tensor::declare(outputSize,1)->create();
        error = Tensor::declare(outputSize,1)->create();

        weights = Tensor::declare(outputSize, inputSize)->create();
        biases = Tensor::declare(outputSize, 1)->create();

        deltaWeights = Tensor::declare(outputSize, inputSize)->create();
        deltaBiases = Tensor::declare(outputSize, 1)->create();
    }

    //forward activation
    void forwardCalc(Tensor* prevA) const;

    //calculate the error of the previous layer
    //from the error of this layer
    void backwardCalc(Tensor* prevError, Tensor* prevZ) const;

    void backwardCalcOut(Tensor* correct) const;

    void recWeights(Tensor* prevA) const;

    void recBiases() const;
};


#endif //SEANN_DENSELAYER_CUH
