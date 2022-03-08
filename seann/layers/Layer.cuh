//
// Created by DanielSun on 3/8/2022.
//

#ifndef SEANN_LAYER_CUH
#define SEANN_LAYER_CUH

#include "../../seblas/gemm/Tensor.cuh"
#include "../../seblas/gemm/TensorTools.cuh"
#include "../../seblas/gemm/GEMM.cuh"
#include "../../seblas/gemm/NeuralUtils.cuh"


using namespace seblas;

class Layer {
    public:
    unsigned int inputSize, outputSize;
    Layer* prev{};
    Layer(unsigned int inputSize, unsigned int outputSize){
        this->inputSize = inputSize;
        this->outputSize = outputSize;
    }

    void bind(Layer* pLayer);

    // forward activation
    virtual void forward(Tensor* prev) = 0;

    // backward propagation
    virtual void backward(Tensor* prev, Tensor* next) = 0;

    // update weights
    virtual void learn(float LEARNING_RATE, float BATCH_SIZE) = 0;
};


#endif //SEANN_LAYER_CUH
