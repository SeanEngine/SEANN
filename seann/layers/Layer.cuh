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
    Tensor* a{};    //the activation values of each layer
    const char * TYPE{};

    Layer(unsigned int inputSize, unsigned int outputSize){
        this->inputSize = inputSize;
        this->outputSize = outputSize;
    }

    Layer* bind(Layer* pLayer);

    void forward();

    void backward();

    // forward activation
    virtual void forward(Layer* prev) = 0;

    // backward propagation
    virtual void backward(Layer* prev) = 0;

    virtual void backwardOut(Tensor* correct) = 0;

    // update weights
    virtual void learn(float LEARNING_RATE, int BATCH_SIZE) = 0;
};


#endif //SEANN_LAYER_CUH
