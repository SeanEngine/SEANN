//
// Created by DanielSun on 3/5/2022.
//

#ifndef SEANN_NEURALUTILS_CUH
#define SEANN_NEURALUTILS_CUH

#include "Tensor.cuh"

/**
 * This code segment is account for activation functions
 */
namespace seblas {

    const unsigned int BLOCK_WARP = 8;

    Tensor* relu(Tensor* input);
    Tensor* reluDerive(Tensor* input);

    Tensor* leakyRelu(Tensor* input, float alpha);
    Tensor* leakyReluDerive(Tensor* input, float alpha);

    Tensor* sigmoid(Tensor* input);
    Tensor* sigmoidDerive(Tensor* input);

    Tensor* tanh(Tensor* input);
    Tensor* tanhDerive(Tensor* input);

    float reduce(Tensor* input, float* buffer);

    Tensor* softmax(Tensor* input);
    Tensor* softmaxDerive(Tensor* input, Tensor* target);
}

#endif //SEANN_NEURALUTILS_CUH
