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

    const unsigned int REDUCE_WARP = 8;
    const unsigned int SOFTMAX_WARP = 32;

    Tensor* relu(Tensor* input, Tensor* output);
    Tensor* reluDerive(Tensor* input);

    Tensor* leakyRelu(Tensor* input, Tensor* output, float alpha);
    Tensor* leakyReluDerive(Tensor* input, float alpha);

    Tensor* sigmoid(Tensor* input, Tensor* output);
    Tensor* sigmoidDerive(Tensor* input);

    Tensor* tanh(Tensor* input, Tensor* output);
    Tensor* tanhDerive(Tensor* input);

    float reduce(Tensor* input, float* buffer);

    Tensor* softmax(Tensor* input, Tensor* output);

    //CE -> Cross Entropy, Uses only as the output layer
    Tensor* softmaxDeriveCE(Tensor* input, Tensor* target);
}

#endif //SEANN_NEURALUTILS_CUH
