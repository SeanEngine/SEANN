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

    const uint32 REDUCE_WARP = 8;
    const uint32 SOFTMAX_WARP = 32;

    Tensor* relu(Tensor* input, Tensor* output);
    Tensor* reluDerive(Tensor* input);

    Tensor* leakyRelu(Tensor* input, Tensor* output, float alpha);
    Tensor* leakyReluDerive(Tensor* input, float alpha);

    Tensor* sigmoid(Tensor* input, Tensor* output);
    Tensor* sigmoidDerive(Tensor* input);

    Tensor* tanh(Tensor* input, Tensor* output);
    Tensor* tanhDerive(Tensor* input);

    float reduce(Tensor* input, float* buffer);
    //this is a row reduce operation
    Tensor* convBiasError(Tensor* input, Tensor* deltaBiases, Tensor* buffer);

    Tensor* softmax(Tensor* input, Tensor* output);
    Tensor* softmaxDerive(Tensor* input);

    //CE -> Cross Entropy, Uses only as the output layer
    Tensor* softmaxDeriveCE(Tensor* input, Tensor* target, Tensor* output);

    Tensor* maxPool(Tensor* input, Tensor* output, Tensor* record, uint32 stride);
    Tensor* maxPoolDerive(Tensor* input, Tensor* output, Tensor* record, uint32 stride);

    //for batch normalization
    Tensor* batchNorm(Tensor* input, Tensor* output, Tensor* mean, Tensor* variance, Tensor* gamma, Tensor* beta, float epsilon);
    Tensor* batchNormConv(Tensor* input, Tensor* output, Tensor* mean, Tensor* variance, Tensor* gamma, Tensor* beta, float epsilon);
}

#endif //SEANN_NEURALUTILS_CUH
