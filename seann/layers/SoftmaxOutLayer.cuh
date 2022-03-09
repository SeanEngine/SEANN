//
// Created by DanielSun on 3/9/2022.
//

#ifndef SEANN_SOFTMAXOUTLAYER_CUH
#define SEANN_SOFTMAXOUTLAYER_CUH

#include "DenseLayer.cuh"

class SoftmaxOutLayer : public DenseLayer {
public:

    void forward(Layer *prev) override;

    void backwardOut(Tensor* correct) override;
};


#endif //SEANN_SOFTMAXOUTLAYER_CUH
