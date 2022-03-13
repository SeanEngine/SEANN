//
// Created by DanielSun on 3/9/2022.
//

#ifndef SEANN_SOFTMAXOUTLAYER_CUH
#define SEANN_SOFTMAXOUTLAYER_CUH

#include "DenseLayer.cuh"
#include "../../seblas/gemm/TensorTools.cuh"

namespace seann{
    class SoftmaxOutLayer : public DenseLayer {
    public:
    SoftmaxOutLayer(uint32 inputSize, uint32 outputSize)
        : DenseLayer(inputSize, outputSize) {
            TYPE = "SOFTMAX_OUT";
        }

        void forward(Layer *prev) override;

        void backwardOut(Tensor *correct) override;
    };
}


#endif //SEANN_SOFTMAXOUTLAYER_CUH
