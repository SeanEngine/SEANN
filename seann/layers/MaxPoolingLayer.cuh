//
// Created by DanielSun on 3/15/2022.
//

#ifndef SEANN_MAXPOOLINGLAYER_CUH
#define SEANN_MAXPOOLINGLAYER_CUH

#include "Layer.cuh"
using namespace seblas;
namespace seann {
    class MaxPoolingLayer : public Layer {
    public:
        Tensor* record;
        uint32 stride;

        MaxPoolingLayer(shape4 inDim, uint32 stride){
            record = Tensor::declare(inDim)->create();
            a = Tensor::declare(inDim.c, inDim.rows/stride, inDim.cols/stride)->create();
            errors = Tensor::declare(a->dims)->create();
            z = Tensor::declare(a->dims)->create();
            this->stride = stride;
        }

        void forward(Layer *prev) override;

        void backward(Layer *prev) override;

        void learn(float LEARNING_RATE, uint32 BATCH_SIZE) override;
    };
}


#endif //SEANN_MAXPOOLINGLAYER_CUH
