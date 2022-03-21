//
// Created by DanielSun on 3/13/2022.
//

#ifndef SEANN_INPUTLAYER_CUH
#define SEANN_INPUTLAYER_CUH


#include "Layer.cuh"

namespace seann {
    class InputLayer : public Layer {
    public:
        InputLayer() {
            TYPE = "INPUT";
        }

        void forward(Layer *prev) override;

        void backward(Layer *prev) override;

        void backwardOut(Tensor *correct) override;

        void learn(float LEARNING_RATE, uint32 BATCH_SIZE) override;
    };
}


#endif //SEANN_INPUTLAYER_CUH
