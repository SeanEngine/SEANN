//
// Created by DanielSun on 3/8/2022.
//

#ifndef SEANN_DENSELAYER_CUH
#define SEANN_DENSELAYER_CUH

#include "Layer.cuh"

using namespace seblas;
using namespace seio;

namespace seann {
    class DenseLayer : public Layer {
    public:
        Tensor *weights, *biases, *deltaWeights, *deltaBiases;
        Tensor *z, *errors;

        DenseLayer(unsigned int inputSize, unsigned int outputSize) {
            TYPE = "DENSE";

            z = Tensor::declare(outputSize, 1)->create();
            a = Tensor::declare(outputSize, 1)->create();
            errors = Tensor::declare(outputSize, 1)->create();

            weights = Tensor::declare(outputSize, inputSize)->create();
            biases = Tensor::declare(outputSize, 1)->create();

            deltaWeights = Tensor::declare(outputSize, inputSize)->create();
            deltaBiases = Tensor::declare(outputSize, 1)->create();

            logInfo(LOG_SEG_SEANN, "Registered FC: Input:" + to_string(inputSize) +
                   " Output:" + to_string(outputSize));
            logDebug(LOG_SEG_SEANN,"Current total memory occupation : " +
                to_string(MEMORY_OCCUPATION/(1024*1024)));
        }

        //forward activation
        void forwardCalc(Tensor *prevA) const;

        //calculate the errors of the previous layer
        //from the errors of this layer
        void backwardCalc(Tensor *prevError, Tensor *prevZ) const;

        void backwardCalcOut(Tensor *correct) const;

        void recWeights(Tensor *prevA) const;

        void recBiases() const;

        void applyWeights(uint32 BATCH_SIZE, float LEARNING_RATE) const;

        void applyBiases(uint32 BATCH_SIZE, float LEARNING_RATE) const;

        //universal methods
        void forward(Layer *prev) override;

        void backward(Layer *prev) override;

        void backwardOut(Tensor *correct) override;

        void learn(float LEARNING_RATE, uint32 BATCH_SIZE) override;
    };
}

#endif //SEANN_DENSELAYER_CUH
