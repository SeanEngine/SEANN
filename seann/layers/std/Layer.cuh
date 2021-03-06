//
// Created by DanielSun on 3/8/2022.
//

#ifndef SEANN_LAYER_CUH
#define SEANN_LAYER_CUH

#include "../../../seblas/gemm/Tensor.cuh"
#include "../../../seblas/gemm/TensorTools.cuh"
#include "../../../seblas/gemm/GEMM.cuh"
#include "../../../seblas/gemm/NeuralUtils.cuh"
#include "../../../seio/logging/LogUtils.cuh"
#include "../../../seblas/gemm/Initializers.cuh"
#include <chrono>


using namespace seblas;
using namespace std::chrono;

namespace seann {
    class Layer {
    public:
        Layer *prev{};
        Tensor *a{}, *errors{}, *z{};    //the activation values of each layer
        const char *TYPE{};

        Layer *bind(Layer *pLayer);

        void forward();

        void backward();

        virtual void initialize();

        // forward activation
        virtual void forward(Layer *prev) = 0;

        // backward propagation
        virtual void backward(Layer *prev) = 0;

        virtual void backwardOut(Tensor *correct) = 0;

        // update weights
        virtual void learn(float LEARNING_RATE, uint32 BATCH_SIZE) = 0;
    };
}

#endif //SEANN_LAYER_CUH
