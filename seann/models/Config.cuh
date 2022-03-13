//
// Created by DanielSun on 3/13/2022.
//

#ifndef SEANN_CONFIG_CUH
#define SEANN_CONFIG_CUH

#include "../../seblas/gemm/Tensor.cuh"

using namespace seblas;

namespace seann{
    struct Config{
        const char* TRAIN_DATA_PATH{};
        const char* TEST_DATA_PATH{};
        const char* MODEL_SAV_PATH{};

        const uint32 BATCH_SIZE{};
        const float LEARNING_RATE{};
        const uint32 EPOCHS{};

        const shape4 DATA_SHAPE;
        const shape4 LABEL_SHAPE;

        const bool LOAD_MODEL_FROM_SAV = false;
    };
}

#endif //SEANN_CONFIG_CUH
