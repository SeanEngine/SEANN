//
// Created by DanielSun on 3/22/2022.
//

#ifndef SEANN_INITIALIZERS_CUH
#define SEANN_INITIALIZERS_CUH

#include "Tensor.cuh"
#include <chrono>

using namespace std::chrono;
namespace seblas{
    Tensor* randNormal(Tensor* A, float mean, float stddev);

    Tensor* randUniform(Tensor* A, float min, float max);
}

#endif //SEANN_INITIALIZERS_CUH
