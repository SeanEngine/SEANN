//
// Created by DanielSun on 3/22/2022.
//

#ifndef SEANN_INITIALIZERS_CUH
#define SEANN_INITIALIZERS_CUH

#include "Tensor.cuh"

namespace seblas{
    Tensor* randNormal(Tensor* A, float mean, float stddev, long seed);

    Tensor* randUniform(Tensor* A, float min, float max, long seed);


}

#endif //SEANN_INITIALIZERS_CUH
