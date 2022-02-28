//
// Created by DanielSun on 2/28/2022.
//

#ifndef SEANN_TENSORTOOLS_CUH
#define SEANN_TENSORTOOLS_CUH
#include "Tensor.cuh"


namespace seblas{
    enum DimPrefix{
        DEPTH,
        ROW,
        COL,
    };

    struct range{
        int a,b;
        range(int a, int b){
            this->a = a; this->b = b;
        }
    };

    Tensor* slice(Tensor* in, Tensor* buffer, DimPrefix dim,range dimRange);

    Tensor* extract4(Tensor* in, Tensor* buffer, shape4 start, shape4 extractRange);
}


#endif //SEANN_TENSORTOOLS_CUH
