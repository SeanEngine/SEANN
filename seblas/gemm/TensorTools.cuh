//
// Created by DanielSun on 2/28/2022.
//

#ifndef SEANN_TENSORTOOLS_CUH
#define SEANN_TENSORTOOLS_CUH
#include "Tensor.cuh"

namespace seblas{

    struct range4{
        index4 began;
        index4 end;
        index4 diff;

        __device__ __host__ range4(index4 b, index4 e)
             : began(b), end(e), diff(e-b){
        }
    };

    Tensor* slice(Tensor* in, Tensor* buffer, range4 range);

    Tensor* extract4(Tensor* in, Tensor* buffer, range4 copyOff);

    Tensor* extract(Tensor* in, Tensor* buffer, range4 copyOff);
}


#endif //SEANN_TENSORTOOLS_CUH
