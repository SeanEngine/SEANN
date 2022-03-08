//
// Created by DanielSun on 2/28/2022.
//

#ifndef SEANN_TENSORTOOLS_CUH
#define SEANN_TENSORTOOLS_CUH
#include "Tensor.cuh"

/**
 * These methods are tools that are crucial for tensor operations
 * However, they are not performance-consuming and does not need to be
 * deeply optimized for now
 *
 * many of these are used in the Tensor operators
 * every method ended in D means they are global functions running on device
 * every method ended in 4 means they use float4 to accelerate memory reading
 *   *- theses methods needs cols to be a multiple of 4
 */
namespace seblas {

    struct range4 {
        index4 began;
        index4 end;
        index4 diff;

        __device__ __host__ range4(index4 b, index4 e)
                : began(b), end(e), diff(e - b) {
        }
    };

    //A function I learned from numpy
    //using memcpy to move entire dimensions of a matrix to another
    Tensor *slice(Tensor *in, Tensor *buffer, range4 range);

    Tensor *add(Tensor *in, Tensor *other);

    Tensor *subtract(Tensor *in, Tensor *other);

    Tensor *subtract(Tensor *A, Tensor* B, Tensor *C);

    Tensor *hadamardProduct(Tensor *in, Tensor *other);

    Tensor *constProduct(Tensor *in, float val);

    Tensor *transpose(Tensor *in, Tensor *out);
}


#endif //SEANN_TENSORTOOLS_CUH
