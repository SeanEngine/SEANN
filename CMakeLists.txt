cmake_minimum_required(VERSION 3.20)
project(SEANN CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(SEANN main.cu seblas/gemm/TensorOp.cu seblas/gemm/TensorOp.cuh)

set_target_properties(SEANN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
