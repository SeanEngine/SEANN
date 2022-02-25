cmake_minimum_required(VERSION 3.20)
project(SEANN CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(OpenCV_DIR D:\\OpenCV_CUDA_4.5.4\\build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SEANN main.cu seblas/gemm/Tensor.cu seblas/gemm/Tensor.cuh seblas/assist/ErrorHandler.cu seblas/assist/ErrorHandler.cuh seblas/gemm/GEMM.cu seblas/gemm/GEMM.cuh seblas/assist/DBGTools.cu seblas/assist/DBGTools.cuh seio/loader/ImageReader.cu seio/loader/ImageReader.cuh)

target_link_libraries(SEANN ${OpenCV_LIBS})

set_target_properties(SEANN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
