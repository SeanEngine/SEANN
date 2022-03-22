cmake_minimum_required(VERSION 3.20)
project(SEANN CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(OpenCV_DIR D:\\OpenCV_CUDA_4.5.4\\build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SEANN main.cu seblas/gemm/Tensor.cu seblas/gemm/Tensor.cuh seblas/assist/ErrorHandler.cu seblas/assist/ErrorHandler.cuh seblas/gemm/GEMM.cu seblas/gemm/GEMM.cuh seblas/assist/DBGTools.cu seblas/assist/DBGTools.cuh seio/loader/ImageReader.cu seio/loader/ImageReader.cuh seblas/gemm/TensorTools.cu seblas/gemm/TensorTools.cuh seblas/gemm/NeuralUtils.cu seblas/gemm/NeuralUtils.cuh seann/layers/std/Layer.cu seann/layers/std/Layer.cuh seann/layers/std/DenseLayer.cu seann/layers/std/DenseLayer.cuh seann/layers/std/SoftmaxOutLayer.cu seann/layers/std/SoftmaxOutLayer.cuh seann/layers/std/ConvLayer.cu seann/layers/std/ConvLayer.cuh seann/models/Model.cu seann/models/Model.cuh seann/models/Config.cuh seann/layers/std/InputLayer.cu seann/layers/std/InputLayer.cuh seio/logging/LogUtils.cu seio/logging/LogUtils.cuh seio/logging/Color.cuh seio/logging/Color.cu seann/models/Resnet50/Resnet50.cu seann/models/Resnet50/Resnet50.cuh seann/models/Resnet50/Resnet50Config.cuh seann/layers/std/MaxPoolingLayer.cu seann/layers/std/MaxPoolingLayer.cuh seio/loader/DataLoader.cu seio/loader/DataLoader.cuh sexec/threading/ThreadController.cu sexec/threading/ThreadController.cuh seann/models/testModel/ConvSTD.cu seann/models/testModel/ConvSTD.cuh seblas/gemm/Initializers.cu seblas/gemm/Initializers.cuh)

target_link_libraries(SEANN ${OpenCV_LIBS})

set_target_properties(SEANN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
