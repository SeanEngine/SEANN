//
// Created by DanielSun on 2/12/2022.
//

#include "ErrorHandler.cuh"
#include <iostream>
#include <cassert>

void seblas::ErrorHandler::checkDeviceStatus() {
     cudaError_t err = cudaGetLastError();
     if(err!=cudaSuccess){
         std::cout<<"Encountered CUDA Errors : "<<__FILE__<<" "<<__LINE__<<cudaGetErrorString(err)<<std::endl;
         assert(err==cudaSuccess);
     }
}
