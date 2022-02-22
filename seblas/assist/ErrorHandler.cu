//
// Created by DanielSun on 2/12/2022.
//

#include "ErrorHandler.cuh"
#include <iostream>
#include <cassert>

void seblas::ErrorHandler::checkDeviceStatus(const char* file, int line) {
     cudaError_t err = cudaGetLastError();
     if(err!=cudaSuccess){
         std::cout<<"Encountered CUDA Errors : "<<file<<" "<<line<<" "<<cudaGetErrorString(err)<<std::endl;
         assert(err==cudaSuccess);
     }
}
