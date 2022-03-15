//
// Created by DanielSun on 2/12/2022.
//

#include "ErrorHandler.cuh"
#include <iostream>
#include <cassert>


void seblas::ErrorHandler::checkDeviceStatus(const char* file, int line) {
     cudaError_t err = cudaGetLastError();
     if(err!=cudaSuccess){
         seio::logFatal(seio::LOG_SEG_SEBLAS,string("Encountered CUDA Errors : ") + " line: " +
         to_string(line) + " " + cudaGetErrorString(err) + "\n" + file);
         assert(err==cudaSuccess);
     }
}
