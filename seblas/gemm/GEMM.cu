//
// Created by DanielSun on 2/12/2022.
//

#include <cassert>
#include <iostream>
#include "GEMM.cuh"
#include "../assist/DBGTools.cuh"

using namespace seblas;

#define assertGemm(A,B,C) assert((A)->dims.cols == (B)->dims.rows \
&& (A)->dims.rows == (C)->dims.rows && (B)->dims.cols==(C)->dims.cols)

#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__global__ void gemmNaive(Tensor *A, Tensor *B, Tensor *C){
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    float value = 0;
    if(row < C->dims.rows && col < C->dims.cols){
        for (int am = 0; am < A->dims.cols; am++) {
            value += A->getD(row, am) * B->getD(am,col);
        }
        C->setD(row,col,value);
    }
}

template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetchingSR(Tensor *A, Tensor *B, Tensor *C){
    const unsigned int M = A->dims.rows;
    const unsigned int N = B->dims.cols;
    const unsigned int K = A->dims.cols;

    ///allocate smems and registers
    //The shared memory tile
    __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
    __shared__ float tileB[2][BLOCK_K][BLOCK_N];

    float regisA[2][REGIS_M];
    float regisB[2][REGIS_N];
    float regisC[REGIS_M][REGIS_N] = {0};

    const int threadDimX = BLOCK_N / REGIS_N;
    const int threadDimY = BLOCK_M / REGIS_M;
    const int threadCount = threadDimX * threadDimY;
    const int tid = threadIdx.y * threadDimX + threadIdx.x;

    ///register for buffering elements during transporting global to shared mem
    float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
    float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};

    ///prepare configs for reading global
    float* ptrA = A->elements + blockIdx.y * BLOCK_M * K;
    float* ptrB = B->elements + blockIdx.x * BLOCK_N;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_K;
    const int readThreadPerRowB = BLOCK_N;

    //the location each thread should be reading relative to smem
    const int readRowA = tid / readThreadPerRowA;
    const int readColA = tid % readThreadPerRowA;

    const int readRowB = tid / readThreadPerRowB;
    const int readColB = tid % readThreadPerRowB;

    //these values are used to determine the amount of rows to jump
    //if there is the need to do read multiple times
    const int readRowStrideA = threadCount / readThreadPerRowA;
    const int readRowStrideB = threadCount / readThreadPerRowB;

    #pragma unroll
    for(int i=0; i<BLOCK_M; i+= readRowStrideA){
        if(blockM + readRowA + i < M && readColA < K){
            tileA[0][readColA][readRowA+i] = ptrA[(readRowA + i)*K + readColA];
        }
    }

    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideB){
        if(readRowB + i< K && blockN + readColB < N){
            tileB[0][readRowB+i][readColB] = ptrB[(readRowB + i)*N + readColB];
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4(regisA[0][rm]) = toFloat4(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4(regisB[0][rn]) = toFloat4(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }

    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID+=BLOCK_K) {
        //prefetch
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                        ptrA[(readRowA + i) * K + readColA + nextTileID] : 0;
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                bufferB[loadIndex] = readRowB + i + nextTileID < K && blockN + readColB < N ?
                        ptrB[(readRowB + i + nextTileID) * N + readColB] : 0;
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4(regisA[(i + 1) % 2][rm]) = toFloat4(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4(regisB[(i + 1) % 2][rn]) = toFloat4(
                        tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
            }

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn++) {
                    regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                }
            }
        }

        //load the data in the register buffers to tiles
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                tileB[writeStageFlag][readRowB + i][readColB] = bufferB[loadIndex];
            }

            __syncthreads();
            writeStageFlag ^= 1;  //switch
        }
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4(regisA[0][rm]) = toFloat4(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4(regisB[0][rn]) = toFloat4(
                    tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
        }

        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
            }
        }
    }
    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm ++){
        #pragma unroll
        for(int rn = 0; rn < REGIS_N; rn ++){
            if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                     + blockN + threadIdx.x * REGIS_N + rn] = regisC[rm][rn];
            }
        }
    }
}

/**
 * The fast gemm that utilized smem and registers with data prefetching
 * @tparam BLOCK_M block size m
 * @tparam BLOCK_N block size n
 * @tparam BLOCK_K block size k
 * @tparam REGIS_M (the size of the sub matrix of C each thread compute : rows)
 * @tparam REGIS_N (the size of the sub matrix of C each thread compute : cols)
 * @param A
 * @param B
 * @param C
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetching(Tensor *A, Tensor *B, Tensor *C) {

    const unsigned int M = A->dims.rows;
    const unsigned int N = B->dims.cols;
    const unsigned int K = A->dims.cols;

    ///allocate smems and registers
    //The shared memory tile
    __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
    __shared__ float tileB[2][BLOCK_K][BLOCK_N];

    float regisA[2][REGIS_M];
    float regisB[2][REGIS_N];
    float regisC[REGIS_M][REGIS_N] = {0};

    const int threadDimX = BLOCK_N / REGIS_N;
    const int threadDimY = BLOCK_M / REGIS_M;
    const int threadCount = threadDimX * threadDimY;
    const int tid = threadIdx.y * threadDimX + threadIdx.x;

    ///register for buffering elements during transporting global to shared mem
    float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
    float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};

    ///prepare configs for reading global
    float* ptrA = A->elements + blockIdx.y * BLOCK_M * K;
    float* ptrB = B->elements + blockIdx.x * BLOCK_N;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_K / 4;
    const int readThreadPerRowB = BLOCK_N / 4;

    //the location each thread should be reading relative to smem
    const int readRowA = tid / readThreadPerRowA;
    const int readColA = tid % readThreadPerRowA * 4;

    const int readRowB = tid / readThreadPerRowB;
    const int readColB = tid % readThreadPerRowB * 4;

    //these values are used to determine the amount of rows to jump
    //if there is the need to do read multiple times
    const int readRowStrideA = threadCount / readThreadPerRowA;
    const int readRowStrideB = threadCount / readThreadPerRowB;


    ///prefetch the first smem and register block before starting the main loop
    #pragma unroll
    for(int i = 0; i < BLOCK_M; i+=readRowStrideA){
        int loadIndex = i / readRowStrideA * 4;
        if(blockM + readRowA + i < M && readColA < K) {
            toFloat4(bufferA[loadIndex]) = toFloat4(ptrA[(readRowA + i) * K + readColA]);
            //transpose
            tileA[0][readColA][readRowA + i] = bufferA[loadIndex];
            tileA[0][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
            tileA[0][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
            tileA[0][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
        }
    }

    #pragma unroll
    for(int i = 0; i < BLOCK_K; i+=readRowStrideB){
        if(readRowB + i < K && blockN + readColB < N){
            toFloat4(tileB[0][readRowB + i][readColB]) = toFloat4(ptrB[(readRowB + i)*N + readColB]);
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4(regisA[0][rm]) = toFloat4(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4(regisB[0][rn]) = toFloat4(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }

    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID+=BLOCK_K){
        //prefetch
        if(nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                if (blockM + readRowA + i < M && readColA + nextTileID < K) {
                    toFloat4(bufferA[loadIndex]) = toFloat4(
                            ptrA[(readRowA + i) * K + readColA + nextTileID]);
                }else{
                    bufferA[loadIndex] = 0;
                    bufferA[loadIndex+1] = 0;
                    bufferA[loadIndex+2] = 0;
                    bufferA[loadIndex+3] = 0;
                }
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB * 4;
                if (readRowB + i + nextTileID < K && blockN + readColB < N) {
                    toFloat4(bufferB[loadIndex]) = toFloat4(
                            ptrB[(readRowB + i + nextTileID) * N + readColB]);
                } else {
                    bufferA[loadIndex] = 0;
                    bufferA[loadIndex+1] = 0;
                    bufferA[loadIndex+2] = 0;
                    bufferA[loadIndex+3] = 0;
                }
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for(int i = 0; i < BLOCK_K-1; i++){

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4(regisA[(i + 1) % 2][rm]) = toFloat4(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4(regisB[(i + 1) % 2][rn]) = toFloat4(
                        tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
            }

            #pragma unroll
            for(int rm = 0; rm < REGIS_M; rm ++){
                #pragma unroll
                for(int rn = 0; rn < REGIS_N; rn ++){
                    regisC[rm][rn] += regisA[i%2][rm] * regisB[i%2][rn];
                }
            }
        }

        //load the data in the register buffers to tiles
        if(nextTileID < K){
            #pragma unroll
            for(int i=0; i<BLOCK_M; i+=readRowStrideA){
                int loadIndex = i/readRowStrideA * 4;
                tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                tileA[writeStageFlag][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                tileA[writeStageFlag][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                tileA[writeStageFlag][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
            }

            #pragma unroll
            for(int i = 0; i < BLOCK_K; i+=readRowStrideB){
                int loadIndex = i/readRowStrideA * 4;
                toFloat4(tileB[writeStageFlag][readRowB + i][readColB]) = toFloat4(bufferB[loadIndex]);
            }

            __syncthreads();
            writeStageFlag ^= 1;  //switch
        }

        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4(regisA[0][rm]) = toFloat4(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4(regisB[0][rn]) = toFloat4(
                    tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
        }

        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
            }
        }
    }

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm ++){
        #pragma unroll
        for(int rn = 0; rn < REGIS_N; rn += 4){
            if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                toFloat4(C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                     + blockN + threadIdx.x * REGIS_N + rn]) = toFloat4(regisC[rm][rn]);
            }
        }
    }
}

Tensor *seblas::callGemmPrefetching(Tensor *A, Tensor *B, Tensor *C) {

    assert(A->dims.cols == B->dims.rows);
    assert(A->dims.rows == C->dims.rows && B->dims.cols == C->dims.cols);


    dim3 grid = dim3((C->dims.cols + BN - 1) / BN, (C->dims.rows + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    if(A->dims.cols%4==0 && B->dims.cols%4==0){
        gemmPrefetching<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
    } else {
        gemmPrefetchingSR<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
    }
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}

Tensor* seblas::callGemmNaive(Tensor* A, Tensor* B, Tensor* C){
    assertGemm(A,B,C);
    dim3 grid = dim3((C->dims.cols + CUDA_BLOCK_SIZE.x-1)/CUDA_BLOCK_SIZE.x ,
                     (C->dims.rows + CUDA_BLOCK_SIZE.y-1)/CUDA_BLOCK_SIZE.y);
    gemmNaive<<<grid, CUDA_BLOCK_SIZE>>>(A,B,C);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}