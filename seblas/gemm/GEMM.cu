//
// Created by DanielSun on 2/12/2022.
//

#include <cassert>
#include "GEMM.cuh"
#include "mma.h"
#include "../assist/DBGTools.cuh"

using namespace seblas;
using namespace nvcuda;

#define assertGemm(A,B,C) assert((A)->dims.cols == (B)->dims.rows \
&& (A)->dims.rows == (C)->dims.rows && (B)->dims.cols==(C)->dims.cols)

//DO NOT MODIFY THESE CONSTANTS
//UNLESS YOU ARE AWARE OF WHAT THEY MEANS
#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__global__ void gemmNaive(Tensor *A, Tensor *B, Tensor *C){
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    float value = 0;
    if(row < C->dims.rows && col < C->dims.cols){
        for (int am = 0; am < A->dims.cols; am++) {
            value += A->get(row, am) * B->get(am,col);
        }
        C->set(value, row, col);
    }
}

/**
 * same as the previous gemmPrefetching4NN but do not use float4 in global memory loading
 * to support all matrix dimensions
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetchingNN(Tensor *A, Tensor *B, Tensor *C){
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
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
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
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
 * The first matrix is transposed before computation
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetchingTN(Tensor *A, Tensor *B, Tensor *C){
    const unsigned int M = A->dims.cols;
    const unsigned int N = B->dims.cols;
    const unsigned int K = A->dims.rows;

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
    float* ptrA = A->elements;
    float* ptrB = B->elements;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_M;
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
    for(int i=0; i<BLOCK_K; i+= readRowStrideA){
        if(readRowA + i < K && blockM + readColA < M){
            //The mat A is not transposed since it will be transposed in smem
            tileA[0][readRowA+i][readColA] = ptrA[(readRowA+i)*M + blockM + readColA];
        }
    }

    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideB){
        if(readRowB + i < K && blockN + readColB < N){
            tileB[0][readRowB+i][readColB] = ptrB[(readRowB + i)*N + blockN + readColB];
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }


    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID+=BLOCK_K) {
        //prefetch
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                //here the mat A is automatially transposed while reading
                bufferA[loadIndex] = readRowA + i + nextTileID < K && blockM + readColA < M ?
                                     ptrA[(readRowA + i + nextTileID) * M + blockM + readColA] : 0;
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                bufferB[loadIndex] = readRowB + i +  nextTileID < K && blockN + readColB < N ?
                                     ptrB[(readRowB + i + nextTileID) * N + blockN + readColB] : 0;
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                tileA[writeStageFlag][readRowA + i][readColA] = bufferA[loadIndex];
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
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
 * The first matrix is transposed before computation
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetchingNT(Tensor *A, Tensor *B, Tensor *C){
    const unsigned int M = A->dims.rows;
    const unsigned int N = B->dims.rows;
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
    float* ptrA = A->elements;
    float* ptrB = B->elements;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_K;
    const int readThreadPerRowB = BLOCK_K;

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
            tileA[0][readColA][readRowA+i] = ptrA[(blockM + readRowA + i)*K + readColA];
        }
    }

    #pragma unroll
    for(int i=0; i<BLOCK_N; i+= readRowStrideB){
        if(blockN + readRowB + i < N && readColB < K){
            tileB[0][readColB][readRowB+i] = ptrB[(blockN + readRowB + i)*K + readColB];
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
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
                                     ptrA[(blockM + readRowA + i) * K + readColA + nextTileID] : 0;
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                bufferB[loadIndex] = blockN + readRowB + i < N && readColB + nextTileID < K ?
                                     ptrB[(blockN + readRowB + i) * K + readColB + nextTileID] : 0;
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                tileB[writeStageFlag][readColB][readRowB+i] = bufferB[loadIndex];
            }

            __syncthreads();
            writeStageFlag ^= 1;  //switch
        }
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
__global__ void gemmPrefetching4NN(Tensor *A, Tensor *B, Tensor *C) {

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
            toFloat4R(bufferA[loadIndex]) = toFloat4R(ptrA[(readRowA + i) * K + readColA]);
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
            toFloat4R(tileB[0][readRowB + i][readColB]) = toFloat4R(ptrB[(readRowB + i) * N + readColB]);
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
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
                    toFloat4R(bufferA[loadIndex]) = toFloat4R(
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
                    toFloat4R(bufferB[loadIndex]) = toFloat4R(
                            ptrB[(readRowB + i + nextTileID) * N + readColB]);
                } else {
                    bufferB[loadIndex] = 0;
                    bufferB[loadIndex+1] = 0;
                    bufferB[loadIndex+2] = 0;
                    bufferB[loadIndex+3] = 0;
                }
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for(int i = 0; i < BLOCK_K-1; i++){

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
                toFloat4R(tileB[writeStageFlag][readRowB + i][readColB]) = toFloat4R(bufferB[loadIndex]);
            }

            __syncthreads();
            writeStageFlag ^= 1;  //switch
        }

        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
                toFloat4R(C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                      + blockN + threadIdx.x * REGIS_N + rn]) = toFloat4R(regisC[rm][rn]);
            }
        }
    }
}

/**
 * The first matrix is transposed before computation
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetching4TN(Tensor *A, Tensor *B, Tensor *C){
    const unsigned int M = A->dims.cols;
    const unsigned int N = B->dims.cols;
    const unsigned int K = A->dims.rows;

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
    float* ptrA = A->elements;
    float* ptrB = B->elements;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_M / 4;
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

    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideA){
        if(readRowA + i < K && blockM + readColA < M){
            //The mat A is not transposed since it will be transposed in smem
            toFloat4R(tileA[0][readRowA+i][readColA]) = toFloat4R(ptrA[(readRowA+i)*M + blockM + readColA]);
        }
    }

    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideB){
        if(readRowB + i< K && blockN + readColB < N){
            toFloat4R(tileB[0][readRowB+i][readColB]) = toFloat4R(ptrB[(readRowB + i)*N + blockN + readColB]);
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }


    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID+=BLOCK_K) {
        //prefetch
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                //here the mat A is automatially transposed while reading
                if(readRowA + i + nextTileID < K && blockM + readColA < M ){
                    toFloat4R(bufferA[loadIndex]) = toFloat4R(
                            ptrA[(readRowA + i + nextTileID) * M + blockM + readColA]);
                }else{
                    bufferA[loadIndex] = 0;
                    bufferA[loadIndex + 1] = 0;
                    bufferA[loadIndex + 2] = 0;
                    bufferA[loadIndex + 3] = 0;
                }
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB * 4;
                if(readRowB + i +  nextTileID < K && blockN + readColB < N){
                    toFloat4R(bufferB[loadIndex]) = toFloat4R(
                            ptrB[(readRowB + i + nextTileID) * N + blockN + readColB]);
                }else{
                    bufferB[loadIndex] = 0;
                    bufferB[loadIndex + 1] = 0;
                    bufferB[loadIndex + 2] = 0;
                    bufferB[loadIndex + 3] = 0;
                }
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                toFloat4R(tileA[writeStageFlag][readRowA + i][readColA]) = toFloat4R(bufferA[loadIndex]);
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB * 4;
                toFloat4R(tileB[writeStageFlag][readRowB + i][readColB]) = toFloat4R(bufferB[loadIndex]);
            }

            __syncthreads();
            writeStageFlag ^= 1;  //switch
        }
        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
                toFloat4R(C[(blockM + threadIdx.y * REGIS_M + rm) * N + blockN + threadIdx.x * REGIS_N + rn])
                = toFloat4R(regisC[rm][rn]);
            }
        }
    }
}


template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmPrefetching4NT(Tensor* A, Tensor* B, Tensor* C) {
    const unsigned int M = A->dims.rows;
    const unsigned int N = B->dims.rows;
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
    float *ptrA = A->elements;
    float *ptrB = B->elements;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_K / 4;
    const int readThreadPerRowB = BLOCK_K / 4;

    //the location each thread should be reading relative to smem
    const int readRowA = tid / readThreadPerRowA;
    const int readColA = tid % readThreadPerRowA * 4;

    const int readRowB = tid / readThreadPerRowB;
    const int readColB = tid % readThreadPerRowB * 4;

    //these values are used to determine the amount of rows to jump
    //if there is the need to do read multiple times
    const int readRowStrideA = threadCount / readThreadPerRowA;
    const int readRowStrideB = threadCount / readThreadPerRowB;

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
        int loadIndex = i / readRowStrideA * 4;
        if (blockM + readRowA + i < M && readColA < K) {
            toFloat4R(bufferA[loadIndex]) = toFloat4R(ptrA[(blockM + readRowA + i) * K + readColA]);
            //transpose
            tileA[0][readColA][readRowA + i] = bufferA[loadIndex];
            tileA[0][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
            tileA[0][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
            tileA[0][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
        int loadIndex = i / readRowStrideB * 4;
        if (blockN + readRowB + i < N && readColB < K) {
            toFloat4R(bufferB[loadIndex]) = toFloat4R(ptrB[(blockN + readRowB + i) * K + readColB]);

            tileB[0][readColB][readRowB + i] = bufferB[loadIndex];
            tileB[0][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
            tileB[0][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
            tileB[0][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int rm = 0; rm < REGIS_M; rm += 4) {
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for (int rn = 0; rn < REGIS_N; rn += 4) {
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }

    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for (int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID += BLOCK_K) {
        //prefetch
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA * 4;
                if (blockM + readRowA + i < M && readColA + nextTileID < K) {
                    toFloat4R(bufferA[loadIndex]) = toFloat4R(
                            ptrA[(readRowA + i) * K + readColA + nextTileID]);
                } else {
                    bufferA[loadIndex] = 0;
                    bufferA[loadIndex + 1] = 0;
                    bufferA[loadIndex + 2] = 0;
                    bufferA[loadIndex + 3] = 0;
                }
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB * 4;
                if (blockN + readRowB + i < N && readColB + nextTileID < K) {
                    toFloat4R(bufferB[loadIndex]) =
                            toFloat4R(ptrB[(blockN + readRowB + i) * K + readColB + nextTileID]);
                } else {
                    bufferB[loadIndex] = 0;
                    bufferB[loadIndex + 1] = 0;
                    bufferB[loadIndex + 2] = 0;
                    bufferB[loadIndex + 3] = 0;
                }
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
                int loadIndex = i / readRowStrideA * 4;
                tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                tileA[writeStageFlag][readColA + 1][readRowA + i] = bufferA[loadIndex + 1];
                tileA[writeStageFlag][readColA + 2][readRowA + i] = bufferA[loadIndex + 2];
                tileA[writeStageFlag][readColA + 3][readRowA + i] = bufferA[loadIndex + 3];
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideA * 4;
                tileB[writeStageFlag][readColB][readRowB + i] = bufferB[loadIndex];
                tileB[writeStageFlag][readColB + 1][readRowB + i] = bufferB[loadIndex + 1];
                tileB[writeStageFlag][readColB + 2][readRowB + i] = bufferB[loadIndex + 2];
                tileB[writeStageFlag][readColB + 3][readRowB + i] = bufferB[loadIndex + 3];
            }

            __syncthreads();
            writeStageFlag ^= 1;
        }

        #pragma unroll
        for (int rm = 0; rm < REGIS_M; rm += 4) {
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
                toFloat4R(C->elements[(blockM + threadIdx.y * REGIS_M + rm) * N
                                      + blockN + threadIdx.x * REGIS_N + rn]) = toFloat4R(regisC[rm][rn]);
            }
        }
    }
}


template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmImplicit(Tensor* A, Tensor* B, Tensor* C, int stride, int padH, int padW){

    // MatA: OC, IC * FH * FW; MatB: IC * FH * FW, OH * OW; Mat C: OC, OH * OW
    ///insert parameters
    unsigned const int M = A->dims.n;
    unsigned const int K = A->dims.c * A->dims.rows * A->dims.cols;
    unsigned const int N = C->dims.rows * C->dims.cols;

    unsigned const int FH = A->dims.rows;
    unsigned const int FW = A->dims.cols;
    unsigned const int IH = B->dims.rows;
    unsigned const int IW = B->dims.cols;
    unsigned const int OW = C->dims.cols;

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
    float* ptrB = B->elements;
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

    ///this section is modified from its original state to suit the need for implicit gemm
    ///we are using a special mapping to create patches as trajectories of conv filters
    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideB){
        if(readRowB + i< K && blockN + readColB < N){

            //map buffer matrix cords to the 3 dimensional feature cords
            int oh = (blockN + readColB)/OW;
            int ow = (blockN + readColB)%OW;
            int ic = (readRowB + i)/(FH * FW);
            int fh = ((readRowB + i)%(FH * FW))/FW;
            int fw = ((readRowB + i)%(FH * FW))%FW;
            int ih = oh * stride - padH + fh;
            int iw = ow * stride - padW + fw;
            //do memory access
            tileB[0][readRowB+i][readColB] = ih >= 0 && iw >= 0 && ih < IH && iw < IW ? ptrB[ic * IH * IW + ih * IW + iw] : 0;
        }
    }
    __syncthreads();


    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
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

                //calculate remapping
                int loadIndex = i / readRowStrideB;
                int oh = (blockN + readColB)/OW;
                int ow = (blockN + readColB)%OW;
                int ic = (readRowB + i + nextTileID)/(FH * FW);
                int fh = ((readRowB + i + nextTileID)%(FH * FW))/FW;
                int fw = ((readRowB + i + nextTileID)%(FH * FW))%FW;
                int ih = oh * stride - padH + fh;
                int iw = ow * stride - padW + fw;

                //do memory access
                bufferB[loadIndex] = (readRowB + i + nextTileID < K && blockN + readColB < N) && (ih >= 0 && iw >= 0)
                        && (ih < IH && iw < IW)? ptrB[ic * IH * IW + ih * IW + iw] : 0;
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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
 * This GEMM kernel is for the back propagation of convolutional layers
 * It will produce errors of this layer based on the errors of the next layer
 * the filter matrix will be transposed and directly multiply with the errors
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A
 * @param B
 * @param C
 * @param stride
 * @param padH
 * @param padW
 */
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
        const int REGIS_M, const int REGIS_N>
__global__ void gemmImplicitBackprop(Tensor *A, Tensor*B, Tensor*C, int stride, int padH, int padW) {
     unsigned const int K = A->dims.n; //OC
     unsigned const int M = A->dims.c * A->dims.rows * A->dims.cols;  //FH * FW * IC
     unsigned const int N = C->dims.rows * C->dims.cols; //HW

    unsigned const int FH = A->dims.rows;
    unsigned const int FW = A->dims.cols;
    unsigned const int IH = C->dims.rows;
    unsigned const int IW = C->dims.cols;
    unsigned const int OW = B->dims.cols;
    unsigned const int IC = A->dims.c;

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
    float* ptrA = A->elements;
    float* ptrB = B->elements;
    const int blockM = blockIdx.y * BLOCK_M;
    const int blockN = blockIdx.x * BLOCK_N;

    const int readThreadPerRowA = BLOCK_M;
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
    for(int i=0; i<BLOCK_K; i+= readRowStrideA){
        if(readRowA + i < K && blockM + readColA < M){
            //The mat A is not transposed since it will be transposed in smem
            tileA[0][readRowA+i][readColA] = ptrA[(readRowA+i)*M + blockM + readColA];
        }
    }

    #pragma unroll
    for(int i=0; i<BLOCK_K; i+= readRowStrideB){
        if(readRowB + i< K && blockN + readColB < N){
            tileB[0][readRowB+i][readColB] = ptrB[(readRowB + i)*N + blockN + readColB];
        }
    }
    __syncthreads();

    #pragma unroll
    for(int rm = 0; rm < REGIS_M; rm += 4){
        toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
    }

    #pragma unroll
    for(int rn = 0; rn < REGIS_N; rn += 4){
        toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
    }


    ///main loop
    int writeStageFlag = 1;
    #pragma unroll
    for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K - 1; nextTileID+=BLOCK_K) {
        //prefetch
        if (nextTileID < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                //here the mat A is automatially transposed while reading
                bufferA[loadIndex] = readRowA + i + nextTileID < K && blockM + readColA < M ?
                                     ptrA[(readRowA + i + nextTileID) * M + blockM + readColA] : 0;
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                int loadIndex = i / readRowStrideB;
                bufferB[loadIndex] = readRowB + i +  nextTileID < K && blockN + readColB < N ?
                                     ptrB[(readRowB + i + nextTileID) * N + blockN + readColB] : 0;
            }
        }

        int nextStageFlag = writeStageFlag ^ 1;

        //compute the part that is already in the registers and load the next segment
        #pragma unroll
        for (int i = 0; i < BLOCK_K - 1; i++) {

            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                        tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
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
            for (int i = 0; i < BLOCK_K; i += readRowStrideA) {
                int loadIndex = i / readRowStrideA;
                tileA[writeStageFlag][readRowA + i][readColA] = bufferA[loadIndex];
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
            toFloat4R(regisA[0][rm]) = toFloat4R(
                    tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for (int rn = 0; rn < REGIS_N; rn += 4) {
            toFloat4R(regisB[0][rn]) = toFloat4R(
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

    //run inverse mapping (col2img)
    for(int rm = 0; rm < REGIS_M; rm++){
        for(int rn = 0; rn < REGIS_N; rn++){
            //calculate remapping
            int oh = (blockN + threadIdx.x * REGIS_N + rn)/OW;
            int ow = (blockN + threadIdx.x * REGIS_N + rn)%OW;
            int ic = (blockM + threadIdx.y * REGIS_M + rm)/(FH * FW);
            int fh = ((blockM + threadIdx.y * REGIS_M + rm)%(FH * FW))/FW;
            int fw = ((blockM + threadIdx.y * REGIS_M + rm)%(FH * FW))%FW;
            int ih = oh * stride - padH + fh;
            int iw = ow * stride - padW + fw;

            if(ih >= 0 && ih < IH && iw >= 0 && iw < IW && ic >= 0 && ic < IC){
                atomicAdd(C->elements + ic * IW * IH + ih * IW + iw, regisC[rm][rn]);
            }
        }
    }
}

Tensor* seblas::conv(Tensor *A, Tensor *B, Tensor *C, int stride, int padH, int padW) {
    assert(A->dims.activeDims == 4 && B->dims.activeDims == 3 && C->dims.activeDims == 3);
    assert(C->dims.rows == (B->dims.rows - A->dims.rows + padH*2)/stride + 1);
    assert(C->dims.cols == (B->dims.cols - A->dims.cols + padW*2)/stride + 1);
    assert(C->dims.c == A->dims.n && B->dims.c == A->dims.c);

    unsigned int M = A->dims.n;
    unsigned int N = C->dims.rows * C->dims.cols;

    dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    gemmImplicit<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, stride, padH, padW);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
    return C;
}

//C is the error of this layer and B is for the next layer
Tensor* seblas::convDeriv(Tensor *A, Tensor *B, Tensor *C, int stride, int padH, int padW) {
    assert(A->dims.activeDims == 4 && B->dims.activeDims == 3 && C->dims.activeDims == 3);
    assert(B->dims.rows == (C->dims.rows - A->dims.rows + padH*2)/stride + 1);
    assert(B->dims.cols == (C->dims.cols - A->dims.cols + padW*2)/stride + 1);
    assert(B->dims.c == A->dims.n && C->dims.c == A->dims.c);

    unsigned int M = A->dims.cols * A->dims.rows * A->dims.c;
    unsigned int N = B->dims.rows * B->dims.cols;

    dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    gemmImplicitBackprop<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, stride, padH, padW);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__, __LINE__);
    return C;
}

Tensor* seblas::sgemmNaive(Tensor* A, Tensor* B, Tensor* C){
    assertGemm(A,B,C);
    dim3 grid = dim3((C->dims.cols + CUDA_BLOCK_SIZE.x-1)/CUDA_BLOCK_SIZE.x ,
                     (C->dims.rows + CUDA_BLOCK_SIZE.y-1)/CUDA_BLOCK_SIZE.y);
    gemmNaive<<<grid, CUDA_BLOCK_SIZE>>>(A,B,C);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}

Tensor* seblas::sgemm(Tensor *A, Tensor *B, Tensor *C) {
    assertGemm(A,B,C);
    dim3 grid = dim3((C->dims.cols + BN - 1) / BN, (C->dims.rows + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    if(A->dims.cols%4==0 && B->dims.cols%4==0){
        gemmPrefetching4NN < BM, BN, BK, RM, RN ><<<grid, block>>>(A, B, C);
    } else {
        gemmPrefetchingNN < BM, BN, BK, RM, RN ><<<grid, block>>>(A, B, C);
    }
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}

//gemm with the first matrix automatically transposed
Tensor* seblas::sgemmTN(Tensor *A, Tensor *B, Tensor *C) {
    assert(A->dims.rows == B->dims.rows);
    assert(A->dims.cols == C->dims.rows && B->dims.cols == C->dims.cols);

    dim3 grid = dim3((C->dims.cols + BN - 1) / BN, (C->dims.rows + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    if(A->dims.cols%4==0 && B->dims.cols%4==0){
        gemmPrefetching4TN <BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C);
    } else {
        gemmPrefetchingTN<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
    }
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}

//geem with the second matrix transposed
Tensor* seblas::sgemmNT(Tensor *A, Tensor *B, Tensor *C) {
    assert(A->dims.cols == B->dims.cols);
    assert(A->dims.rows == C->dims.rows && B->dims.rows == C->dims.cols);

    dim3 grid = dim3((C->dims.cols + BN - 1) / BN, (C->dims.rows + BM - 1) / BM);
    dim3 block = dim3(BN / RN, BM / RM);

    if(A->dims.cols%4 == 0 && B->dims.cols%4 == 0){
        gemmPrefetching4NT<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
    }else
        gemmPrefetchingNT<BM, BN, BK, RM, RN><<<grid, block>>>(A,B,C);
    cudaDeviceSynchronize();
    ErrorHandler::checkDeviceStatus(__FILE__,__LINE__);
    return C;
}