//
// Created by DanielSun on 3/15/2022.
//

#include "MaxPoolingLayer.cuh"

void seann::MaxPoolingLayer::forward(seann::Layer *prev) {
    maxPool(prev->a, a, record, stride);
    cudaMemcpy(z->elements, a->elements, z->dims.size * sizeof(float), cudaMemcpyDeviceToDevice);
}

void seann::MaxPoolingLayer::backward(seann::Layer *prev) {
    maxPoolDerive(errors, prev->errors, record, stride);
}

void seann::MaxPoolingLayer::learn(float LEARNING_RATE, uint32 BATCH_SIZE) {
    //This layer do not learn anything
}
