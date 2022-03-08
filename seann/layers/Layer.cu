//
// Created by DanielSun on 3/8/2022.
//

#include "Layer.cuh"

void Layer::bind(Layer *pLayer) {
    this->prev = pLayer;
}
