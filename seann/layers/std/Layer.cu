//
// Created by DanielSun on 3/8/2022.
//

#include "Layer.cuh"

namespace seann {
    Layer *Layer::bind(Layer *pLayer) {
        this->prev = pLayer;
        return this;
    }

    void Layer::forward() {
        forward(prev);
    }

    void Layer::backward() {
        backward(prev);
    }

    void Layer::initialize() {

    }
}
