//
// Created by DanielSun on 3/13/2022.
//

#ifndef SEANN_INPUTLAYER_CUH
#define SEANN_INPUTLAYER_CUH


#include "Layer.cuh"

namespace seann {
    class InputLayer : public Layer {
        public:
        InputLayer() : Layer(){
            TYPE = "INPUT";
        }
    };
}


#endif //SEANN_INPUTLAYER_CUH
