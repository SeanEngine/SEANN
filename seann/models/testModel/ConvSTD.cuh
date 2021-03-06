//
// Created by DanielSun on 3/21/2022.
//

#ifndef SEANN_CONVSTD_CUH
#define SEANN_CONVSTD_CUH

#include "../Model.cuh"
#include "../../layers/std/SoftmaxOutLayer.cuh"

namespace seann {
    class ConvSTD : public Model {
    public:
        void registerModel() override;

        void loadDataset(Config conf) override;

        void loadModel(Config conf) override;

        void saveModel(Config conf) override;

        float calcSampleCost(Tensor *label) override;
    };
}


#endif //SEANN_CONVSTD_CUH
