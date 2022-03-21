//
// Created by DanielSun on 3/21/2022.
//

#ifndef SEANN_CONVSTD_CUH
#define SEANN_CONVSTD_CUH

#include "../Model.cuh"

namespace seann {
    class ConvSTD : public Model {
    public:
        void registerModel() override;

        void loadDataset() override;

        void loadModel() override;

        void saveModel() override;

        void initModel() override;

        float calcSampleCost(Tensor *label) override;
    };
}


#endif //SEANN_CONVSTD_CUH
