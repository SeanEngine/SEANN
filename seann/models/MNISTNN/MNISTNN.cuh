//
// Created by DanielSun on 3/30/2022.
//

#ifndef SEANN_MNISTNN_CUH
#define SEANN_MNISTNN_CUH

#include "../Model.cuh"

namespace seann {
    class MNISTNN : public Model {
    public:
        void registerModel() override;

        void loadDataset(Config conf) override;

        void loadModel(Config conf) override;

        void saveModel(Config conf) override;

        float calcSampleCost(Tensor *label) override;
    };
}

#endif //SEANN_MNISTNN_CUH
