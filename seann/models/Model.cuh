//
// Created by DanielSun on 3/13/2022.
//

#ifndef SEANN_MODEL_CUH
#define SEANN_MODEL_CUH

#include "../layers/Layer.cuh"
#include "Config.cuh"

namespace seann {
    class Model {
    public:
        vector<Layer *> layers;

        vector<Tensor *> dataset[2];  //on host
        vector<Tensor *> labelset[2];   //on host

        vector<Tensor *> dataBatch[2];
        vector<Tensor *> labelBatch[2];

        virtual void registerModel() = 0;

        virtual void loadDataset() = 0;

        virtual void loadModel() = 0;

        virtual void saveModel() = 0;

        virtual void initModel() = 0;

        void prepareBatch(Config conf);

        void fetchBatch(Config conf, uint32* batchId, uint32* epochId);

        void train(Config conf);

        virtual float calcSampleCost(Tensor* label) = 0;
    };
}


#endif //SEANN_MODEL_CUH
