//
// Created by DanielSun on 3/13/2022.
//

#ifndef SEANN_MODEL_CUH
#define SEANN_MODEL_CUH

#include "../layers/std/Layer.cuh"
#include "../../seio/logging/LogUtils.cuh"
#include "Config.cuh"
#include "../layers/std/InputLayer.cuh"
#include "../layers/std/ConvLayer.cuh"
#include "../layers/std/DenseLayer.cuh"
#include "../layers/std/SoftmaxOutLayer.cuh"
#include "../../seio/loader/DataLoader.cuh"
#include "../../seblas/gemm/NeuralUtils.cuh"


using namespace seio;
using namespace seblas;
namespace seann {
    class Model {
    public:
        Tensor* modelOutH, *labelH, *costBuf;
        vector<Layer *> layers;

        vector<Tensor *> dataset[2];  //on host
        vector<Tensor *> labelset[2];   //on host

        vector<Tensor *> dataBatch[2];
        vector<Tensor *> labelBatch[2];
        uint32 DATA_SIZE;

        virtual void registerModel() = 0;

        virtual void loadDataset(Config conf) = 0;

        virtual void loadModel(Config conf) = 0;

        virtual void saveModel(Config conf) = 0;

        virtual void initModel();

        void prepareBatch(Config conf);

        void fetchBatch(Config conf, uint32 batchId, uint32 epochId);

        void train(Config conf);

        virtual float calcSampleCost(Tensor* label) = 0;
    };
}


#endif //SEANN_MODEL_CUH
