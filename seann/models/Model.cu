//
// Created by DanielSun on 3/13/2022.
//

#include "Model.cuh"
#include <random>
#include <thread>

using namespace std;

using namespace seio;
namespace seann {
    
    /** 
     * this method will be running async while the GPU is working on training 
     * the previous batch 
     */
    void Model::fetchBatch(Config conf, uint32* batchIdx, uint32* epochIdx) {
        uint32 batchId = *batchIdx;
        uint32 epochId = *epochIdx;

        uint32 extractFlag = batchId & 0b1;
        uint32 dataFlag = epochId & 0b1;
        assert(!dataset[dataFlag].empty());
        auto generator = std::default_random_engine(batchId);
        bool isLastBatch = dataset[dataFlag].size() < conf.BATCH_SIZE;
        
        for(uint32 count = 0; count < conf.BATCH_SIZE; count++) {
            auto distrib = std::uniform_int_distribution<uint32>(0,dataset[dataFlag].size()-1);
            uint32 index = distrib(generator);

            //when the dataset is smaller than the batch size, we will allow repeating samples in the same batch
            if(count + dataset[dataFlag].size() < conf.BATCH_SIZE) {
                cudaMemcpy(dataBatch[extractFlag][count]->elements, dataset[dataFlag][index]->elements,
                           conf.DATA_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(labelBatch[extractFlag][count]->elements, labelset[extractFlag][index]->elements,
                           conf.LABEL_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
                continue;
            }

            cudaMemcpy(dataBatch[extractFlag][count]->elements, dataset[dataFlag][index]->elements,
                       conf.DATA_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(labelBatch[extractFlag][count]->elements, labelset[extractFlag][index]->elements,
                       conf.LABEL_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);

            //the used elements are going to be transfered to another vector
            //this makes sure every training sample is used once in the epoch
            
            dataset[dataFlag^1].push_back(dataset[dataFlag][index]);
            labelset[dataFlag^1].push_back(labelset[extractFlag][index]);
            dataset[dataFlag].erase(dataset[dataFlag].begin() + index);
            labelset[extractFlag].erase(labelset[extractFlag].begin() + index);
        }

        //update the params
        *batchIdx = batchId + 1;
        *epochIdx = isLastBatch ? *epochIdx + 1 : *epochIdx;
    }

    void Model::prepareBatch(Config conf) {
        for(uint32 i = 0; i < conf.BATCH_SIZE; i++) {
            dataBatch[0].push_back(Tensor::declare(conf.DATA_SHAPE)->create());
            labelBatch[0].push_back(Tensor::declare(conf.LABEL_SHAPE)->create());
        }
    }

    void Model::train(Config conf) {
        registerModel();
        if(conf.LOAD_MODEL_FROM_SAV) loadModel();
        else initModel();

        loadDataset();
        prepareBatch(conf);

        //prefetch the first batch
        uint32 batchId = 0;
        uint32 epochId = 0;
        fetchBatch(conf, &batchId, &epochId);

        while(epochId < conf.EPOCHS){
            //prefetch the next batch while training for this batch
            uint32 currentEpoch = epochId;
            uint32 currentBatch = batchId;
            auto loadThread = thread(&Model::fetchBatch, this, conf, &batchId, &epochId);

            uint32 usingFlag = (batchId & 0b1) ^ 1;
            float batchCost = 0.0f;
            for(uint32 i = 0; i < conf.BATCH_SIZE; i++) {
                ///forward
                layers[0]->a = dataBatch[usingFlag][i];
                for(uint32 layerIdx = 1; layerIdx < layers.size(); layerIdx++) {
                    layers[layerIdx]->forward();
                }
                ///cost
                batchCost += calcSampleCost(labelBatch[usingFlag][i]);

                ///backward
                layers[layers.size()-1]->backwardOut(labelBatch[usingFlag][i]);
                for(uint32 layerIdx = layers.size()-1; layerIdx > 0; layerIdx--) {
                    layers[layerIdx]->backward();
                }
            }

            ///update parameters
            for(uint32 i = 1; i < layers.size(); i++) {
                layers[i]->learn(conf.LEARNING_RATE, conf.BATCH_SIZE);
            }

            //log the batch status for monitoring
            logInfo(LOG_SEG_SEANN, "Epoch: " + to_string(currentEpoch) + ", Batch: " + to_string(currentBatch) +
            ", Cost: " + to_string(batchCost));

            loadThread.join();
        }
    }

    void Model::initModel() {
        logInfo(LOG_SEG_SEANN, "Initializing model...", seio::LOG_COLOR_LIGHT_PURPLE);
        for (uint32 i = 1; i < layers.size(); i++) {
            layers[i]->bind(layers[i-1]);
            layers[i]->initialize();
        }
    }
}

