//
// Created by DanielSun on 3/13/2022.
//

#include "Model.cuh"
#include <random>
#include <thread>
#include "../../seblas/assist/DBGTools.cuh"

using namespace std;

using namespace seio;
namespace seann {
    
    /** 
     * this method will be running async while the GPU is working on training 
     * the previous batch 
     */
    void Model::fetchBatch(Config conf, uint32 batchId, uint32 epochId) {

        uint32 extractFlag = batchId & 0b1;
        uint32 dataReadFlag = epochId & 0b1;
        assert(!dataset[dataReadFlag].empty());
        auto generator = std::default_random_engine(batchId);
        bool isLastBatch = dataset[dataReadFlag].size() < conf.BATCH_SIZE;
        
        for(uint32 count = 0; count < conf.BATCH_SIZE; count++) {
            auto distrib = std::uniform_int_distribution<uint32>(0, dataset[dataReadFlag].size() - 1);
            uint32 index = distrib(generator);

            //when the dataset is smaller than the batch size, we will allow repeating samples in the same batch
            if(count + dataset[dataReadFlag].size() < conf.BATCH_SIZE) {
                cudaMemcpy(dataBatch[extractFlag][count]->elements, dataset[dataReadFlag][index]->elements,
                           conf.DATA_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(labelBatch[extractFlag][count]->elements, labelset[dataReadFlag][index]->elements,
                           conf.LABEL_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
                continue;
            }

            cudaMemcpy(dataBatch[extractFlag][count]->elements, dataset[dataReadFlag][index]->elements,
                       conf.DATA_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(labelBatch[extractFlag][count]->elements, labelset[dataReadFlag][index]->elements,
                       conf.LABEL_SHAPE.size * sizeof(float), cudaMemcpyHostToDevice);

            //the used elements are going to be transfered to another vector
            //this makes sure every training sample is used once in the epoch
            
            dataset[dataReadFlag ^ 1].push_back(dataset[dataReadFlag][index]);
            labelset[dataReadFlag ^ 1].push_back(labelset[dataReadFlag][index]);
            dataset[dataReadFlag].erase(dataset[dataReadFlag].begin() + index);
            labelset[dataReadFlag].erase(labelset[dataReadFlag].begin() + index);
        }
    }

    void Model::prepareBatch(Config conf) {
        for(uint32 i = 0; i < conf.BATCH_SIZE; i++) {
            dataBatch[0].push_back(Tensor::declare(conf.DATA_SHAPE)->create());
            labelBatch[0].push_back(Tensor::declare(conf.LABEL_SHAPE)->create());
            dataBatch[1].push_back(Tensor::declare(conf.DATA_SHAPE)->create());
            labelBatch[1].push_back(Tensor::declare(conf.LABEL_SHAPE)->create());
        }
    }

    void Model::train(Config conf) {
        registerModel();
        if(conf.LOAD_MODEL_FROM_SAV) loadModel(conf);
        else initModel();

        loadDataset(conf);
        prepareBatch(conf);

        //prefetch the first batch
        uint32 batchId = 0;
        uint32 epochId = 0;
        fetchBatch(conf, batchId, epochId);
        batchId++;
        uint32 batches = (DATA_SIZE + conf.BATCH_SIZE - 1)/conf.BATCH_SIZE;

        while(epochId < conf.EPOCHS){
            float epochCost = 0.0f;
            float epochAccuracy = 0;
            //prefetch the next batch while training for this batch
            for(uint32 bID = batchId % batches; bID < batches; bID++) {
                auto loadThread = thread(&Model::fetchBatch, this, conf, batchId, epochId);
                uint32 usingFlag = (batchId & 0b1) ^ 1;
                float batchCost = 0.0f;
                float batchAccuracy = 0;
                for (uint32 i = 0; i < conf.BATCH_SIZE; i++) {
                    ///forward
                    layers[0]->a = dataBatch[usingFlag][i];
                    for (uint32 layerIdx = 1; layerIdx < layers.size(); layerIdx++) {
                        layers[layerIdx]->forward();
                    }

                    auto* proc = (SoftmaxOutLayer*)layers.back();
                    /*
                    if(epochId > 0) {
                        inspect(proc->weights);
                        inspect(proc->z);
                        inspect(proc->a);
                    }
                     */

                    ///cost
                    batchCost += calcSampleCost(labelBatch[usingFlag][i]);
                    cudaMemcpy(modelOutH->elements,layers[layers.size()-1]->a->elements,
                               layers[layers.size()-1]->a->dims.size * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(labelH->elements,labelBatch[usingFlag][i]->elements,
                               labelBatch[usingFlag][i]->dims.size * sizeof(float), cudaMemcpyDeviceToHost);
                    batchAccuracy += (float)judgeCorrection(modelOutH, labelH);

                    ///backward
                    layers[layers.size() - 1]->backwardOut(labelBatch[usingFlag][i]);
                    for (uint32 layerIdx = layers.size() - 1; layerIdx > 0; layerIdx--) {
                        layers[layerIdx]->backward();
                    }
                }

                ///update parameters
                for (uint32 i = 1; i < layers.size(); i++) {
                    layers[i]->learn(conf.LEARNING_RATE, conf.BATCH_SIZE);
                }

                loadThread.join();
                batchId++;

                epochCost += batchCost;
                epochAccuracy += batchAccuracy;

                logTrainingProcess(bID, epochId, batches, conf.EPOCHS
                    , batchCost / (float)conf.BATCH_SIZE, batchAccuracy / (float)conf.BATCH_SIZE
                    , epochCost / (float)((bID+1) * conf.BATCH_SIZE),
                    epochAccuracy / (float)((bID+1) * conf.BATCH_SIZE));
            }
            epochId++;
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

