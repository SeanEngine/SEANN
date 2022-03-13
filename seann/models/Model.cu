//
// Created by DanielSun on 3/13/2022.
//

#include "Model.cuh"
#include <random>

namespace seann {
    
    /** 
     * this method will be running async while the GPU is working on training 
     * the previous batch 
     */
    void Model::fetchBatch(Config conf, uint32 batchId, uint32 epochId) {
        uint32 extractFlag = batchId & 0b1;
        uint32 dataFlag = epochId & 0b1;
        auto generator = std::default_random_engine(batchId);
        
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
    }

    void Model::prepareBatch(Config conf) {
        for(uint32 i = 0; i < conf.BATCH_SIZE; i++) {
            dataBatch[0].push_back(Tensor::declare(conf.DATA_SHAPE)->create());
            labelBatch[0].push_back(Tensor::declare(conf.LABEL_SHAPE)->create());
        }
    }

    void Model::train(Config conf) {

    }
}

