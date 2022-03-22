//
// Created by DanielSun on 3/20/2022.
//

#include "DataLoader.cuh"

namespace seio{

    void fetchDataBinThread(int tx, int threads, vector<Tensor*>* dataset, vector<Tensor*>* labels,
                            uchar* bytes, uint32 labelOffset, uint32 dataOffset){
        uint32 stride = labelOffset + dataOffset;
        uint32 dataSize = (*dataset).size();
        uint32 beg = (dataSize/threads) * tx;
        uint32 end = tx == threads - 1 ? dataSize : (dataSize/threads) * (tx + 1);
        for (uint32 i = beg; i < end; i++){
            uchar label = bytes[i * stride];
            readBinRGB(bytes + i * stride + labelOffset, (*dataset)[i]);
            (*labels)[i]->elements[(int)label] = 1;
        }
    }

    void placeHoldDataset(vector<Tensor*>* dataset, uint32 DATA_SIZE, shape4 DATA_SHAPE){
        for(uint32 i = 0; i < DATA_SIZE; i++){
            dataset->push_back(Tensor::declare(DATA_SHAPE)->createHost());
        }
    }

    void placeHoldLabelSet(vector<Tensor*>* labels, uint32 LABEL_SIZE, shape4 LABEL_SHAPE){
        for(uint32 i = 0; i < LABEL_SIZE; i++){
            labels->push_back(Tensor::declare(LABEL_SHAPE)->createHost());
        }
    }

    void fetchCIFAR10(vector<Tensor*>* dataset, vector<Tensor*>* labels, vector<string> filenames){
        uint32 DATA_OFFSET = 3072;
        uint32 LABEL_OFFSET = 1;
        placeHoldDataset(dataset, 50000, shape4(3,32,32));
        placeHoldLabelSet(labels, 50000, shape4(10,1));
        uchar* data;
        cudaMallocHost(&data, sizeof(uchar) * dataset->size() * (LABEL_OFFSET + DATA_OFFSET));
        uint32 stride = (dataset->size() / filenames.size()) * (LABEL_OFFSET + DATA_OFFSET);
        for(int i = 0; i < filenames.size(); i++){
            logDebug(LOG_SEG_SEIO, "Fetching CIFAR10 binary datafile : " + filenames[i],
                     LOG_COLOR_LIGHT_YELLOW);
            loadBinFile(filenames[i].c_str(), data + i * stride, stride);
        }
        _alloc<CPU_THREADS>(fetchDataBinThread, dataset, labels, data, LABEL_OFFSET, DATA_OFFSET);
        cudaFreeHost(data);
        logInfo(LOG_SEG_SEIO,"Dataset CIFAR10 loading complete");
    }
}