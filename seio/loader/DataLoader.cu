//
// Created by DanielSun on 3/20/2022.
//

#include "DataLoader.cuh"

namespace seio{

    //TODO: THESE MOUNTAINS OF SHIT MUST BE CLEANED AFTER I MAKE CONV WORKING
    void fetchDataLabelThread(int tx, int threads, vector<Tensor*>* dataset, vector<Tensor*>* labels,
                              uchar* bytes, uint32 labelOffset, uint32 dataOffset){
        uint32 stride = labelOffset + dataOffset;
        uint32 dataSize = (*dataset).size();
        uint32 beg = (dataSize/threads) * tx;
        uint32 end = tx == threads - 1 ? dataSize : (dataSize/threads) * (tx + 1);
        for (uint32 i = beg; i < end; i++){
            uchar label = bytes[i * stride];
            readBinPixels(bytes + i * stride + labelOffset, (*dataset)[i]);
            (*labels)[i]->elements[(int)label] = 1;
        }
    }

    void fetchDataThread(int tx, int threads, vector<Tensor*>* dataset,  uchar* bytes,
                         uint32 dataOffset){
        uint32 dataSize = (*dataset).size();
        uint32 beg = (dataSize/threads) * tx;
        uint32 end = tx == threads - 1 ? dataSize : (dataSize/threads) * (tx + 1);
        for (uint32 i = beg; i < end; i++){
            readBinPixels(bytes + i * dataOffset, (*dataset)[i]);
        }
    }

    void fetchLabelThread(int tx, int threads, vector<Tensor*>* labels, const uchar* bytes,
                          uint32 labelOffset){
        uint32 dataSize = (*labels).size();
        uint32 beg = (dataSize/threads) * tx;
        uint32 end = tx == threads - 1 ? dataSize : (dataSize/threads) * (tx + 1);
        for (uint32 i = beg; i < end; i++){
            uchar label = bytes[i * labelOffset];
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
        _alloc<CPU_THREADS>(fetchDataLabelThread, dataset, labels, data, LABEL_OFFSET, DATA_OFFSET);
        cudaFreeHost(data);
        logInfo(LOG_SEG_SEIO,"Dataset CIFAR10 loading complete");
    }

    void fetchIDX(vector<Tensor*>* dataset, vector<Tensor*>* labels, const string& dataPath, const string& labelPath,
                  uint32 DATA_SIZE, shape4 DATA_SHAPE, shape4 LABEL_SHAPE){
        placeHoldDataset(dataset, DATA_SIZE, DATA_SHAPE);
        placeHoldLabelSet(labels, DATA_SIZE, LABEL_SHAPE);

        //load data files
        uchar* dataBytes;
        uchar* labelBytes;
        cudaMallocHost(&dataBytes, sizeof(uchar) * dataset->size() * DATA_SHAPE.size);
        cudaMallocHost(&labelBytes, sizeof(uchar) * dataset->size() * LABEL_SHAPE.size);
        loadBinFile(dataPath.c_str(), dataBytes, dataset->size() * DATA_SHAPE.size);
        loadBinFile(labelPath.c_str(), labelBytes, dataset->size() * LABEL_SHAPE.size);

        logDebug(LOG_SEG_SEIO, "Fetching IDX data", LOG_COLOR_LIGHT_YELLOW);
        logDebug(LOG_SEG_SEIO, "Expected Data Size : " + to_string(DATA_SIZE),
                 LOG_COLOR_LIGHT_YELLOW);
        //Skip the headers
        dataBytes += 16 * sizeof(uchar);
        labelBytes += 8 * sizeof(uchar);

        //Fetch data
        _alloc<CPU_THREADS>(fetchDataThread, dataset, dataBytes, DATA_SHAPE.size);
        _alloc<CPU_THREADS>(fetchLabelThread, labels, labelBytes, 1); //Each byte is a label value

        cudaFreeHost(dataBytes - 16 * sizeof(uchar));
        cudaFreeHost(labelBytes - 8 * sizeof(uchar));
    }
}