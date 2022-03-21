//
// Created by DanielSun on 3/20/2022.
//

#ifndef SEANN_DATALOADER_CUH
#define SEANN_DATALOADER_CUH

#include "../../seblas/gemm/Tensor.cuh"
#include <vector>
#include "../logging/LogUtils.cuh"
#include "ImageReader.cuh"
#include "../../sexec/threading/ThreadController.cuh"

#define CPU_THREADS 12

using namespace std;
using namespace sexec;
using namespace seblas;

namespace seio{

    //allocate dataset place hold to memory
    void placeHoldDataset(vector<Tensor*>* dataset, uint32 DATA_SIZE, shape4 DATA_SHAPE);

    void placeHoldLabelSet(vector<Tensor*>* labels, uint32 LABEL_SIZE, shape4 LABEL_SHAPE);

    void fetchCIFAR10(vector<Tensor*>* dataset, vector<Tensor*>* labels, vector<string> filenames);
}

#endif //SEANN_DATALOADER_CUH
