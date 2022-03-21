//
// Created by DanielSun on 3/21/2022.
//

#include "ConvSTD.cuh"

void seann::ConvSTD::registerModel() {
    logInfo(LOG_SEG_SEANN, "------< Registering model : ConvSTD >------");
    layers.push_back(new InputLayer());
    layers.push_back(new ConvLayer(shape4(8, 3, 3, 3), 32, 32, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(8, 8, 3, 3), 32, 32, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(8, 8, 3, 3), 32, 32, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(16, 8, 3, 3), 32, 32, 2, 2, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(16, 16, 3, 3), 16, 16, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(16, 16, 3, 3), 16, 16, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(32, 16, 3, 3), 16, 16, 2, 2, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(32, 32, 3, 3), 8, 8, 1, 1, 1, 1, true));
    layers.push_back(new ConvLayer(shape4(32, 32, 3, 3), 8, 8, 1, 1, 1, 1, true));
    layers.push_back(new DenseLayer(2048,1024));
    layers.push_back(new DenseLayer(1024,1024));
    layers.push_back(new DenseLayer(1024,1024));
    layers.push_back(new SoftmaxOutLayer(1024,10));
}

void seann::ConvSTD::loadDataset() {
    vector<string> filenames = {
            R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_1.bin)",
            R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_2.bin)",
            R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_3.bin)",
            R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_4.bin)",
            R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_5.bin)"
    };
    fetchCIFAR10(dataset, labelset, filenames);
}

void seann::ConvSTD::loadModel() {

}

void seann::ConvSTD::saveModel() {

}

void seann::ConvSTD::initModel() {

}

float seann::ConvSTD::calcSampleCost(Tensor *label) {
    return 0;
}
