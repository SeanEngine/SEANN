//
// Created by DanielSun on 3/21/2022.
//

#include "ConvSTD.cuh"


namespace seann {
    void seann::ConvSTD::registerModel() {
        logInfo(LOG_SEG_SEANN, "------< Registering model : ConvSTD >------", LOG_COLOR_LIGHT_PURPLE);
        layers.push_back(new InputLayer());\

        layers.push_back(new ConvLayer(shape4(8, 3, 3, 3), 32, 32, 1, 1, 1, 1, false));
        layers.push_back(new ConvLayer(shape4(8, 8, 3, 3), 32, 32, 1, 1, 1, 1, false));
        layers.push_back(new ConvLayer(shape4(8, 8, 3, 3), 32, 32, 1, 1, 1, 1, false));
        layers.push_back(new ConvLayer(shape4(8, 8, 3, 3), 32, 32, 1, 1, 1, 1, false));
        layers.push_back(new ConvLayer(shape4(16, 8, 3, 3), 32, 32, 2, 2, 1, 1, false));
        layers.push_back(new DenseLayer(3072, 1024));
        layers.push_back(new DenseLayer(1024, 1024));
        layers.push_back(new DenseLayer(1024, 1024));
        layers.push_back(new SoftmaxOutLayer(1024, 10));

        modelOutH = Tensor::declare(10, 1)->createHost();
        labelH = Tensor::declare(10, 1)->createHost();
        costBuf = Tensor::declare(10, 1)->create();
    }

    void seann::ConvSTD::loadDataset(Config conf) {
        vector<string> filenames = {
                R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_1.bin)",
                R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_2.bin)",
                R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_3.bin)",
                R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_4.bin)",
                R"(C:\Users\DanielSun\Desktop\resources\MLDatasets\cifar-10-batches-bin\data_batch_5.bin)"
        };
        fetchCIFAR10(dataset, labelset, filenames);
        DATA_SIZE = dataset[0].size();
    }

    void seann::ConvSTD::loadModel(Config conf) {

    }

    void seann::ConvSTD::saveModel(Config conf) {

    }

    float seann::ConvSTD::calcSampleCost(Tensor *label) {
        auto *softmax = (SoftmaxOutLayer *) layers.back();
        return softmaxCECost(softmax->a, label, costBuf);
    }
}