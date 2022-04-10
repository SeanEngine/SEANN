//
// Created by DanielSun on 3/30/2022.
//

#include "MNISTNN.cuh"

void seann::MNISTNN::registerModel() {
    logInfo(LOG_SEG_SEANN, "------< Registering model : MNIST >------", LOG_COLOR_LIGHT_PURPLE);
    layers.push_back(new InputLayer());
    layers.push_back(new ConvLayer(shape4(4,1,3,3), 28,28,1,1,1,1,false));
    layers.push_back(new ConvLayer(shape4(4,4,3,3), 28,28,1,1,1,1,false));
    layers.push_back(new ConvLayer(shape4(4,4,3,3), 28,28,1,1,1,1,false));
    layers.push_back(new ConvLayer(shape4(4,4,3,3), 28,28,1,1,1,1,false));
    layers.push_back(new DenseLayer(3136, 120));
    layers.push_back(new SoftmaxOutLayer(120, 10));

    modelOutH = Tensor::declare(10,1)->createHost();
    labelH = Tensor::declare(10,1)->createHost();
    costBuf = Tensor::declare(10,1)->create();
}

void seann::MNISTNN::loadDataset(Config conf) {
    DATA_SIZE = 60000;

    fetchIDX(dataset, labelset,
             R"(C:\Users\DanielSun\Desktop\resources\mnist-bin\train-images.idx3-ubyte)",
             R"(C:\Users\DanielSun\Desktop\resources\mnist-bin\train-labels.idx1-ubyte)",
             DATA_SIZE, conf.DATA_SHAPE, conf.LABEL_SHAPE);
}

void seann::MNISTNN::loadModel(Config conf) {

}

void seann::MNISTNN::saveModel(Config conf) {

}

float seann::MNISTNN::calcSampleCost(Tensor *label) {
    auto *softmax = (SoftmaxOutLayer *) layers.back();
    return softmaxCECost(softmax->a, label, costBuf);
}
