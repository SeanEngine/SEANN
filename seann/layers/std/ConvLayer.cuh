//
// Created by DanielSun on 3/9/2022.
//

#ifndef SEANN_CONVLAYER_CUH
#define SEANN_CONVLAYER_CUH


#include "Layer.cuh"
#include "../../../seblas/gemm/NeuralUtils.cuh"
using namespace seio;
namespace seann {
    class ConvLayer : public Layer {
    public:
        Tensor *filters, *biases, *deltaFilters, *deltaBiases, *buffer;
        bool enableBias;

        int strideH, strideW, padH, padW;

        ConvLayer(shape4 filterSize, uint32 ih, uint32 iw, int strideH, int strideW, int padH, int padW,
                  bool enableBiases) {
            TYPE = "CONV";

            uint32 oh = (ih - filterSize.rows + 2 * padH) / strideH + 1;
            uint32 ow = (iw - filterSize.cols + 2 * padW) / strideW + 1;
            this->enableBias = enableBiases;

            filters = Tensor::declare(filterSize)->create();
            biases = enableBiases ? Tensor::declare(filterSize.n, 1)->create() : nullptr;

            deltaFilters = Tensor::declare(filterSize)->create();
            deltaBiases = enableBiases ? Tensor::declare(filterSize.n, 1)->create() : nullptr;

            a = Tensor::declare(filterSize.n, oh, ow)->create();
            z = Tensor::declare(filterSize.n, oh, ow)->create();
            errors = Tensor::declare(filterSize.n, oh, ow)->create();

            uint32 bufCols = (oh * ow + WARP_SIZE * REDUCE_WARP - 1) / (WARP_SIZE * REDUCE_WARP);
            buffer = enableBiases ? Tensor::declare(filterSize.n, bufCols)->create() : nullptr;

            this->strideH = strideH;
            this->strideW = strideW;
            this->padH = padH;
            this->padW = padW;

            logInfo(LOG_SEG_SEANN,"Registered CONV : filters:" + filterSize.toString() +
                " input:" + shape4(filterSize.c,ih,iw).toString() + " output:" + a->dims.toString());
            logDebug(LOG_SEG_SEANN,"Current total memory occupation : " +
                to_string((double)MEMORY_OCCUPATION/(1024*1024)));
        }

        void forwardCalc(Tensor *prevA);

        void backwardCalc(Tensor *prevZ, Tensor *prevError) const;

        void recFilters(Tensor *prevA) const;

        void recBiases() const;

        void applyFilters(float LEARNING_RATE, uint32 BATCH_SIZE) const;

        void applyBiases(float LEARNING_RATE, uint32 BATCH_SIZE) const;

        void initialize() override;

        void forward(Layer *prev) override;

        void backward(Layer *prev) override;

        void backwardOut(Tensor *correct) override;

        void learn(float LEARNING_RATE, uint32 BATCH_SIZE) override;
    };
}

#endif //SEANN_CONVLAYER_CUH
