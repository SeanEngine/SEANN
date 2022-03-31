//
// Created by DanielSun on 2/24/2022.
//

#include "ImageReader.cuh"
#define RGB_DECAY 1.0f/256.0f

namespace seio{

    //TODO: Optimize this and prevent any malloc in loops
    Tensor* readRGB(const char *path, Tensor* reserved) {

        cv::Mat procImage = cv::imread(path, cv::IMREAD_COLOR);
        int h = procImage.size().height;
        int w = procImage.size().width;
        int smallDim = h < w ? h : w;
        procImage = procImage(cv::Range((h - smallDim) / 2, h - (h - smallDim) / 2),
                              cv::Range((w - smallDim) / 2, w - (w - smallDim) / 2));
        cv::Mat in;
        cv::resize(procImage, in, cv::Size((int)reserved->dims.rows, (int)reserved->dims.cols), cv::INTER_LINEAR);

        for(int c = 0; c < reserved->dims.c; c++){
            for(int i = 0; i < reserved->dims.rows; i++){
                for(int j = 0; j < reserved->dims.cols; j++){
                    reserved->elements[c * reserved->dims.rows * reserved->dims.cols + i * reserved->dims.cols + j] =
                            (float)in.at<cv::Vec3b>(i, j)[c] * RGB_DECAY;
                }
            }
        }

        return reserved;
    }

    void loadBinFile(const char *path, uchar *buffer, uint32 size) {
        FILE* file = fopen(path, "rb");
        fread(buffer, sizeof(uchar), size, file);
        fclose(file);
    }

    Tensor* readBinPixels(const uchar* target, Tensor* reserved){
        for (int i = 0; i < reserved->dims.size; i++) {
            reserved->elements[i] = (float)target[i] * RGB_DECAY;
        }
        return reserved;
    }
}
