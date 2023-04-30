#pragma once
#include <opencv2/opencv.hpp>

#include "detection/xyxy.hpp"

class PredictorBase
{
    public:
        PredictorBase() = default;
        ~PredictorBase() = default;
    virtual std::vector<float> precict(cv::Mat &image){
        std::vector<float> xyxy = {1, 1, 1, 1};
        return xyxy;
    }
};