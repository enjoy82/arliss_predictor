#pragma once
#include <opencv2/opencv.hpp>

#include "detection/predictor_base.hpp"

class ArPredictor: public PredictorBase
{
private:
    cv::aruco::ArucoDetector detector;
    
    std::vector<float> point2f_to_vec(cv::Mat &image, std::vector<std::vector<cv::Point2f>> &corners);
public:
    virtual std::vector<float> precict(cv::Mat &image) override;
    ArPredictor();
    ~ArPredictor() = default;
};