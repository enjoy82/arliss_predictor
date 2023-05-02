#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "detection/preprocess.hpp"
#include "detection/xyxy.hpp"

class Yolov8: public PredictorBase
{
private:
    int image_size;
    float confidence_threshold;
    float score_threshold;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::preprocess::PrePostProcessor ppp;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    void infer(cv::Mat &image);
    std::vector<float> pull_result();
public:
    virtual std::vector<float> precict(cv::Mat &image) override;
    Yolov8(std::string &model_path, int image_size, float confidence_threshold, float score_threshold);
    ~Yolov8() = default;
};