#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "detection/predictor_base.hpp"
#include "detection/preprocess.hpp"
#include "detection/xyxy.hpp"

class Yolov8: public PredictorBase
{
private:
    int m_image_size;
    float m_confidence_threshold;
    float m_score_threshold;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    std::vector<float> infer(cv::Mat &image);
public:
    virtual std::vector<float> precict(cv::Mat &image) override;
    Yolov8(std::string &model_path, int image_size = 640, float confidence_threshold = 0.4, float score_threshold = 0.2);
    ~Yolov8() = default;
};