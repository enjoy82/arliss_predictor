#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class Yolov8: public PredictorBase
{
private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::preprocess::PrePostProcessor ppp;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    void infer(cv::Mat &image);
    void pull_result();
public:
    virtual std::vector<float> precict(cv::Mat &image) override;
    Yolov8(std::string &model_path);
    ~Yolov8() = default;
};