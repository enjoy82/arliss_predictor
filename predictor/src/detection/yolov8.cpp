
#include "detection/yolov8.hpp"

Yolov8::Yolov8(std::string &model_path)
{
    std::cout << "read model" << std::endl;
    std::shared_ptr<ov::Model> model(this->core.read_model(model_path));
    std::cout << "PrePostProcessor model" << std::endl;
    ov::preprocess::PrePostProcessor ppp(ov::preprocess::PrePostProcessor(model));
    // TODO preprocess実装  
    model = ppp.build();
    std::cout << "compiled_model model" << std::endl;
    ov::CompiledModel compiled_model(core.compile_model(model, "MYRIAD"));
    std::cout << "infer_request" << std::endl;
    ov::InferRequest infer_request(compiled_model.create_infer_request());
}

void Yolov8::infer(cv::Mat &image)
{
    // TODO inferするためのtensor化
    this->infer_request.set_input_tensor(input_tensor);
    this->infer_request.infer();
}

void Yolov8::pull_result()
{
    // TODO v8に合わせて取り出し
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();
}


Yolov8::precict(cv::Mat &image)
{   
    std::cout << "infer start" << std::endl;
    infer(cv::Mat &image)
    std::cout << "pull_result start" << std::endl;
    pull_result();
}