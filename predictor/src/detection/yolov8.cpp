//参考元 https://github.com/dacquaviva/yolov5-openvino-cpp-python/blob/main/cpp/main.cpp
#pragma once
#include "detection/yolov8.hpp"


Yolov8::Yolov8(std::string &model_path, int image_size = 640, float confidence_threshold = 0.4, float score_threshold = 0.2)
    : this->image_size(image_size)
    , this->confidence_threshold(confidence_threshold)
    , this->score_threshold(score_threshold)
{
    std::cout << "read model" << std::endl;
    std::shared_ptr<ov::Model> model(this->core.read_model(model_path));
    std::cout << "PrePostProcessor model" << std::endl;
    ov::preprocess::PrePostProcessor ppp(ov::preprocess::PrePostProcessor(model));
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();

    // Load a model to the device
    std::cout << "compiled_model model" << std::endl;
    ov::CompiledModel compiled_model(core.compile_model(model, "CPU"));
    // Create an infer request
    std::cout << "infer_request" << std::endl;
    ov::InferRequest infer_request(compiled_model.create_infer_request());
}

void Yolov8::infer(cv::Mat &image)
{
    Resize res = resize_and_pad(image, cv::Size(this->image_size, this->image_size));
    float *input_data = (float *) res.resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    this->infer_request.set_input_tensor(input_tensor);
    this->infer_request.infer();
}

std::vector<float> Yolov8::pull_result()
{
    // TODO v8に合わせて取り出し
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();

    // Step 8. Postprocessing including NMS  
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]];

        float confidence = detection[4];
        if (confidence >= confidence_threshold){
            float *classes_scores = &detection[5];
            cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > score_threshold){

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                float xmin = x - (w / 2);
                float ymin = y - (h / 2);

                boxes.push_back(cv::Rect(xmin, ymin, w, h));
            }
        }
    }

    // extract max score
    float max_confidence = -1.0;
    int idx = -1;
    for(int i = 0; i < boxes.size(); i++){
        if(confidences[i] > max_confidence){
            max_confidence = confidences[i];
            idx = i;
        }
    }
    
    if(idx == -1){
        std::vector<float> result = {0.0, 0.0, 0.0, 0.0};
        return result;
    }
    Xyxy result(boxes[idx].x, boxes[idx].y, boxes[idx].x + boxes[idx].w, boxes[idx].y + boxes[idx].h);
    std::vector<float> result_vec = result.normalized(boxes[idx].h, boxes[idx].w);
    return result_vec;
}

Yolov8::precict(cv::Mat &image)
{   
    std::cout << "infer start" << std::endl;
    infer(image)
    std::cout << "pull_result start" << std::endl;
    std::vector<float> result = pull_result();
    return result;
}