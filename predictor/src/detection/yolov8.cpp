//参考元 https://github.com/dacquaviva/yolov5-openvino-cpp-python/blob/main/cpp/main.cpp
#include "detection/yolov8.hpp"


namespace{
    void printInputAndOutputsInfo(const ov::Model& network) {
        std::cout << "model name: " << network.get_friendly_name() << std::endl;

        const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
        for (const ov::Output<const ov::Node>& input : inputs) {
            std::cout << "    inputs" << std::endl;

            const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
            std::cout << "        input name: " << name << std::endl;

            const ov::element::Type type = input.get_element_type();
            std::cout << "        input type: " << type << std::endl;

            const ov::Shape shape = input.get_shape();
            std::cout << "        input shape: " << shape << std::endl;
        }

        const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
        for (const ov::Output<const ov::Node>& output : outputs) {
            std::cout << "    outputs" << std::endl;

            const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
            std::cout << "        output name: " << name << std::endl;

            const ov::element::Type type = output.get_element_type();
            std::cout << "        output type: " << type << std::endl;

            const ov::Shape shape = output.get_shape();
            std::cout << "        output shape: " << shape << std::endl;
        }
    }

}


Yolov8::Yolov8(std::string &model_path, int image_size, float confidence_threshold, float score_threshold)
    : m_image_size(image_size)
    , m_confidence_threshold(confidence_threshold)
    , m_score_threshold(score_threshold)
{
    std::cout << "read model" << std::endl;
    model = std::shared_ptr<ov::Model>(this->core.read_model(model_path));
    std::cout << "PrePostProcessor model" << std::endl;
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    ppp.output().postprocess().convert_layout({0, 2, 1});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    printInputAndOutputsInfo(*model);
    // Load a model to the device
    std::cout << "compiled_model model" << std::endl;
    compiled_model = ov::CompiledModel(core.compile_model(model, "CPU"));
}

std::vector<float> Yolov8::infer(cv::Mat &image)
{
    // preprcess and infer
    Preprocess input(image, cv::Size(this->m_image_size, this->m_image_size));
    float *input_data = (float *) input.resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    this->infer_request = ov::InferRequest(compiled_model.create_infer_request());
    this->infer_request.set_input_tensor(input_tensor);
    this->infer_request.infer();

    const ov::Tensor &output_tensor = this->infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();

    // Step 8. Postprocessing including NMS  
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]];
        float *classes_scores = &detection[4];
        cv::Mat scores(1, output_shape[2] - 4, CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);        

        if (max_class_score > m_score_threshold){
            confidences.push_back(max_class_score);
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
    std::cout << " max conf" << confidences[idx] << " class" << class_ids[idx] << std::endl;
    std::cout << boxes[idx].x << " " << boxes[idx].y << " " << boxes[idx].x + boxes[idx].width << " " << boxes[idx].y + boxes[idx].height << std::endl;
    Xyxy result(boxes[idx].x, boxes[idx].y, boxes[idx].x + boxes[idx].width , boxes[idx].y + boxes[idx].height);
    std::vector<float> result_vec = result.normalized(m_image_size - input.dh, m_image_size - input.dw);
    return result_vec;
}

std::vector<float> Yolov8::precict(cv::Mat &image)
{   
    std::cout << "infer start" << std::endl;
    std::vector<float> result = infer(image);
    return result;
}