#pragma once
#include <opencv2/opencv.hpp>

struct Preprocess
{
    cv::Mat resized_image;
    int dw;
    int dh;

    Preprocess(cv::Mat& img, cv::Size new_shape) {
        float width = img.cols;
        float height = img.rows;
        float r = float(new_shape.width / std::max(width, height));
        int new_unpadW = int(round(width * r));
        int new_unpadH = int(round(height * r));
        cv::Mat resized_image;
        cv::resize(img, resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

        this->dw = new_shape.width - new_unpadW;
        this->dh = new_shape.height - new_unpadH; 
        cv::Scalar color = cv::Scalar(100, 100, 100);
        cv::copyMakeBorder(resized_image, resized_image, 0, this->dh, 0, this->dw, cv::BORDER_CONSTANT, color);
        this->resized_image = resized_image;
    }
};