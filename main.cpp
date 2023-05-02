#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "detection/xyxy.hpp"
#include "detection/predictor_base.hpp"
#include "detection/ar_predictor.hpp"
#include "detection/yolov8.hpp"

namespace{
    std::string model_path = "./model/test.onnx";
};

void draw_result(cv::Mat &frame, std::vector<float>  &xyxy)
{
    int height = frame.rows;
    int width = frame.cols;
    cv::Point left_top = cv::Point(static_cast<int>(xyxy[0] * width), static_cast<int>(xyxy[1] * height));
    cv::Point right_bottom = cv::Point(static_cast<int>(xyxy[2] * width), static_cast<int>(xyxy[3] * height));
    cv::rectangle(frame, left_top, right_bottom, cv::Scalar(0,0,255), 5);
    return;
}

int main(){
    ArPredictor ar_predictor;
    Yolov8 yolo_predictor(model_path);

    int predictor_mode = 1;
    cv::VideoCapture cap(0);//デバイスのオープン
    if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        //読み込みに失敗したときの処理
        return -1;
    }

    cv::Mat frame; //取得したフレーム
    while(cap.read(frame))//無限ループ
    {
        std::vector<float> result;
        if(predictor_mode == 1)
            result = ar_predictor.precict(frame);
        else
            result = yolo_predictor(frame);

        draw_result(frame, result);
        cv::imshow("predictor", frame);//画像を表示．
        const int key = cv::waitKey(1);
        if(key == 'q'/*113*/)//qボタンが押されたとき
        {
            break;//whileループから抜ける．
        }
        if(key == 's'/*113*/)//qボタンが押されたとき
        {
            predictor_mode *= -1;
        }
    }
    cv::destroyAllWindows();
    return 0;
}