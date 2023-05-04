#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>

#include "detection/ar_predictor.hpp"

ArPredictor::ArPredictor()
{
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
}

std::vector<float> ArPredictor::point2f_to_vec(cv::Mat &image, std::vector<std::vector<cv::Point2f>> &corners)
{
    if(corners.size() == 0){
        std::vector<float> result_vec = {0.0, 0.0, 0.0, 0.0};
        return result_vec;
    }
    Xyxy result(corners[0][0].x, corners[0][0].y, corners[0][2].x, corners[0][2].y);
    int height = image.rows;
    int width = image.cols;
    std::vector<float> result_vec = result.normalized(height, width);
    return result_vec;
}

std::vector<float> ArPredictor::precict(cv::Mat &image)
{
    std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
    std::vector<int> ids;
    // ArUcoマーカの検出
    detector.detectMarkers(image, corners, ids, rejectedCandidates);
    std::vector<float> result = point2f_to_vec(image, corners);
    return result;
}