#include <iostream>
#include <opencv2/opencv.hpp>
#include "warpAffine.cu"

using namespace std;
using namespace cv;


// gpu 实现的函数声明
Mat warp_affine_gpu(const Mat& image, const Mat& M, const Size& size);

int main() {

    cv::Mat image = cv::imread("../demo.jpg");
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(image.cols * 0.5, image.rows * 0.5), 30, 0.85);
    cv::Mat affine;

    cv::warpAffine(image, affine, M, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 128, 255));

    cv::imwrite("affine_cpu.jpg", affine);

    affine = warp_affine_gpu(image, M, image.size());
    cv::imwrite("affine_gpu.jpg", affine);

    return 0;
}
