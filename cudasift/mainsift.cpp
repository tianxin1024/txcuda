#include <iostream>
#include <opencv2/opencv.hpp>
#include "cudaSift.h"
#include "cudaImage.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

    int devNum = 0, imgSet = 0;
    if (argc > 1) {
        devNum = std::atoi(argv[1]);
    }

    if (argc > 2) {
        imgSet = std::atoi(argv[2]);
    }

    // Read images using OpenCV
    cv::Mat limg, rimg;
    if (imgSet) {
        cv::imread("../data/left.pgm", 0).convertTo(limg, CV_32FC1);
        cv::imread("../data/right.pgm", 0).convertTo(rimg, CV_32FC1);
    } else {
        cv::imread("../data/img1.png", 0).convertTo(limg, CV_32FC1);
        cv::imread("../data/img2.png", 0).convertTo(rimg, CV_32FC1);
    }

    unsigned int w = limg.cols;
    unsigned int h = limg.rows;
    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    // Initial Cuda images and download images to device
    std::cout << " Initializing data ... " << std::endl;
    InitCuda(devNum);
    CudaImage img1, img2;
    img1.Allocate(w, h, iAlignUp(w, 128), false, nullptr, (float*)limg.data);
    img2.Allocate(w, h, iAlignUp(w, 128), false, nullptr, (float*)rimg.data);
    img1.Download();
    img2.Download();

    // Extract Sift features from images
    SiftData siftData1, siftData2;
    float initBlur = 1.0f;
    float thresh = (imgSet ? 4.5f : 3.0f);
    InitSiftData(siftData1, 32768, true, true);
    InitSiftData(siftData2, 32768, true, true);

    // A bit of benchmarking
    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    for (int i = 0; i < 1000; i++) {
        ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmp);
        cout << "----------------------" << endl;
        ExtractSift(siftData1, img2, 5, initBlur, thresh, 0.0f, false, memoryTmp);
    }
    // FreeSiftTempMemory(memoryTmp);

    return 0;
}
