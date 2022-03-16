#include <iostream>
#include "cudaSift.cuh"


void InitCuda(int devNum) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices) {
        std::cerr << "No CUDA devices available" << std::endl;
        return ;
    }

    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
    std::cout << "Device Number: " << devNum << std::endl;

    return ;
}

