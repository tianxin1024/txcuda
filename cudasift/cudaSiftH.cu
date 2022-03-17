#include <iostream>
#include "cudaSift.h"
#include "cudautils.h"


void InitCuda(int devNum) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices) {
        std::cerr << "No CUDA devices available" << std::endl;
        return ;
    }

    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
    std::cout << "\t Device Number: " << devNum << std::endl;
    std::cout << "\t Memory name: " << prop.name << std::endl;
    std::cout << "\t Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    printf("\t Peak Memory Bandwidth (GB/s): %.1f\n\n",
	            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    return ;
}

void InitSiftData(SiftData &data, int num, bool host, bool dev) {
    data.numPts = 0;
    data.maxPts = num;
    int sz = sizeof(SiftPoint) * num;
#ifdef MANAGEDMEN
    safeCall(cudaMallocManaged((void **) &data.m_data, sz));
#else
    data.h_data = nullptr;
    if (host) {
        data.h_data = (SiftPoint *)malloc(sz);
    }
    data.d_data = nullptr;
    if (dev) {
        safeCall(cudaMalloc((void **)&data.d_data, sz));
    }
#endif
}
