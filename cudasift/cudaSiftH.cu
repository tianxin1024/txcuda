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

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp) {
    TimerGPU timer(0);
    const int nd = NUM_SCALES + 3;
    int w = width * (scaleUp ? 2 : 1);
    int h = height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int sizeTmp = nd * h * p;
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = nullptr;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
#ifndef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocated time = \t %.2f ms \n\n", timer.read());
#endif
    return memoryTmp;
}


void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, 
        float thresh, float lowerScale, bool scaleUp, float *tempMemory) {
    TimerGPU timer(0);
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void **)&d_PointCounterAddr, d_PointCounter));
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int)));
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

    const int nd = NUM_SCALES + 3;
    int w = img.width * (scaleUp ? 2 : 1);
    int h = img.height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int size = h * p;
    int sizeTmp = nd * h * p;
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory) {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095)/4096 * sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocated time = \t %.2f ms \n\n", timer.read());
#endif
    }

    float *memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    lowImg.Allocated(width, height, iAlignUp(width, 128), false, memorySub);
    if (!scaleUp) {
        float kernel[8 * 12 * 16];
        PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
        LowPass(lowImg, img, max(initBlur, 0.001f));
        TimerGPU timer1(0);
        ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, meoryTmp, memorySub + height * iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
        printf("SIFT extraction time = \t %.2f ms %d\n", timer1.read(), siftData.numPts);
    } else {
        CudaImage upImg;

    }

    if (!tempMemory) {
        safeCall(cudaFree(memoryTmp));
    }
#ifdef MANAGEDMEN
    safeCall(cudaDeviceSynchronize());
#else
    if (siftData.h_data) {
        safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, cudaMemcpyDeviceToHost));
    }
#endif
    double totTime = timer.read();
    printf("Incl prefiltering & memcpy = \t %.2f ms %d\n\n", totTime, siftData.numPts);
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
